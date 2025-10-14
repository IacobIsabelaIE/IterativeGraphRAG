import os
import csv
import pandas as pd
from ms_graphrag_bridge import MSGraphRAG
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from configuration import * 
from llm_interaction import *


def compute_similarity_metrics(answer: str, context: str) -> dict:
    P, R, F1 = bertscore([answer], [context], lang="en", verbose=False)
    rouge_l = rouge.score(context, answer)["rougeL"].fmeasure
    ref_tokens = [context.split()]
    cand_tokens = answer.split()
    bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SmoothingFunction().method7)
    emb_answer = sbert_model.encode(answer, convert_to_tensor=True)
    emb_context = sbert_model.encode(context, convert_to_tensor=True)
    cosine_sim = float(util.cos_sim(emb_answer, emb_context).item())
    return {
        "bertscore_f1": float(F1.mean().item()),
        "rougeL": rouge_l,
        "bleu": bleu,
        "cosine_sim": cosine_sim,
    }


def log_qa_interaction(log_data: dict, log_path: str):
    dir_name = os.path.dirname(log_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys(), quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow(log_data)


def extract_initial_entities(question: str, rag, top_k=50) -> list:
    """
    NER-free version:
    Uses only LanceDB similarity search to find relevant entity candidates.
    """
    print("  [NO NER MODE] Using retrieval-based entity initialization only.")
    initial_ids = set()

    evidence = rag.retrieve_text_units(question, k_text=initial_k_text, k_comm=initial_k_comm)
    for e in evidence:
        meta = e.get("meta", {})
        ent_id = meta.get("entity_id") or meta.get("id") or e.get("id")
        if isinstance(ent_id, list):
            for ei in ent_id:
                if ei and ei in rag.G:
                    initial_ids.add(ei)
        elif ent_id and ent_id in rag.G:
            initial_ids.add(ent_id)

    print(f"  Found {len(initial_ids)} candidate entities via retrieval only.")
    return list(initial_ids)[:top_k]


def get_relevant_neighbors(rag, entity_id, question_emb, max_neighbors=30):
    """
    Get neighbors and score them by semantic similarity to the question.
    Adjusted threshold to be more permissive (0.1) to catch indirectly related facts.
    """
    if entity_id not in rag.G:
        return []

    neighbors = list(rag.G.neighbors(entity_id))
    if not neighbors:
        return []

    neighbor_contexts = []
    neighbor_ids = []

    for neighbor in neighbors:
        context = rag.get_entity_context(neighbor)
        if context:
            neighbor_contexts.append(context)
            neighbor_ids.append(neighbor)

    if not neighbor_contexts:
        return []

    c_embs = sbert_model.encode(neighbor_contexts, convert_to_tensor=True)
    scores = util.cos_sim(question_emb, c_embs)[0]
    ranked_indices = scores.argsort(descending=True)
    top_neighbors = [neighbor_ids[i.item()] for i in ranked_indices if scores[i.item()] > 0.1]

    return top_neighbors[:max_neighbors]


def rerank_and_format_context(chunks, question, top_k=20, max_chars=4000):
    """Rerank chunks, remove duplicates, and format into structured context."""
    if not chunks:
        return ""

    unique_chunks = list(set(chunks))
    relationship_chunks = [c for c in unique_chunks if "—" in c]
    text_unit_chunks = [c for c in unique_chunks if "—" not in c]

    q_emb = sbert_model.encode(question, convert_to_tensor=True)

    if text_unit_chunks:
        c_embs = sbert_model.encode(text_unit_chunks, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, c_embs)[0]
        ranked_idx = scores.argsort(descending=True)
        ranked_text_units = [text_unit_chunks[i] for i in ranked_idx]
    else:
        ranked_text_units = []

    final_chunks = relationship_chunks + ranked_text_units
    final_chunks = final_chunks[:top_k]

    context_parts = []
    if relationship_chunks:
        context_parts.append("## Key Relationships (Structured Facts):")
        context_parts.append("\n".join(relationship_chunks))

    if ranked_text_units:
        context_parts.append("\n\n## Relevant Text Units:")
        context_parts.append("\n".join(ranked_text_units))

    combined = "\n".join(context_parts)
    return combined[:max_chars]


def run_qags_graph_loop(questions_csv, graph_output_dir, output_csv, num_questions=3):
    """
    NER-free version of QAGS Graph Loop.
    """
    rag = MSGraphRAG(output_dir=graph_output_dir, verbose=True)
    df = pd.read_csv(questions_csv).drop_duplicates(subset=["question_text"]).head(num_questions)

    question_embeddings = {
        row["question_id"]: sbert_model.encode(row["question_text"], convert_to_tensor=True)
        for _, row in df.iterrows()
    }

    for idx, row in df.iterrows():
        qid = row["question_id"]
        question = row["question_text"]
        q_emb = question_embeddings[qid]
        print(f"\n{'='*60}")
        print(f"Processing question {idx+1}/{len(df)}: QID={qid}")
        print(f"{'='*60}")

        # 🔹 Fără NER
        entity_ids = extract_initial_entities(question, rag, top_k=25)
        print(f"[START] QID={qid} | Found {len(entity_ids)} candidate entities")

        max_iters = 4
        visited_entities = set()
        all_context_chunks = []

        current_k_text = initial_k_text
        current_k_comm = initial_k_comm
        current_max_neighbors = initial_max_neighbors

        for ent_id in entity_ids:
            visited_entities.add(ent_id)

        for i in range(max_iters):
            print(f"\n[ITER {i+1}] QID={qid}")

            iteration_chunks = []
            newly_visited_in_iter = set()

            if i > 0:
                current_k_text += 100
                current_k_comm += 5
                current_max_neighbors += 10

            k_text = current_k_text
            k_comm = current_k_comm
            max_neighbors = current_max_neighbors

            print(f"  Params: k_text={k_text}, k_comm={k_comm}, max_neighbors={max_neighbors}")

            if i == 0:
                print(f"  Getting context for {len(entity_ids)} initial entities...")
                for entity_id in entity_ids:
                    try:
                        context_text = rag.get_entity_context(entity_id)
                        if context_text.strip():
                            iteration_chunks.append(context_text)
                    except Exception as e:
                        if rag.verbose:
                            print(f"  Warning: Could not get context for {entity_id}: {e}")

                evidence = rag.retrieve_text_units(question, k_text=k_text, k_comm=k_comm)
                for e in evidence:
                    text = e.get("text", "")
                    if text and text.strip():
                        iteration_chunks.append(text)

            else:
                print(f"  Expanding from {len(visited_entities)} entities...")
                new_entities_to_check = set()

                for ent in list(visited_entities):
                    relevant_neighbors = get_relevant_neighbors(rag, ent, q_emb, max_neighbors=max_neighbors)
                    new_entities_to_check.update(relevant_neighbors)

                for ent in new_entities_to_check:
                    if ent not in visited_entities:
                        newly_visited_in_iter.add(ent)
                        try:
                            context_text = rag.get_entity_context(ent)
                            if context_text.strip():
                                iteration_chunks.append(context_text)
                        except Exception as e:
                            if rag.verbose:
                                print(f"  Warning: Could not get context for {ent}: {e}")

                visited_entities.update(newly_visited_in_iter)

                evidence = rag.retrieve_text_units(question, k_text=k_text, k_comm=k_comm)
                for e in evidence:
                    text = e.get("text", "")
                    if text and text.strip():
                        iteration_chunks.append(text)

                print(f"  Expanded to {len(visited_entities)} total entities, added {len(newly_visited_in_iter)} new entity contexts and {len(evidence)} text/community units.")

            all_context_chunks.extend(iteration_chunks)

            top_k_rerank = 15 + i * 10
            max_chars = 4000 + i * 1000

            context = rerank_and_format_context(
                all_context_chunks,
                question,
                top_k=top_k_rerank,
                max_chars=max_chars
            )

            print(f"  Final Context: {len(context)} chars (top_k={top_k_rerank}, max={max_chars})")

            answer = call_llm(question, context)
            scores = judge_answer_metrics(answer, context)
            extra_scores = compute_similarity_metrics(answer, context)

            log_data = {
                "question_id": qid,
                "iteration": i + 1,
                "question": question,
                "answer": answer,
                "context": context,
                "faithfulness": scores.get("faithfulness", 0.0),
                "completeness": scores.get("completeness", 0.0),
                "relevance": scores.get("relevance", 0.0),
                "bertscore_f1": extra_scores["bertscore_f1"],
                "rougeL": extra_scores["rougeL"],
                "bleu": extra_scores["bleu"],
                "cosine_sim": extra_scores["cosine_sim"],
                "mode": "no_NER",
            }
            log_qa_interaction(log_data, output_csv)

            print(f"  Scores - F:{log_data['faithfulness']:.2f} C:{log_data['completeness']:.2f} R:{log_data['relevance']:.2f}")

            if log_data["faithfulness"] >= 0.8 and log_data["completeness"] >= 0.8:
                print(f"✅ QID {qid} converged at iteration {i+1}")
                break

            if i >= max_iters - 1:
                print(f"⚠️ QID {qid} reached max iterations without convergence")
                break

        print(f"\n[DONE] QID {qid} finished after {i+1} iterations")
