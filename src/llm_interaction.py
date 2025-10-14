import os
import csv
import time
import random
import json
import re
import pandas as pd
from ms_graphrag_bridge import MSGraphRAG
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from configuration import *
import spacy


def safe_chat_complete(messages, model, temperature=0.0, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            time.sleep(1.2)
            return response
        except Exception as e:
            if "429" in str(e) or "capacity_exceeded" in str(e).lower():
                wait_time = 2 ** attempt + random.random()
                print(f"[WARN] Rate limit hit for {model}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Max retries reached for model {model}")

def call_llm(question: str, context: str) -> str:
    prompt = f"""
You are a helpful question-answering assistant.

Question: {question}

Context:
{context}

Instructions:
- Use the information provided in the context to answer the question.
- **Pay special attention to the 'Key Relationships' section** which contains structured facts from the knowledge graph.
- Be direct and concise in your answer.
- If the context truly lacks the information needed to answer, say: "The context does not provide enough information to answer this question."
- Do not fabricate information that is not supported by the context.

Answer:
"""
    response = safe_chat_complete(
        model="open-mistral-7b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def judge_answer_metrics(answer: str, context: str) -> dict:

    JUDGE_MODEL = judge_model 

    refusal_patterns = [
        "the context does not provide this information",
        "not enough information",
        "cannot answer based on the context",
        "no sufficient information",
        "insufficient context",
    ]
    if any(p.lower() in answer.lower() for p in refusal_patterns):
        return {"faithfulness": 1.0, "completeness": 0.0, "relevance": 0.0}

    prompt = f"""
You are an expert evaluator that scores a generated answer based on how well it matches the given evidence.

Answer:
{answer}

Evidence:
{context}

You must return ONLY a valid JSON object with exactly these three keys:
faithfulness, completeness, relevance

Scoring rules:
- faithfulness: 1.0 if the answer is fully supported by the evidence, 0.0 if it contains contradictions or hallucinations.
- completeness: 1.0 if the answer covers all information the evidence provides that is relevant to the question; 0.0 if it omits key evidence.
- relevance: 1.0 if the answer directly addresses the question and uses the context appropriately; 0.0 if it is off-topic.

All values must be floats between 0.0 and 1.0.

Output only JSON. No explanation or extra text.
"""
    response = safe_chat_complete(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed
        except Exception:
            pass
    return {"faithfulness": 0.0, "completeness": 0.0, "relevance": 0.0}

