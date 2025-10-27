# IterativeGraphRAG

**KGiRAG** is an enhanced Retrieval-Augmented Generation (RAG) framework that integrates a **preprocessed Knowledge Graph (KG)** with **iterative, feedback-driven retrieval** and **quality-aware answer generation**.

Unlike traditional RAG systems that rely on static context retrieval, **KGiRAG dynamically expands and refines the context** based on answer quality-enabling **adaptive, context-aware reasoning** over complex knowledge spaces with minimal wasted compute.

---

##  Prerequisites

| **Library**                          |                                                                  
| ------------------------------------ |
| `lancedb`                            |
| `spacy`, `transformers`, `langchain` |
| `ms_graphrag_bridge`                 | 
| `bert_score`, `nltk` (BLEU)          |
| `openai`, `anthropic` clients        | 

>  **Recommended Python version: 3.10** 

---

##  Get Started

### 1. Clone the repository

```bash
git clone https://github.com/IacobIsabelaIE/IterativeGraphRAG.git
cd IterativeGraphRAG/src
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Add your **API keys / credentials** inside the `configuration.py` file  

---

##  Run IterativeGraphRAG (with NER enabled)

```bash
python main_with_ner.py
```

---

## Results for KGiRAG compared to other architectures

| **Metric**            | **KGiRAG (with NER)** | **KGiRAG (no NER)**     | **Microsoft GraphRAG** |
|-----------------------|------------------------|--------------------------|--------------------------|
| Faithfulness          | 0.86 ± 0.03075          | 0.49 ± 0.04305            | 0.95 ± 0.01757            |
| Completeness          | 0.46 ± 0.04305          | 0.39 ± 0.03954            | 0.62 ± 0.03954            |
| Relevance             | 0.43 ± 0.04305          | 0.54 ± 0.04218            | 0.23 ± 0.03515            |
| **BERTScore**         | 0.79 ± 0.001757         | 0.80 ± 0.001757           | 0.77 ± 0.001757           |
| ROUGE-L               | 0.03 ± 0.002636         | 0.05 ± 0.002636           | 0.10 ± 0.006151           |
| BLEU                  | 5.29 ± 0.0008787        | 6.16 ± 0.002636           | 0.02 ± 0.004393           |
| Cosine Similarity     | 0.23 ± 0.02548          | 0.48 ± 0.01757            | 0.08 ± 0.01845            |

**Table 1.** KGiRAG retrieval performance comparison.


