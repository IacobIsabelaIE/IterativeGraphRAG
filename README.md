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

| **Metric**         | **KGiRAG (with NER)** | **KGiRAG (no NER)** | **Microsoft GraphRAG** | **RARR**          |
|--------------------|------------------------|----------------------|-------------------------|-------------------|
| Faithfulness       | 0.95 ± 0.01881          | 0.78 ± 0.03613        | 0.95 ± 0.01749           | 0.90 ± 0.02615     |
| Completeness       | 0.86 ± 0.029802         | 0.69 ± 0.03663        | 0.62 ± 0.0396            | 0.41 ± 0.04145     |
| Relevance          | 0.64 ± 0.042269         | 0.76 ± 0.0371         | 0.23 ± 0.03549           | 0.59 ± 0.04282     |
| **BERTScore**      | 0.79 ± 0.001603         | 0.81 ± 0.001423       | 0.77 ± 0.001551          | 0.797 ± 0.0021     |
| Cosine Similarity  | 0.325 ± 0.02692         | 0.52 ± 0.01828        | 0.075 ± 0.01831          | 0.33 ± 0.02427     |

**Table 1.** Comparison of retrieval architectures across evaluation metrics (mean ± margin of error for the 95% confidence intervals).
