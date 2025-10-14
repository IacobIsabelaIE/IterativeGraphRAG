from sentence_transformers import SentenceTransformer, util
import os
from rouge_score import rouge_scorer
from mistralai import Mistral

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "Z7fOWcOPo91uJFHFsLMX4NmHkrHzkGcL")
client = Mistral(api_key=MISTRAL_API_KEY)
judge_model = "open-mistral-7b"  
initial_k = 50
initial_k_text = 50
initial_k_comm = 10
initial_max_neighbors = 30