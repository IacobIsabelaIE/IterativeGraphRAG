from llm_interaction import *
from pathlib import Path
from graph_search_no_ner import *

if __name__ == "__main__":
    QUESTIONS_CSV = "questions.csv"
    GRAPH_OUTPUT_DIR = "output"
    OUTPUT_DIR = Path("experiments")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV = OUTPUT_DIR / "experiment_optimized_real_final_no_ner_500.csv"
    run_qags_graph_loop(QUESTIONS_CSV, GRAPH_OUTPUT_DIR, OUTPUT_CSV, num_questions=470)
