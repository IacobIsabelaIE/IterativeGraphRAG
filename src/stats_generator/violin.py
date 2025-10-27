import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "../experiments/experiment_optimized_real_final_ner_500.csv"
OUTPUT_DIR = "../statistics"
OUTPUT_FILE = "violin_ner.png"

METRICS = [
    "faithfulness",
    "completeness",
    "relevance",
    "BERTScore",
    "rougeL",
]

def main():
    # Load
    df = pd.read_csv(INPUT_FILE)
    df = df.rename(columns={"bertscore_f1": "BERTScore"})

    # Collect clean arrays per metric
    data = []
    labels = []
    for m in METRICS:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors="coerce").dropna()
            if len(col) > 0:
                data.append(col.values)
                labels.append(m)

    if not data:
        raise ValueError("No valid metric columns found with numeric data.")

    # Figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(14, 7))

    parts = plt.violinplot(
        data,
        showmeans=False,
        showmedians=True,   # highlight medians
        showextrema=False
    )

    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Add median markers styling if present
    if 'cmedians' in parts:
        parts['cmedians'].set_linewidth(2)

    # Axes labels & ticks
    plt.xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Score density distribution evaluation metrics for KGiRAG with NER")

    # Grid & layout
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(out_path, dpi=300)


if __name__ == "__main__":
    main()
