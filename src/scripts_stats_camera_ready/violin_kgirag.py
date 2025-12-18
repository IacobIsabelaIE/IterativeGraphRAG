import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "../experiments/experiment_optimized_real_final_ner_500.csv"
OUT_PNG = "../stats_camera_ready/violin_ner.png"

# Metrics to plot (match your QA CSV column names)
METRICS = [
    "faithfulness",
    "completeness",
    "relevance",
    "bertscore_f1",
    "rougeL"
]

# Display labels (optional)
LABELS = {
    "faithfulness": "faithfulness",
    "completeness": "completeness",
    "relevance": "relevance",
    "bertscore_f1": "BERTScore",
    "rougeL": "rougeL"
}

df = pd.read_csv(INPUT_CSV)

# Select ONLY the highest-iteration row per question_id
# (If duplicates exist at max iteration, keep the last occurrence deterministically)
df_sorted = df.sort_values(["question_id", "iteration"]).copy()
max_iter = df_sorted.groupby("question_id")["iteration"].transform("max")
final_rows = df_sorted[df_sorted["iteration"] == max_iter]
final_rows = final_rows.sort_index().drop_duplicates(subset=["question_id"], keep="last")

# Build data arrays for violinplot
data = []
for m in METRICS:
    vals = pd.to_numeric(final_rows[m], errors="coerce").dropna()
    data.append(vals.to_numpy())

# Plot
plt.figure(figsize=(12, 5))
vp = plt.violinplot(
    data,
    showmeans=True,
    showmedians=False,
    showextrema=False,
)

for pc in vp['bodies']:
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

plt.xticks(
    np.arange(1, len(METRICS) + 1),
    [LABELS[m] for m in METRICS],
    rotation=25,
    ha="right",
)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Score density distribution evaluation metrics for KGiRAG")

plt.grid(True, axis="y", alpha=0.4, linestyle="--")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)

print(f"Questions plotted: {final_rows['question_id'].nunique()}")
print(f"Saved: {OUT_PNG}")
