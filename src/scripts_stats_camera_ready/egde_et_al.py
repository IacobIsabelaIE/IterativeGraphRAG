import pandas as pd

INPUT_CSV = "../experiments/rarr_metrics_final_review.csv"
OUTPUT_CSV = "../stats_per_iteration_csv/rarr.csv"

METRICS = [
    "faithfulness",
    "completeness",
    "relevance",
    "bertscore_f1",
    "rougeL",
    "bleu",
    "cosine_sim",
]

df = pd.read_csv(INPUT_CSV)

# Ensure metrics are numeric
df[METRICS] = df[METRICS].apply(pd.to_numeric, errors="coerce")

summary = (
    df[METRICS]
    .agg(["mean", "std"])
    .T
    .reset_index()
    .rename(columns={"index": "metric"})
)

summary.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote: {OUTPUT_CSV}")
