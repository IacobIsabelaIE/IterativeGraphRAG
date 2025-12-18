import pandas as pd
import numpy as np

INPUT_CSV = "../experiments/experiment_optimized_real_final_ner_500.csv"
OUTPUT_CSV = "../stats_camera_ready/ner.csv"

# Metrics to summarize
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

# For each question_id, determine its max iteration (how far it progressed)
max_iter_per_q = df.groupby("question_id")["iteration"].max()

def snapshot_at_k(k: int) -> pd.DataFrame:
    """
    For each question_id, pick the row at iteration = min(max_iteration_for_question, k).
    This matches:
      - table k includes iteration k rows
      - plus questions that never reached k, using their highest iteration below k
    """
    target_iter = max_iter_per_q.clip(upper=k).rename("target_iter").reset_index()

    merged = df.merge(target_iter, on="question_id", how="inner")
    snap = merged[merged["iteration"] == merged["target_iter"]].copy()

    # If there can be duplicates per (question_id, iteration), keep one deterministically. Prefer the last occurrence in file order
    snap = snap.sort_index().drop_duplicates(subset=["question_id", "iteration"], keep="last")
    return snap

def summarize_metrics(snap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        vals = pd.to_numeric(snap[m], errors="coerce")
        rows.append(
            {
                "metric": m,
                "mean": float(vals.mean(skipna=True)),
                "std": float(vals.std(skipna=True, ddof=1)),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean", "std"])

# Build CSV
blocks = []
for k in [1, 2, 3, 4]:
    snap = snapshot_at_k(k)
    summary = summarize_metrics(snap)
    title_row = pd.DataFrame(
        [{"metric": f"TABLE {k}: snapshot at iteration {k} (use min(max_iter, {k}))", "mean": "", "std": ""}],
        columns=["metric", "mean", "std"],
    )
    header_row = pd.DataFrame([{"metric": "metric", "mean": "mean", "std": "std"}], columns=["metric", "mean", "std"])
    blank_row = pd.DataFrame([{"metric": "", "mean": "", "std": ""}], columns=["metric", "mean", "std"])

    blocks.append(pd.concat([title_row, header_row, summary, blank_row], ignore_index=True))

out = pd.concat(blocks, ignore_index=True)
out.to_csv(OUTPUT_CSV, index=False, header=False)

print(f"Wrote: {OUTPUT_CSV}")
