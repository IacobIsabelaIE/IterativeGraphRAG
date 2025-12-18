import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_CSV = "../stats_camera_ready/mean_no_ner.csv"
STOPPED_CSV = "../stats_camera_ready/q_stopped_at_iteration_no_ner.csv"
OUT_PNG = "../stats_camera_ready/graph_metrics_no_ner_all.png"

METRICS_WANTED = ["faithfulness", "completeness", "relevance", "bertscore_f1", "rougeL", "bleu", "cosine_sim"]
LABELS = {
    "faithfulness": "Faithfullness",
    "completeness": "Completeness",
    "relevance": "Relevance",
    "bertscore_f1": "BertScore",
    "rougeL": "ROUGE-L",
    "bleu": "BLEU",
    "cosine_sim": "Cosine Similarity"
}


def parse_combined_summary_csv(path: str, metrics=METRICS_WANTED) -> pd.DataFrame:
    # Read as raw rows with 3 columns: metric, mean, std
    raw = pd.read_csv(path, header=None, names=["c0", "c1", "c2"], dtype=str).fillna("")
    current_k = None
    rows = []

    for _, r in raw.iterrows():
        c0 = r["c0"].strip()
        c1 = r["c1"].strip()

        # Detect table title row like: "TABLE 2: snapshot at iteration 2 ..."
        if c0.startswith("TABLE "):
            # Extract table number after "TABLE "
            try:
                current_k = int(c0.split()[1].replace(":", ""))
            except Exception:
                current_k = None
            continue

        if current_k in (1, 2, 3, 4) and c0 in metrics:
            try:
                mean_val = float(c1)
            except Exception:
                mean_val = np.nan
            rows.append({"iteration": current_k, "metric": c0, "mean": mean_val})

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No metric means were parsed from the summary CSV. Check the file format/path.")
    return out

# 1) Read metric means per iteration snapshot
means_long = parse_combined_summary_csv(SUMMARY_CSV, metrics=METRICS_WANTED)
means_wide = (
    means_long.pivot(index="iteration", columns="metric", values="mean")
    .reindex([1, 2, 3, 4])
)

# 2) Read stopped counts and compute cumulative stop rate
stopped = pd.read_csv(STOPPED_CSV)
stopped["stopped_iteration"] = pd.to_numeric(stopped["stopped_iteration"], errors="coerce").astype("Int64")
stopped["num_questions"] = pd.to_numeric(stopped["num_questions"], errors="coerce")

stopped = stopped.dropna(subset=["stopped_iteration", "num_questions"])
stopped = stopped.groupby("stopped_iteration", as_index=False)["num_questions"].sum()
stopped = stopped.set_index("stopped_iteration").reindex([1, 2, 3, 4], fill_value=0)

total_questions = float(stopped["num_questions"].sum())
if total_questions <= 0:
    raise ValueError("Total questions computed from q_stopped_at_iteration.csv is 0. Check the file contents.")

cumulative_stop_rate = (stopped["num_questions"].cumsum() / total_questions).values

# 3) Plot
x = np.array([1, 2, 3, 4], dtype=float)

plt.figure()
for m in METRICS_WANTED:
    y = means_wide[m].to_numpy(dtype=float)
    plt.plot(x, y, marker="o", label=LABELS[m])

plt.plot(x, cumulative_stop_rate, marker="o", label="Cumulative stop rate")

plt.xticks([1, 2, 3, 4])
plt.ylim(0, 1)
plt.xlabel("Iteration snapshot KGiRAG")
plt.ylabel("Score / Rate")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=200)

print(f"Saved plot to: {OUT_PNG}")
print(f"Total questions used for stop-rate denominator: {int(total_questions)}")
