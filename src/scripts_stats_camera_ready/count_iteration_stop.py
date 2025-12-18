import pandas as pd

INPUT_CSV = "../experiments/experiment_optimized_real_final_ner_500.csv"
OUTPUT_CSV = "../stats_camera_ready/q_stopped_at_iteration_ner.csv"

df = pd.read_csv(INPUT_CSV)

required_cols = {"question_id", "iteration"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").astype("Int64")
df = df.dropna(subset=["question_id", "iteration"])

# "Stopped at iteration k" = max iteration reached by that question_id is k
max_iter = df.groupby("question_id")["iteration"].max()

counts = (
    max_iter.value_counts()
    .reindex([1, 2, 3, 4], fill_value=0)
    .rename_axis("stopped_iteration")
    .reset_index(name="num_questions")
)

counts.to_csv(OUTPUT_CSV, index=False)

print(counts)
print(f"Wrote: {OUTPUT_CSV}")
