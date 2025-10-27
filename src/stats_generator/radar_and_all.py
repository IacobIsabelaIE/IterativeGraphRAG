import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


experiments = {
    "NER": "../experiments/experiment_optimized_real_final_ner_500.csv",
    "NO_NER": "../experiments/experiment_optimized_real_final_no_ner_500.csv",
    "MICROSOFT": "../experiments/results_final_1_microsoft.csv",
}

metrics = [
    "faithfulness", "completeness", "relevance",
    "bertscore_f1", "rougeL", "bleu", "cosine_sim"
]

# Pretty labels for metrics (display-only)
metric_label_map = {
    "faithfulness": "Faithfulness",
    "completeness": "Completeness",
    "relevance": "Relevance",
    "bertscore_f1": "BERTScore",
    "rougeL": "ROUGE-L",
    "bleu": "BLEU",
    "cosine_sim": "Cosine Sim.",
}

# Pretty labels for experiment names (legend)
experiment_label_map = {
    "MICROSOFT": "Edge et al.",
    "NER": "KGiRAG with NER",
    "NO_NER": "KGiRAG",  # (requested: NO_NER -> "KGiRAG")
}

output_dir = "../statistics"
os.makedirs(output_dir, exist_ok=True)


all_dfs = []
summary_stats = {}

for name, path in experiments.items():
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df["experiment"] = name
    all_dfs.append(df)

    stats = df[metrics].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary_stats[name] = stats

combined_df = pd.concat(all_dfs, ignore_index=True)
summary_df = pd.concat(summary_stats, axis=1)
summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)
summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
print("📊 Summary statistics saved.")


mean_values = combined_df.groupby("experiment")[metrics].mean().T  # rows: metrics, cols: experiments

# Plot 1: Average Metric Comparison (Bar Chart)

plt.figure(figsize=(12, 7))
ax = mean_values.plot(kind="bar", figsize=(12, 7))

# X tick labels -> pretty metric names
ax.set_xticklabels([metric_label_map.get(m, m) for m in mean_values.index], rotation=45, ha="right")

ax.set_title("Average Metric Comparison per Experiment")
ax.set_ylabel("Average Score")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Legend -> map experiment names to pretty names
current_labels = [t.get_text() for t in ax.legend_.texts]
pretty_labels = [experiment_label_map.get(l, l) for l in current_labels]
ax.legend(pretty_labels, title="Method", loc="best")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_means.png"), dpi=300)
plt.close()

# Plot 2: Radar Chart (Overall Metric Comparison)

labels = list(mean_values.index)
pretty_labels = [metric_label_map.get(m, m) for m in labels]

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Choose a stable color cycle (matplotlib default) by not specifying colors explicitly
# Plot each experiment
exp_order = list(mean_values.columns)
for exp in exp_order:
    values = mean_values[exp].tolist()
    values += values[:1]  # close the loop
    ax.plot(angles, values, linewidth=2, label=experiment_label_map.get(exp, exp))
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(pretty_labels)

ax.set_title("Radar Chart: Overall Metric Comparison")
ax.legend(title="Method", loc="upper right", bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "radar_overall.png"), dpi=300)
plt.close()

print(f"✅ Saved plots (bar_means.png, radar_overall.png) in '{output_dir}'")
