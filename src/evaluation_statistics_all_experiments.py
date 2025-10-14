import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

experiments = {
    "NER": "experiments/experiment_optimized_real_final_ner_500.csv",
    "NO_NER": "experiments/experiment_optimized_real_final_no_ner_500.csv",
    "MICROSOFT": "experiments/results_final_1_microsoft.csv"  
}

metrics = [
    "faithfulness", "completeness", "relevance",
    "bertscore_f1", "rougeL", "bleu", "cosine_sim"
]

output_dir = "statistics"
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

mean_values = combined_df.groupby("experiment")[metrics].mean().T
plt.figure(figsize=(12, 7))
mean_values.plot(kind="bar")
plt.title("Average Metric Comparison per Experiment")
plt.ylabel("Average Score")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_means.png"), dpi=300)
plt.close()

for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="experiment", y=metric, data=combined_df, palette="Set2")
    plt.title(f"Distribution of {metric} Scores by Experiment")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"box_{metric}.png"), dpi=300)
    plt.close()

for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="experiment", y=metric, data=combined_df, palette="muted", inner="quartile")
    plt.title(f"Score Density of {metric} by Experiment")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violin_{metric}.png"), dpi=300)
    plt.close()

labels = metrics
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (exp, values) in enumerate(mean_values.items()):
    stats = values.tolist()
    stats += stats[:1]
    ax.plot(angles, stats, color=colors[i], linewidth=2, label=exp)
    ax.fill(angles, stats, color=colors[i], alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], labels)
plt.title("Radar Chart: Overall Metric Comparison")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "radar_overall.png"), dpi=300)
plt.close()

corr = combined_df[metrics].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Metrics")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metric_correlation.png"), dpi=300)
plt.close()

print(f"All comparison plots saved in '{output_dir}'")