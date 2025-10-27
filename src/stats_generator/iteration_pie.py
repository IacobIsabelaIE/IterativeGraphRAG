import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configuration ===
INPUT_FILE = "../experiments/experiment_optimized_real_final_ner_500.csv"
OUTPUT_FILE = "pie_iteration_ner.png"
OUTPUT_DIR = "../statistics"

# === Load Data ===
df = pd.read_csv(INPUT_FILE)

# Ensure the iteration column is numeric
df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")

# Count how many questions per iteration
iteration_counts = df["iteration"].value_counts().sort_index()

# === Plot Pie Chart ===
plt.figure(figsize=(8, 8))
plt.pie(
    iteration_counts,
    labels=[f"Iteration {int(i)}" for i in iteration_counts.index],
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "black"}
)
plt.title("Distribution of Questions per Iteration", fontsize=14)

# === Save Output ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
plt.savefig(output_path, dpi=300)

print(f"Saved pie chart to {output_path}")
