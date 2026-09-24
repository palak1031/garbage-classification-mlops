import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\deep learning"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model performance
comparison = pd.DataFrame({
    "Model": [
        "Custom CNN",
        "MobileNetV2"
    ],
    "Accuracy": [
        0.6294,
        0.9060
    ],
    "Precision": [
        0.6420,
        0.9085
    ],
    "Recall": [
        0.6294,
        0.9060
    ],
    "F1 Score": [
        0.6292,
        0.9064
    ]
})

# Save CSV
csv_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

comparison.to_csv(csv_path, index=False)

print("========================================")
print("MODEL PERFORMANCE COMPARISON")
print("========================================")

print(comparison.to_string(index=False))

# Create comparison chart
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    comparison.loc[0, metrics],
    width=width,
    label="Custom CNN"
)

plt.bar(
    [i + width / 2 for i in x],
    comparison.loc[1, metrics],
    width=width,
    label="MobileNetV2"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Custom CNN vs MobileNetV2")
plt.legend()

plt.tight_layout()

chart_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.png"
)

plt.savefig(chart_path, dpi=300)
plt.show()

print("\n========================================")
print("COMPARISON COMPLETED")
print("========================================")

print("\nGenerated files:")
print("1.", csv_path)
print("2.", chart_path)