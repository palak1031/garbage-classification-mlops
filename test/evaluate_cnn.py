import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PATHS
# ==========================================

MODEL_PATH = r"C:\deep learning\models\custom_cnn.keras"
RESULTS_PATH = r"C:\deep learning\results"

CSV_PATH = os.path.join(
    RESULTS_PATH,
    "dataset_split.csv"
)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading Custom CNN...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD TEST DATA
# ==========================================

df = pd.read_csv(CSV_PATH)

test_df = df[df["split"] == "test"].copy()

print("\nTest images:", len(test_df))


# ==========================================
# CLASS NAMES
# ==========================================

class_names = sorted(df["class"].unique())

print("\nClasses:")
print(class_names)


# ==========================================
# CREATE TEST GENERATOR
# ==========================================

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = test_generator.classes


# ==========================================
# BASIC METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n================================")
print("CNN PERFORMANCE METRICS")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# SAVE OVERALL METRICS
# ==========================================

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(
    os.path.join(
        RESULTS_PATH,
        "custom_cnn_metrics.csv"
    ),
    index=False
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(
        RESULTS_PATH,
        "custom_cnn_classification_report.csv"
    )
)

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Custom CNN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "custom_cnn_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n================================")
print("EVALUATION COMPLETED")
print("================================")

print("\nFiles generated:")

print("1. custom_cnn_metrics.csv")
print("2. custom_cnn_classification_report.csv")
print("3. custom_cnn_confusion_matrix.png")