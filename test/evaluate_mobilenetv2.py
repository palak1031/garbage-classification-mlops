import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\deep learning"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2.keras"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "results",
    "dataset_split.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# ============================================================
# START
# ============================================================

print("========================================")
print("MOBILENETV2 EVALUATION")
print("========================================")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading dataset split...")

df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns.tolist())

print("\nTotal images:", len(df))


# ============================================================
# SELECT TEST DATA
# ============================================================

test_df = df[df["split"] == "test"].copy()

print("Test images:", len(test_df))


# ============================================================
# IMAGE PATH
# ============================================================

# Your CSV already contains the complete image path.
# Therefore, we directly use the image_path column.

test_df["image_path"] = test_df["image_path"].astype(str)


# ============================================================
# CHECK TEST IMAGES
# ============================================================

missing = []

for path in test_df["image_path"]:

    if not os.path.exists(path):
        missing.append(path)


if len(missing) > 0:

    print("\nERROR: Some image files were not found.")

    print("Number of missing images:", len(missing))

    print("\nFirst few missing paths:")

    for path in missing[:10]:
        print(path)

    raise FileNotFoundError(
        "Test image paths are incorrect."
    )


print("All test images found successfully!")


# ============================================================
# TEST DATA GENERATOR
# ============================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=
    tf.keras.applications.mobilenet_v2.preprocess_input
)


test_generator = test_datagen.flow_from_dataframe(

    dataframe=test_df,

    x_col="image_path",

    y_col="class",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASS_NAMES,

    shuffle=False
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating MobileNetV2 predictions...")

predictions = model.predict(
    test_generator
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_generator.classes


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# DISPLAY METRICS
# ============================================================

print("\n========================================")
print("MOBILENETV2 PERFORMANCE METRICS")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

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


metrics_path = os.path.join(
    RESULTS_PATH,
    "mobilenetv2_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report_text = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    zero_division=0
)

print(report_text)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    output_dict=True,

    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


report_path = os.path.join(
    RESULTS_PATH,
    "mobilenetv2_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES

)


plt.title(
    "MobileNetV2 Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()


cm_path = os.path.join(
    RESULTS_PATH,
    "mobilenetv2_confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=300
)

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print("\nFiles generated:")

print("1. mobilenetv2_metrics.csv")

print("2. mobilenetv2_classification_report.csv")

print("3. mobilenetv2_confusion_matrix.png")

print("\nAll files saved in:")

print(RESULTS_PATH)