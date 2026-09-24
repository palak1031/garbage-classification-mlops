import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# =====================================================
# 1. SETTINGS
# =====================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
NUM_CLASSES = 6

RESULTS_PATH = r"C:\deep learning\results"
MODEL_PATH = r"C:\deep learning\models"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


# =====================================================
# 2. LOAD DATASET SPLIT
# =====================================================

csv_path = os.path.join(
    RESULTS_PATH,
    "dataset_split.csv"
)

df = pd.read_csv(csv_path)

print("Total images:", len(df))

# Separate datasets
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))


# =====================================================
# 3. CLASS NAMES
# =====================================================

class_names = sorted(df["class"].unique())

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# =====================================================
# 4. DATA GENERATORS
# =====================================================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)


# =====================================================
# 5. LOAD IMAGES
# =====================================================

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

validation_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# =====================================================
# 6. LOAD PRE-TRAINED MOBILENETV2
# =====================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pre-trained layers
base_model.trainable = False

print("MobileNetV2 loaded successfully.")


# =====================================================
# 7. BUILD MOBILE CNN MODEL
# =====================================================

inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# =====================================================
# 8. MODEL SUMMARY
# =====================================================

model.summary()


# =====================================================
# 9. COMPILE
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# 10. EARLY STOPPING
# =====================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# =====================================================
# 11. TRAIN
# =====================================================

print("\n========================================")
print("STARTING MOBILENETV2 TRAINING")
print("========================================")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# =====================================================
# 12. SAVE MODEL
# =====================================================

model_file = os.path.join(
    MODEL_PATH,
    "mobilenetv2.keras"
)

model.save(model_file)

print("\nModel saved to:")
print(model_file)


# =====================================================
# 13. TEST EVALUATION
# =====================================================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\n========================================")
print("MOBILENETV2 TEST RESULT")
print("========================================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# =====================================================
# 14. SAVE TRAINING HISTORY
# =====================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    os.path.join(
        RESULTS_PATH,
        "mobilenetv2_history.csv"
    ),
    index=False
)


# =====================================================
# 15. ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "mobilenetv2_accuracy.png"
    ),
    dpi=300
)

plt.show()


# =====================================================
# 16. LOSS GRAPH
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "MobileNetV2 Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "mobilenetv2_loss.png"
    ),
    dpi=300
)

plt.show()


print("\n========================================")
print("MOBILENETV2 TRAINING COMPLETED")
print("========================================")