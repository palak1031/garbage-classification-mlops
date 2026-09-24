import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================================================
# 1. SETTINGS
# =====================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
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

print("\nSplit distribution:")
print(df["split"].value_counts())

# =====================================================
# 3. CLASS LABELS
# =====================================================

class_names = sorted(df["class"].unique())

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

# Convert class names to numerical labels
class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

df["label"] = df["class"].map(class_to_index)

# =====================================================
# 4. CREATE DATA GENERATORS
# =====================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# =====================================================
# 5. DATA AUGMENTATION
# =====================================================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

# =====================================================
# 6. LOAD IMAGES
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
# 7. BUILD CUSTOM CNN
# =====================================================

model = models.Sequential([
    
    # Input
    layers.Input(shape=(224, 224, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 4
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# =====================================================
# 8. DISPLAY MODEL
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
# 10. CALLBACKS
# =====================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =====================================================
# 11. TRAIN
# =====================================================

print("\nStarting CNN training...\n")

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
    "custom_cnn.keras"
)

model.save(model_file)

print("\nModel saved to:")
print(model_file)

# =====================================================
# 13. EVALUATE ON TEST DATA
# =====================================================

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print("\n================================")
print("CUSTOM CNN TEST RESULT")
print("================================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# =====================================================
# 14. SAVE TRAINING HISTORY
# =====================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    os.path.join(
        RESULTS_PATH,
        "custom_cnn_history.csv"
    ),
    index=False
)

# =====================================================
# 15. TRAINING / VALIDATION ACCURACY GRAPH
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

plt.title("Custom CNN Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "custom_cnn_accuracy.png"
    ),
    dpi=300
)

plt.show()

# =====================================================
# 16. TRAINING / VALIDATION LOSS GRAPH
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

plt.title("Custom CNN Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "custom_cnn_loss.png"
    ),
    dpi=300
)

plt.show()

print("\n================================")
print("CUSTOM CNN TRAINING COMPLETED")
print("================================")