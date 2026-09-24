import os
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\deep learning"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2.keras"
)

TFLITE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2.tflite"
)

# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("TENSORFLOW LITE CONVERSION")
print("========================================")

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# CONVERT TO TENSORFLOW LITE
# ============================================================

print("\nConverting model to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# ============================================================
# SAVE TFLITE MODEL
# ============================================================

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("\nTensorFlow Lite conversion completed!")

print("\nTFLite model saved to:")
print(TFLITE_PATH)

# ============================================================
# MODEL SIZE
# ============================================================

keras_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

print("\n========================================")
print("MODEL SIZE")
print("========================================")

print(f"Keras model size : {keras_size:.2f} MB")
print(f"TFLite model size: {tflite_size:.2f} MB")

print("\n========================================")
print("CONVERSION COMPLETED")
print("========================================")