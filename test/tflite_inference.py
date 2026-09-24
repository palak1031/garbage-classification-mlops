import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\deep learning"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2.tflite"
)

# Use one test image from your dataset
IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "images",
    "plastic",
    "plastic_00003.jpg"
)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("========================================")
print("TENSORFLOW LITE INFERENCE")
print("========================================")

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

print("TFLite model loaded successfully!")

# ============================================================
# GET INPUT AND OUTPUT DETAILS
# ============================================================

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("\nInput shape:")
print(input_details[0]["shape"])

print("\nOutput shape:")
print(output_details[0]["shape"])

# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}\n\n"
        "Please change IMAGE_PATH to an existing image."
    )

image = Image.open(IMAGE_PATH).convert("RGB")

image = image.resize(IMG_SIZE)

image_array = np.array(image).astype(np.float32)

# MobileNetV2 preprocessing
image_array = image_array / 127.5 - 1.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

print("Image loaded successfully!")

# ============================================================
# RUN TFLITE INFERENCE
# ============================================================

interpreter.set_tensor(
    input_index,
    image_array
)

print("\nRunning inference...")

start_time = time.perf_counter()

interpreter.invoke()

end_time = time.perf_counter()

inference_time = (end_time - start_time) * 1000

# ============================================================
# GET PREDICTION
# ============================================================

prediction = interpreter.get_tensor(output_index)

prediction = prediction[0]

predicted_index = np.argmax(prediction)

predicted_class = CLASS_NAMES[predicted_index]

confidence = prediction[predicted_index] * 100

# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n========================================")
print("TFLITE PREDICTION RESULT")
print("========================================")

print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")
print(f"Inference Time  : {inference_time:.2f} ms")

print("\nAll class probabilities:")

for class_name, probability in zip(
    CLASS_NAMES,
    prediction
):
    print(
        f"{class_name:10s}: {probability * 100:.2f}%"
    )

print("\n========================================")
print("EDGE INFERENCE COMPLETED")
print("========================================")