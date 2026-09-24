import os
import time
import numpy as np
import pandas as pd
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

INPUT_FOLDER = os.path.join(
    BASE_DIR,
    "input"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results"
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
# CREATE FOLDERS
# ============================================================

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================
# FIND IMAGE
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

images = [
    file for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(valid_extensions)
]

if len(images) == 0:
    print("========================================")
    print("NO IMAGE FOUND")
    print("========================================")
    print()
    print("Please put a smartphone image inside:")
    print(INPUT_FOLDER)
    print()
    print("Supported formats:")
    print("JPG, JPEG, PNG, WEBP")
    exit()

# Use the first image
image_name = images[0]

IMAGE_PATH = os.path.join(
    INPUT_FOLDER,
    image_name
)

print("========================================")
print("SMARTPHONE IMAGE GARBAGE CLASSIFICATION")
print("========================================")

print()
print("Input image:")
print(image_name)

# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print()
print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("TFLite model loaded successfully!")

# ============================================================
# LOAD IMAGE
# ============================================================

print()
print("Loading smartphone image...")

image = Image.open(IMAGE_PATH).convert("RGB")

original_size = image.size

image = image.resize(IMG_SIZE)

image_array = np.array(
    image,
    dtype=np.float32
)

# MobileNetV2 preprocessing
image_array = image_array / 127.5 - 1.0

# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)

print("Original image size:", original_size)
print("Model input size:", IMG_SIZE)

# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("Running edge inference...")

start_time = time.perf_counter()

interpreter.set_tensor(
    input_index,
    image_array
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_index
)

end_time = time.perf_counter()

# ============================================================
# PROCESS RESULT
# ============================================================

inference_time = (
    end_time - start_time
) * 1000

prediction = prediction[0]

predicted_index = int(
    np.argmax(prediction)
)

predicted_class = CLASS_NAMES[
    predicted_index
]

confidence = (
    float(prediction[predicted_index])
    * 100
)

# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("========================================")
print("PREDICTION RESULT")
print("========================================")

print()
print("Image            :", image_name)
print("Predicted Class  :", predicted_class)
print(f"Confidence       : {confidence:.2f}%")
print(f"Inference Time   : {inference_time:.2f} ms")

print()
print("Class Probabilities")
print("----------------------------------------")

for class_name, probability in zip(
    CLASS_NAMES,
    prediction
):
    print(
        f"{class_name:<12}: "
        f"{probability * 100:.2f}%"
    )

# ============================================================
# SAVE RESULT FOR DASHBOARD
# ============================================================

result = pd.DataFrame({
    "image": [image_name],
    "predicted_class": [predicted_class],
    "confidence": [confidence],
    "inference_time_ms": [inference_time]
})

result_path = os.path.join(
    RESULTS_FOLDER,
    "phone_prediction.csv"
)

result.to_csv(
    result_path,
    index=False
)

print()
print("========================================")
print("INFERENCE COMPLETED")
print("========================================")

print()
print("Result saved to:")
print(result_path)