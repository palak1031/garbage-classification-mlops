import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
images_folder = r"C:\deep learning\dataset\images"

# Classes
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Create figure
plt.figure(figsize=(12, 8))

# Show one random image from each class
for i, class_name in enumerate(classes):

    class_path = os.path.join(images_folder, class_name)

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    # Select random image
    image_name = random.choice(image_files)

    image_path = os.path.join(class_path, image_name)

    image = Image.open(image_path)

    # Plot
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(class_name.capitalize())
    plt.axis("off")

plt.suptitle(
    "Sample Images from Garbage Waste Dataset",
    fontsize=16
)

plt.tight_layout()

# Save
os.makedirs(r"C:\deep learning\results", exist_ok=True)

plt.savefig(
    r"C:\deep learning\results\sample_images.png",
    dpi=300
)

plt.show()