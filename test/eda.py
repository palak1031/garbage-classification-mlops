import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# DATASET PATH
# ==========================================

dataset_root = r"C:\deep learning\dataset"

# Find images folder automatically
images_folder = None

for root, dirs, files in os.walk(dataset_root):
    if "images" in dirs:
        images_folder = os.path.join(root, "images")
        break

if images_folder is None:
    print("ERROR: Images folder not found.")
    exit()

print("Dataset location:")
print(images_folder)

# ==========================================
# FIND CLASSES
# ==========================================

classes = sorted([
    folder
    for folder in os.listdir(images_folder)
    if os.path.isdir(os.path.join(images_folder, folder))
])

print("\nClasses:")
for i, class_name in enumerate(classes):
    print(i, "->", class_name)

# ==========================================
# COLLECT IMAGE INFORMATION
# ==========================================

data = []
corrupted_images = []

for class_name in classes:

    class_path = os.path.join(images_folder, class_name)

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        if not file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        try:

            with Image.open(file_path) as img:

                width, height = img.size
                mode = img.mode

            data.append({
                "filename": file_name,
                "class": class_name,
                "width": width,
                "height": height,
                "mode": mode
            })

        except Exception:
            corrupted_images.append(file_path)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(data)

print("\n================================")
print("DATASET OVERVIEW")
print("================================")

print("Total images:", len(df))
print("Total classes:", len(classes))

print("\nClass names:")
print(classes)

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

class_counts = df["class"].value_counts()

print("\n================================")
print("IMAGES PER CLASS")
print("================================")

print(class_counts)

# ==========================================
# IMAGE DIMENSIONS
# ==========================================

print("\n================================")
print("IMAGE DIMENSIONS")
print("================================")

print(df[["width", "height"]].describe())

# ==========================================
# IMAGE COLOR MODE
# ==========================================

print("\n================================")
print("COLOR MODE")
print("================================")

print(df["mode"].value_counts())

# ==========================================
# CORRUPTED IMAGES
# ==========================================

print("\n================================")
print("DATA QUALITY")
print("================================")

print("Corrupted images:", len(corrupted_images))

# ==========================================
# CREATE RESULTS FOLDER
# ==========================================

results_folder = r"C:\deep learning\results"

os.makedirs(results_folder, exist_ok=True)

# ==========================================
# SAVE DATASET INFORMATION
# ==========================================

df.to_csv(
    os.path.join(results_folder, "dataset_info.csv"),
    index=False
)

class_counts_df = class_counts.reset_index()

class_counts_df.columns = [
    "class",
    "image_count"
]

class_counts_df.to_csv(
    os.path.join(results_folder, "class_distribution.csv"),
    index=False
)

# ==========================================
# CLASS DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

class_counts.plot(kind="bar")

plt.title("Garbage Waste Class Distribution")
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(results_folder, "class_distribution.png"),
    dpi=300
)

plt.show()

# ==========================================
# PIE CHART
# ==========================================

plt.figure(figsize=(8, 8))

class_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Garbage Waste Distribution")

plt.ylabel("")

plt.tight_layout()

plt.savefig(
    os.path.join(results_folder, "class_distribution_pie.png"),
    dpi=300
)

plt.show()

# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("================================")

print("\nFiles generated in:")
print(results_folder)

print("\nGenerated files:")
print("1. dataset_info.csv")
print("2. class_distribution.csv")
print("3. class_distribution.png")
print("4. class_distribution_pie.png")