import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# PATHS
# ==========================================

images_folder = r"C:\deep learning\dataset\images"
results_folder = r"C:\deep learning\results"

os.makedirs(results_folder, exist_ok=True)

# ==========================================
# COLLECT ALL IMAGES
# ==========================================

data = []

classes = sorted([
    folder
    for folder in os.listdir(images_folder)
    if os.path.isdir(os.path.join(images_folder, folder))
])

for class_name in classes:

    class_folder = os.path.join(images_folder, class_name)

    for file_name in os.listdir(class_folder):

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            image_path = os.path.join(class_folder, file_name)

            data.append({
                "image_path": image_path,
                "class": class_name
            })

# Create DataFrame
df = pd.DataFrame(data)

print("Total images:", len(df))

# ==========================================
# TRAIN = 70%
# TEMP = 30%
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"]
)

# ==========================================
# VALIDATION = 15%
# TEST = 15%
# ==========================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"]
)

# ==========================================
# ADD SPLIT COLUMN
# ==========================================

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"

# Combine
final_df = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True
)

# Shuffle
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================
# SAVE CSV
# ==========================================

output_file = os.path.join(
    results_folder,
    "dataset_split.csv"
)

final_df.to_csv(
    output_file,
    index=False
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("DATASET SPLIT")
print("====================================")

print("\nOverall:")
print("Total:", len(final_df))

print("\nSplit distribution:")
print(final_df["split"].value_counts())

print("\nClass distribution:")
print(
    pd.crosstab(
        final_df["class"],
        final_df["split"]
    )
)

print("\n====================================")
print("DATASET SPLIT COMPLETED")
print("====================================")

print("\nSaved to:")
print(output_file)