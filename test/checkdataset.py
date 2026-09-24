import os

# Main dataset folder
dataset_root = r"C:\deep learning\dataset"

# Find the "images" folder automatically
images_folder = None

for root, dirs, files in os.walk(dataset_root):
    if "images" in dirs:
        images_folder = os.path.join(root, "images")
        break

if images_folder is None:
    print("❌ 'images' folder was not found inside the dataset folder.")
    print("\nPlease check that your dataset has been extracted.")
else:
    print("✅ Dataset found!")
    print("Images folder:")
    print(images_folder)

    print("\n==============================")
    print("WASTE CLASSES")
    print("==============================")

    total_images = 0

    for folder in sorted(os.listdir(images_folder)):

        folder_path = os.path.join(images_folder, folder)

        if os.path.isdir(folder_path):

            # Count image files only
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                ))
            ]

            count = len(image_files)
            total_images += count

            print(f"{folder}: {count} images")

    print("\n==============================")
    print("TOTAL")
    print("==============================")
    print("Total images:", total_images)