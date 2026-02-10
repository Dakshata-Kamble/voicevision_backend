import os

DATASET_DIR = "dataset_final"

print("\nFinal Dataset Summary:\n")

total = 0
for folder in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(path):
        count = len([
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        print(f"{folder}: {count} images")
        total += count

print("\nTotal images:", total)
