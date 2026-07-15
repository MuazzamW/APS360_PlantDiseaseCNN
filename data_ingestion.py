from pathlib import Path
import re
import json
import hashlib
import shutil

import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from class_mapping import CLASS_MAPPING1


# --------------------------------------------------
# Paths
# --------------------------------------------------

plantvillage_root = Path("PlantVillage_Dataset")
plantdoc_root = Path("PlantDoc_Dataset")
output_root = Path("Combined_Dataset")


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def get_class_names(root, split_names):
    classes = set()

    for split in split_names:
        split_path = root / split

        if not split_path.exists():
            print(f"Warning: {split_path} does not exist")
            continue

        for path in split_path.iterdir():
            if path.is_dir():
                classes.add(path.name)

    return sorted(classes)


def normalize_class_name(name):
    name = name.lower()

    name = name.replace("___", " ")
    name = name.replace("_", " ")
    name = name.replace("-", " ")

    removable_words = {
        "leaf",
        "leaves",
        "plant",
        "disease"
    }

    words = re.findall(r"[a-z0-9]+", name)
    words = [word for word in words if word not in removable_words]

    return " ".join(words)


def collect_images_from_folder(
    folder,
    standardized_class,
    dataset_name,
    original_split,
    original_class
):
    records = []

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return records

    for image_path in folder.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append({
                "path": str(image_path),
                "class": standardized_class,
                "dataset": dataset_name,
                "original_split": original_split,
                "original_class": original_class
            })

    return records


def unique_filename(row):
    source_path = Path(row["path"])

    identifier = "|".join([
        row["dataset"],
        row["original_split"],
        row["original_class"],
        str(source_path)
    ])

    short_hash = hashlib.md5(
        identifier.encode("utf-8")
    ).hexdigest()[:10]

    return f"{row['dataset']}_{short_hash}_{source_path.name}"


# --------------------------------------------------
# Inspect classes
# --------------------------------------------------

pv_classes = get_class_names(
    plantvillage_root,
    ["train", "val"]
)

pd_classes = get_class_names(
    plantdoc_root,
    ["train", "test"]
)

print("PlantVillage classes:")
for class_name in pv_classes:
    print(class_name)

print("\nPlantDoc classes:")
for class_name in pd_classes:
    print(class_name)


# --------------------------------------------------
# Collect matching images using CLASS_MAPPING1
# --------------------------------------------------

records = []

for standardized_class, source_mapping in CLASS_MAPPING1.items():
    print(f"\nProcessing standardized class: {standardized_class}")
    print(f"Mapping: {source_mapping}")

    for original_class in source_mapping.get("plantvillage", []):
        for split in ["train", "val"]:
            folder = plantvillage_root / split / original_class

            records.extend(
                collect_images_from_folder(
                    folder=folder,
                    standardized_class=standardized_class,
                    dataset_name="plantvillage",
                    original_split=split,
                    original_class=original_class
                )
            )

    for original_class in source_mapping.get("plantdoc", []):
        for split in ["train", "test"]:
            folder = plantdoc_root / split / original_class

            records.extend(
                collect_images_from_folder(
                    folder=folder,
                    standardized_class=standardized_class,
                    dataset_name="plantdoc",
                    original_split=split,
                    original_class=original_class
                )
            )


df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(
        "No images were found. Check your dataset paths and CLASS_MAPPING1."
    )

print("\nPreview:")
print(df.head())

print("\nImages per class:")
print(df["class"].value_counts())

print("\nImages per dataset:")
print(df["dataset"].value_counts())

df.to_csv("combined_dataset_manifest_before_split.csv", index=False)


# --------------------------------------------------
# Approach A:
# Keep PlantDoc test as final test set
# --------------------------------------------------

test_df = df[
    (df["dataset"] == "plantdoc")
    & (df["original_split"] == "test")
].copy()

train_val_df = df.drop(test_df.index).copy()

if test_df.empty:
    print(
        "\nWarning: PlantDoc test set is empty. "
        "Check your PlantDoc/test folders and class mapping."
    )

class_counts = train_val_df["class"].value_counts()
too_small = class_counts[class_counts < 2]

if not too_small.empty:
    raise ValueError(
        "The following classes do not have enough images "
        f"for stratified splitting:\n{too_small}"
    )

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.20,
    random_state=42,
    stratify=train_val_df["class"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# --------------------------------------------------
# Create combined folder structure
# --------------------------------------------------

if output_root.exists():
    print(f"\nWarning: {output_root} already exists. Files may be overwritten or duplicated.")

for _, row in final_df.iterrows():
    source_path = Path(row["path"])

    destination_folder = (
        output_root
        / row["split"]
        / row["class"]
    )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_path = (
        destination_folder
        / unique_filename(row)
    )

    shutil.copy2(source_path, destination_path)


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

final_df.to_csv(
    output_root / "dataset_manifest.csv",
    index=False
)

summary = (
    final_df
    .groupby(["split", "class", "dataset"])
    .size()
    .reset_index(name="image_count")
)

summary.to_csv(
    output_root / "dataset_summary.csv",
    index=False
)

print("\nCombined dataset counts:")
print(
    final_df.groupby(["split", "class"])
    .size()
    .unstack(fill_value=0)
)

print(f"\nDataset created at: {output_root.resolve()}")


# --------------------------------------------------
# Create PyTorch DataLoaders
# --------------------------------------------------



