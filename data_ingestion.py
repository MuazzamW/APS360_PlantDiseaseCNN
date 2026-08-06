from pathlib import Path
import re
import json
import hashlib
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from class_mapping import CLASS_MAPPING1


plantvillage_root = Path("PlantVillage_Dataset")
plantdoc_root = Path("PlantDoc_Dataset")
output_root = Path("Combined_Dataset")



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


#if the plantdoc test set has 500 images, the script
#samples about 7 225 plantvillage images
#to match the original proportion of plantvillage:plantdoc images

RANDOM_STATE = 42

# Count full dataset source proportions
dataset_counts = df["dataset"].value_counts()

num_total_plantvillage = dataset_counts.get("plantvillage", 0)
num_total_plantdoc = dataset_counts.get("plantdoc", 0)

if num_total_plantvillage == 0:
    raise RuntimeError("No PlantVillage images found.")

if num_total_plantdoc == 0:
    raise RuntimeError("No PlantDoc images found.")

plantvillage_to_plantdoc_ratio = (
    num_total_plantvillage / num_total_plantdoc
)

print("\nFull dataset source counts:")
print(dataset_counts)

print(
    f"\nPlantVillage:PlantDoc ratio = "
    f"{plantvillage_to_plantdoc_ratio:.4f}:1"
)

# Use PlantDoc's existing test split as the PlantDoc portion of the test set
plantdoc_test_df = df[
    (df["dataset"] == "plantdoc")
    & (df["original_split"] == "test")
].copy()

if plantdoc_test_df.empty:
    raise RuntimeError(
        "PlantDoc test set is empty. Check your PlantDoc/test folders and CLASS_MAPPING1."
    )

# Candidate PlantVillage images for the PlantVillage portion of the test set
# These come from PlantVillage train and val because PlantVillage has no test split.
plantvillage_candidates_df = df[
    df["dataset"] == "plantvillage"
].copy()

if plantvillage_candidates_df.empty:
    raise RuntimeError(
        "No PlantVillage images found. Check your PlantVillage folders and CLASS_MAPPING1."
    )

# Number of PlantVillage images needed so test source proportions match the full dataset
num_plantdoc_test = len(plantdoc_test_df)

num_plantvillage_needed = round(
    num_plantdoc_test * plantvillage_to_plantdoc_ratio
)

print(
    f"\nUsing {num_plantdoc_test} PlantDoc test images."
)

print(
    f"Sampling {num_plantvillage_needed} PlantVillage test images "
    f"to preserve the original source proportion."
)

if len(plantvillage_candidates_df) < num_plantvillage_needed:
    raise ValueError(
        f"Not enough PlantVillage images to create proportional test set. "
        f"Need {num_plantvillage_needed}, but only have {len(plantvillage_candidates_df)}."
    )

# Try to sample PlantVillage test images with class proportions based on PlantDoc test,
# scaled by the PlantVillage:PlantDoc ratio.
plantdoc_test_class_counts = plantdoc_test_df["class"].value_counts()

plantvillage_test_parts = []

for class_name, pd_count in plantdoc_test_class_counts.items():
    pv_class_candidates = plantvillage_candidates_df[
        plantvillage_candidates_df["class"] == class_name
    ]

    pv_needed_for_class = round(
        pd_count * plantvillage_to_plantdoc_ratio
    )

    if pv_needed_for_class == 0:
        continue

    if len(pv_class_candidates) >= pv_needed_for_class:
        sampled = pv_class_candidates.sample(
            n=pv_needed_for_class,
            random_state=RANDOM_STATE
        )
    else:
        print(
            f"Warning: Not enough PlantVillage images for class '{class_name}'. "
            f"Needed {pv_needed_for_class}, found {len(pv_class_candidates)}. "
            f"Using all available PlantVillage images for this class."
        )

        sampled = pv_class_candidates

    plantvillage_test_parts.append(sampled)

if len(plantvillage_test_parts) == 0:
    raise RuntimeError(
        "No PlantVillage test images were sampled. Check class mappings."
    )

plantvillage_test_df = pd.concat(
    plantvillage_test_parts,
    ignore_index=False
)

# Adjust if rounding or class shortages made the PlantVillage test count too small or too large
current_pv_count = len(plantvillage_test_df)

if current_pv_count < num_plantvillage_needed:
    remaining_needed = num_plantvillage_needed - current_pv_count

    already_selected_indices = plantvillage_test_df.index

    remaining_pv_candidates = plantvillage_candidates_df.drop(
        already_selected_indices
    )

    if len(remaining_pv_candidates) < remaining_needed:
        raise ValueError(
            f"After class-proportional sampling, still need {remaining_needed} "
            f"PlantVillage images, but only {len(remaining_pv_candidates)} remain."
        )

    top_up_df = remaining_pv_candidates.sample(
        n=remaining_needed,
        random_state=RANDOM_STATE
    )

    plantvillage_test_df = pd.concat(
        [plantvillage_test_df, top_up_df],
        ignore_index=False
    )

elif current_pv_count > num_plantvillage_needed:
    plantvillage_test_df = plantvillage_test_df.sample(
        n=num_plantvillage_needed,
        random_state=RANDOM_STATE
    )

# Final test set: same PlantVillage/PlantDoc source proportion as full dataset
test_df = pd.concat(
    [plantdoc_test_df, plantvillage_test_df],
    ignore_index=False
).copy()

# Everything not selected for test becomes train/val pool
train_val_df = df.drop(test_df.index).copy()

print("\nTest set source counts:")
print(test_df["dataset"].value_counts())

print("\nTest set source proportions:")
print(test_df["dataset"].value_counts(normalize=True))

print("\nTest set class/source counts:")
print(
    test_df.groupby(["class", "dataset"])
    .size()
    .reset_index(name="count")
    .sort_values(["class", "dataset"])
)

# Verify no original file path appears in both test and train/val
train_val_paths = set(train_val_df["path"])
test_paths = set(test_df["path"])

path_overlap = train_val_paths.intersection(test_paths)

print(f"\nTrain-Val/Test path overlap: {len(path_overlap)}")

if len(path_overlap) > 0:
    raise RuntimeError(
        "Data leakage detected: some image paths appear in both train/val and test."
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


if output_root.exists():
    print(f"\nRemoving existing output folder: {output_root}")
    shutil.rmtree(output_root)

output_root.mkdir(parents=True, exist_ok=True)

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





