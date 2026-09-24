"""
AssureX Claim Engine - Batch Claim Summary Card Generator
----------------------------------------------------------------
Reads claims_train.csv, claims_val.csv, and claims_test.csv, and creates
a Claim Summary Card image for every single row -- organized into folders
by class, exactly how Google Teachable Machine expects images to be
uploaded (one folder per class).

Rules followed (from the SRS):
- TRAINING images: 2 visual variations per claim (so the model learns the
  INFORMATION, not just one exact look).
- VALIDATION and TEST images: only 1 image each, and they must NOT be
  used for training -- kept in separate folders.
- A claim used for val/test must never appear in the training images
  (this is automatically true here, because we read from 3 already-
  separated CSV files).

Output folder structure created:
    claim_cards/
        train/
            Valid/           <claim_id>_v1.png, <claim_id>_v2.png, ...
            Invalid/
            ManualReview/
        val/
            Valid/            <claim_id>.png
            Invalid/
            ManualReview/
        test/
            Valid/
            Invalid/
            ManualReview/
    claim_image_mapping.csv   <- maps every claim_id to its image filename(s)

Run it with:
    python generate_training_cards.py
"""

import os
import pandas as pd
from card_generator import generate_card

DATA_DIR = "../data"
OUTPUT_DIR = "../data/claim_cards"

TRAIN_PATH = os.path.join(DATA_DIR, "claims_train.csv")
VAL_PATH = os.path.join(DATA_DIR, "claims_val.csv")
TEST_PATH = os.path.join(DATA_DIR, "claims_test.csv")


def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def generate_for_split(csv_path, split_name, variations_per_claim):
    """
    Generates card images for every row in the given CSV.
    split_name is "train", "val", or "test" -- used for folder naming.
    variations_per_claim: how many differently-styled images to make
    per claim (2 for train, 1 for val/test).
    """
    df = pd.read_csv(csv_path)
    mapping_rows = []

    for _, row in df.iterrows():
        claim = row.to_dict()
        class_label = claim["class_label"]

        class_folder = os.path.join(OUTPUT_DIR, split_name, class_label)
        ensure_folder(class_folder)

        image_filenames = []
        for v in range(variations_per_claim):
            img = generate_card(claim, variation=v)

            if variations_per_claim == 1:
                filename = f"{claim['claim_id']}.png"
            else:
                filename = f"{claim['claim_id']}_v{v + 1}.png"

            img.save(os.path.join(class_folder, filename))
            image_filenames.append(filename)

        mapping_rows.append({
            "claim_id": claim["claim_id"],
            "split": split_name,
            "class_label": class_label,
            "image_filenames": ";".join(image_filenames),
        })

    print(f"[{split_name}] Generated images for {len(df)} claims "
          f"({variations_per_claim} variation(s) each) -> {OUTPUT_DIR}/{split_name}/")
    return mapping_rows


if __name__ == "__main__":
    ensure_folder(OUTPUT_DIR)

    all_mapping_rows = []
    all_mapping_rows += generate_for_split(TRAIN_PATH, "train", variations_per_claim=2)
    all_mapping_rows += generate_for_split(VAL_PATH, "val", variations_per_claim=1)
    all_mapping_rows += generate_for_split(TEST_PATH, "test", variations_per_claim=1)

    mapping_df = pd.DataFrame(all_mapping_rows)
    mapping_path = os.path.join(DATA_DIR, "claim_image_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)

    total_images = sum(
        len(row["image_filenames"].split(";")) for row in all_mapping_rows
    )
    print(f"\nTotal images generated: {total_images}")
    print(f"Mapping file saved: {mapping_path}")
