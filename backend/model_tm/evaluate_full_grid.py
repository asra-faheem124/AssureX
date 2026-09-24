"""
AssureX Claim Engine - Teachable Machine Full Diagnostic Grid
--------------------------------------------------------------------
Round 1 found that pixel scaling matters (0-to-1 beat -1-to-1), but
the resulting accuracy (46.7%) doesn't match what you saw directly in
the Teachable Machine website (where Invalid was "almost perfect").
That mismatch suggests a SECOND factor is still off: how the
non-square 700x900 card image gets turned into a square 224x224
image.

This script tests all 4 combinations of:
    RESIZE METHOD:  crop-to-fit   vs   stretch-to-fit
    PIXEL SCALING:  -1 to 1        vs   0 to 1

against your real validation images (known labels from folder names)
and reports full accuracy + confusion matrix for each combination, so
we can see, with evidence, which exact combination matches Teachable
Machine's real behavior.

Run this from your model_tm/ folder:
    python evaluate_full_grid.py
"""

import os
import glob
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_unquant.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
VAL_DIR = os.path.join(BASE_DIR, "..", "data", "claim_cards", "val")
IMAGE_SIZE = (224, 224)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# ---- Resize methods ----
def resize_crop(image):
    """Crops the image to a square (centered), then resizes to 224x224.
    This is what our current code does."""
    return ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)

def resize_stretch(image):
    """Squishes/stretches the image directly to 224x224, ignoring aspect
    ratio -- no cropping, but distorts proportions."""
    return image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

RESIZE_METHODS = {
    "crop-to-fit": resize_crop,
    "stretch-to-fit": resize_stretch,
}

# ---- Pixel scaling formulas ----
SCALING_METHODS = {
    "-1 to 1": lambda arr: (arr / 127.5) - 1,
    "0 to 1": lambda arr: arr / 255.0,
}


def predict(path, resize_fn, scale_fn):
    image = Image.open(path).convert("RGB")
    image = resize_fn(image)
    image_array = np.asarray(image).astype(np.float32)
    image_array = scale_fn(image_array)

    data = np.expand_dims(image_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return class_names[int(np.argmax(output))]


# Collect all validation images with known labels
test_images = []
for class_name in class_names:
    folder = os.path.join(VAL_DIR, class_name)
    for path in glob.glob(os.path.join(folder, "*.png")):
        test_images.append((path, class_name))

print(f"Found {len(test_images)} validation images.\n")
print("=" * 75)

results_summary = []

for resize_name, resize_fn in RESIZE_METHODS.items():
    for scale_name, scale_fn in SCALING_METHODS.items():
        matrix = {t: {p: 0 for p in class_names} for t in class_names}

        for path, true_label in test_images:
            predicted = predict(path, resize_fn, scale_fn)
            matrix[true_label][predicted] += 1

        correct = sum(matrix[c][c] for c in class_names)
        total = len(test_images)
        accuracy = correct / total * 100

        print(f"\nRESIZE: {resize_name}   |   SCALING: {scale_name}")
        print(f"  Overall accuracy: {accuracy:.1f}% ({correct}/{total})")
        header = "  " + "Actual\\Pred".ljust(15) + "".join(c.ljust(15) for c in class_names)
        print(header)
        for true_c in class_names:
            row = matrix[true_c]
            row_total = sum(row.values())
            row_acc = (row[true_c] / row_total * 100) if row_total else 0
            print("  " + true_c.ljust(15) + "".join(str(row[p]).ljust(15) for p in class_names) + f"  ({row_acc:.0f}% correct)")

        results_summary.append((resize_name, scale_name, accuracy))

print("\n" + "=" * 75)
print("SUMMARY (best to worst):")
for resize_name, scale_name, accuracy in sorted(results_summary, key=lambda x: -x[2]):
    print(f"  {accuracy:5.1f}%   resize={resize_name:15s} scaling={scale_name}")

print("\nTell me the full printout -- the winning combination's confusion")
print("matrix is what we'll lock into the real predictor code.")
