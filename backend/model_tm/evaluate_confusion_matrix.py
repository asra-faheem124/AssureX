"""
AssureX Claim Engine - Teachable Machine Detailed Evaluation
--------------------------------------------------------------
Runs the CORRECTED preprocessing (0 to 1 scaling, confirmed via
evaluate_teachable_machine.py) against all 225 validation images and
produces a real per-class confusion matrix -- so we can see exactly
which classes are still being confused, not just one overall accuracy
number.

Run this from your model_tm/ folder:
    python evaluate_confusion_matrix.py
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


def predict_image(path):
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0  # CORRECTED formula
    data = np.expand_dims(image_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return class_names[int(np.argmax(output))]


# Build confusion matrix: rows = true label, columns = predicted label
matrix = {true_c: {pred_c: 0 for pred_c in class_names} for true_c in class_names}

for true_class in class_names:
    folder = os.path.join(VAL_DIR, true_class)
    for path in glob.glob(os.path.join(folder, "*.png")):
        predicted = predict_image(path)
        matrix[true_class][predicted] += 1

# ---- Print a readable confusion matrix ----
print("\nConfusion Matrix (corrected 0-to-1 preprocessing)")
print("Rows = actual class, Columns = predicted class\n")

header = "Actual \\ Predicted".ljust(20) + "".join(c.ljust(15) for c in class_names)
print(header)
print("-" * len(header))

total_correct = 0
total_count = 0
for true_class in class_names:
    row = matrix[true_class]
    row_total = sum(row.values())
    row_correct = row[true_class]
    total_correct += row_correct
    total_count += row_total

    accuracy_pct = (row_correct / row_total * 100) if row_total else 0
    row_str = true_class.ljust(20) + "".join(str(row[c]).ljust(15) for c in class_names)
    print(f"{row_str}   ({accuracy_pct:.0f}% correct, {row_total} images)")

print("-" * len(header))
print(f"\nOverall accuracy: {total_correct}/{total_count} = {total_correct/total_count*100:.1f}%")
