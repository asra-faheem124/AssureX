"""
AssureX Claim Engine - Teachable Machine Preprocessing Diagnostic
--------------------------------------------------------------------
WHY THIS SCRIPT EXISTS:
Your model works correctly when tested directly on the Teachable
Machine website (Invalid ~perfect, ManualReview reasonably mixed,
Valid distinguishable) -- but our Python/Flask code is heavily
biased toward "Valid" for almost everything. Since the SAME trained
model behaves DIFFERENTLY in the browser vs our code, the model
itself is not the problem -- our image PREPROCESSING code is.

This script tests 3 different ways of preparing an image for the
model against your REAL validation images (which have known correct
labels, since they're organized by folder: val/Valid/, val/Invalid/,
val/ManualReview/). Whichever preprocessing method scores highest
(and isn't just guessing one class every time) is the correct one.

Run this from your model_tm/ folder (needs the val/ card images to
already exist under ../data/claim_cards/val/):
    python evaluate_teachable_machine.py
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

print("Model input details (useful for debugging):")
print(f"  dtype: {input_details[0]['dtype']}")
print(f"  shape: {input_details[0]['shape']}")
print(f"  quantization (scale, zero_point): {input_details[0]['quantization']}")
print(f"Classes (from labels.txt): {class_names}\n")


def load_and_resize(path):
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.float32)


# The 3 candidate preprocessing formulas we're testing
PREPROCESSORS = {
    "A: scaled -1 to 1  [current code uses this]": lambda arr: (arr / 127.5) - 1,
    "B: scaled 0 to 1": lambda arr: arr / 255.0,
    "C: raw 0 to 255 (no scaling)": lambda arr: arr,
}


def predict_with(preprocess_fn, image_array):
    data = preprocess_fn(image_array)
    data = np.expand_dims(data, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_index = int(np.argmax(output))
    return class_names[predicted_index]


# Collect ALL validation images with their TRUE label (taken from the
# folder they're sitting in -- val/Valid/, val/Invalid/, val/ManualReview/)
test_images = []
for class_name in class_names:
    folder = os.path.join(VAL_DIR, class_name)
    for path in glob.glob(os.path.join(folder, "*.png")):
        test_images.append((path, class_name))

print(f"Found {len(test_images)} validation images with known labels.\n")
print("=" * 70)

if len(test_images) == 0:
    print("No validation images found! Check that VAL_DIR points to the "
          "right folder and that generate_training_cards.py has been run.")
else:
    # Test each preprocessing formula against ALL validation images
    for method_name, preprocess_fn in PREPROCESSORS.items():
        correct = 0
        predictions_by_class = {c: 0 for c in class_names}

        for path, true_label in test_images:
            image_array = load_and_resize(path)
            predicted = predict_with(preprocess_fn, image_array)
            predictions_by_class[predicted] += 1
            if predicted == true_label:
                correct += 1

        accuracy = correct / len(test_images) * 100
        print(f"\n{method_name}")
        print(f"  Accuracy vs known labels: {accuracy:.1f}%  ({correct}/{len(test_images)} correct)")
        print(f"  How many times each class was predicted: {predictions_by_class}")

    print("\n" + "=" * 70)
    print("READ THIS: the winning formula is the one with the HIGHEST accuracy")
    print("AND a prediction distribution that's roughly spread across all 3")
    print("classes (not just guessing one class every time). Tell me which")
    print("letter (A / B / C) wins, and we'll update the real predictor code.")
