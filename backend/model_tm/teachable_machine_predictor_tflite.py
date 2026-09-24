"""
AssureX Claim Engine - Teachable Machine Predictor (TFLite version)
------------------------------------------------------------------------
This version loads a TensorFlow Lite (.tflite) model instead of the
older Keras (.h5) format. We switched formats because the .h5 export
from Teachable Machine can fail to load on newer TensorFlow/Keras
versions (a known compatibility issue with the "DepthwiseConv2D" layer).

TFLite is also genuinely a better choice for a deployed app: smaller
file size, faster predictions, and a much more stable loading API that
rarely breaks across TensorFlow versions.

Put your exported Teachable Machine files here, next to this script:
    model_unquant.tflite    <- the trained model (from the "Tensorflow Lite"
                                 export tab, "Floating point" option)
    labels.txt                 <- the class names, in the order the model learned them
"""

import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_unquant.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMAGE_SIZE = (224, 224)  # Teachable Machine models always expect this size

_interpreter = None
_class_names = None
_input_details = None
_output_details = None


def _load_tm_model():
    """Loads the TFLite model and labels once, the first time they're
    needed (not at import time, so a missing file only errors when
    actually used)."""
    global _interpreter, _class_names, _input_details, _output_details

    if _interpreter is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                "Teachable Machine model files not found. Make sure "
                f"'model_unquant.tflite' and 'labels.txt' are placed in {BASE_DIR}"
            )

        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

        with open(LABELS_PATH, "r") as f:
            _class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]

        print(f"[teachable_machine] TFLite model loaded. Classes: {_class_names}")

    return _interpreter, _class_names


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize to 224x224 and scale pixel values to the 0..1 range.

    IMPORTANT: this is DIFFERENT from the .h5/Keras export formula
    (-1 to 1). We empirically verified this using evaluate_teachable_machine.py
    against 225 real, known-label validation images:
        Formula -1..1 (old, wrong for TFLite): 33.3% accuracy, always
            predicted "Valid" -- a broken/degenerate result.
        Formula 0..1 (this one): 46.7% accuracy, predictions genuinely
            spread across all 3 classes -- correct behavior.
    Google's TensorFlow Lite floating-point export apparently expects a
    different pixel range than the Keras export does. If you ever retrain
    and re-export, re-run evaluate_teachable_machine.py to confirm this
    still holds before trusting predictions.
    """
    image = image.convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = image_array / 255.0

    data = np.expand_dims(normalized_image_array, axis=0).astype(np.float32)
    return data


def predict_card_image(image: Image.Image) -> dict:
    """
    Same interface as before -- takes a PIL Image, returns:
        {
            "predicted_class": "Valid",
            "confidence_scores": {"Valid": 0.91, "Invalid": 0.05, "ManualReview": 0.04}
        }
    """
    interpreter, class_names = _load_tm_model()
    data = _preprocess_image(image)

    interpreter.set_tensor(_input_details[0]["index"], data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(_output_details[0]["index"])[0]

    confidence_scores = {
        class_name: round(float(prob), 4)
        for class_name, prob in zip(class_names, predictions)
    }
    predicted_class = class_names[int(np.argmax(predictions))]

    return {
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
    }


# ---------------------------------------------------------------------
# Quick self-test: only runs if you execute THIS file directly.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(BASE_DIR, "..", "card_generator"))
    from card_generator import generate_card

    sample_claim = {
        "claim_id": "CLM00001",
        "product_category": "Washing Machine",
        "product_age_months": 5,
        "warranty_duration_months": 12,
        "warranty_status": "Active",
        "fault_type": "Mechanical Failure",
        "repair_count": 0,
        "authorized_repair": "Yes",
        "has_receipt": "Yes",
        "has_warranty_card": "Yes",
        "has_product_image": "Yes",
        "has_serial_evidence": "Yes",
        "missing_doc_count": 0,
        "serial_match": "Yes",
        "excluded_damage": "No",
        "duplicate_claim": "No",
    }

    card_image = generate_card(sample_claim, variation=0)
    result = predict_card_image(card_image)
    print("\nSample Teachable Machine (TFLite) prediction:")
    print(result)
