"""
AssureX Claim Engine - Predictor Module
-----------------------------------------
This file's ONLY job is: load the trained model (once, when the app starts)
and provide one function, predict_claim(), that Flask routes can call to
get a prediction for a brand-new claim.

We keep this separate from app.py on purpose -- it keeps your Flask routes
clean and makes this logic reusable/testable on its own.
"""

import os
import joblib
import pandas as pd

# ---------------------------------------------------------------------
# 1. Figure out where the model files live, relative to THIS file.
#    This means it works no matter which folder you run Flask from.
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../model/
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# ---------------------------------------------------------------------
# 2. Load everything ONCE when this file is first imported.
#    (Loading a model from disk takes time -- we do NOT want to reload
#    it on every single request, only once when the server starts.)
# ---------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_columns = joblib.load(FEATURES_PATH)

target_encoder = encoders["class_label"]          # converts 0/1/2 back to Valid/Invalid/ManualReview
CATEGORICAL_COLS = [c for c in encoders.keys() if c != "class_label"]

print(f"[predictor] Model loaded successfully. Expects {len(feature_columns)} features.")
print(f"[predictor] Classes: {list(target_encoder.classes_)}")


def predict_claim(claim_data: dict):
    """
    Takes ONE claim as a plain Python dictionary, e.g.:

        {
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
            "duplicate_claim": "No"
        }

    Returns a dictionary with the predicted class and confidence scores
    for all three classes, e.g.:

        {
            "predicted_class": "Valid",
            "confidence_scores": {
                "Valid": 0.91,
                "Invalid": 0.05,
                "ManualReview": 0.04
            }
        }
    """

    # ---- Step A: turn the single claim dict into a 1-row table ----
    # The model was trained on a table (many rows), so even a single new
    # claim needs to be shaped like a table with exactly 1 row.
    df = pd.DataFrame([claim_data])

    # ---- Step B: make sure every expected column exists ----
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields in claim data: {missing}")

    # ---- Step C: encode the text columns using the SAME encoders ----
    # used during training (this is exactly why we saved encoders.pkl!)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            enc = encoders[col]
            # If a brand-new value appears that the encoder has never seen,
            # this would crash -- so we catch that clearly instead of a
            # confusing error.
            unseen = set(df[col]) - set(enc.classes_)
            if unseen:
                raise ValueError(
                    f"Unrecognized value(s) {unseen} in column '{col}'. "
                    f"Expected one of: {list(enc.classes_)}"
                )
            df[col] = enc.transform(df[col])

    # ---- Step D: put columns in the EXACT same order used in training ----
    df = df[feature_columns]

    # ---- Step E: ask the model to predict ----
    predicted_index = model.predict(df)[0]            # e.g. 2
    predicted_class = target_encoder.inverse_transform([predicted_index])[0]  # e.g. "Valid"

    # ---- Step F: get confidence scores for ALL three classes ----
    probabilities = model.predict_proba(df)[0]          # e.g. [0.05, 0.04, 0.91]
    confidence_scores = {
        class_name: round(float(prob), 4)
        for class_name, prob in zip(target_encoder.classes_, probabilities)
    }

    return {
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
    }


# ---------------------------------------------------------------------
# Quick self-test: only runs if you execute THIS file directly
# (python predictor.py) -- lets you test the model without Flask at all.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_claim = {
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

    result = predict_claim(sample_claim)
    print("\nSample prediction result:")
    print(result)
