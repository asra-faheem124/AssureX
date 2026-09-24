"""
AssureX Claim Engine - Full Combined Flask App
----------------------------------------------------
This is the endpoint that ties your ENTIRE AI pipeline together:

    claim JSON in
        -> Python model predicts (predictor.py)
        -> Claim Summary Card image generated (card_generator.py)
        -> Teachable Machine predicts from that image (teachable_machine_predictor.py)
        -> both predictions compared (compare_predictions.py)
        -> one combined JSON result out

Folder layout this file expects (adjust the sys.path lines below if
yours is different):

    backend/
        app.py                          <- this file (or merge into your existing app.py)
        model/
            predictor.py
            model.pkl, encoders.pkl, feature_columns.pkl
        model_tm/
            teachable_machine_predictor.py
            model_unquant.tflite, labels.txt
        card_generator/
            card_generator.py
        compare_predictions.py
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "model_tm"))
sys.path.append(os.path.join(BASE_DIR, "card_generator"))

from predictor import predict_claim                          # Python model
from teachable_machine_predictor_tflite import predict_card_image     # Teachable Machine
from card_generator import generate_card                        # Claim Summary Card
from compare_predictions import compare_predictions             # Comparison logic

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "AssureX backend is working..."

def decide_final_result(python_result, comparison):
    """
    Combines everything into ONE final decision, following the SRS's
    "Final Claim Decision" rule: consider both model results plus the
    consistency status, and output Likely Valid / Likely Invalid /
    Manual Review Required.

    This is intentionally simple for now -- it does NOT yet include
    warranty rule checks (expiry, exclusions, etc. as separate business
    rules), which we'll layer on next. For now it combines just the two
    models' agreement/disagreement.
    """
    status = comparison["consistency_status"]
    python_class = python_result["predicted_class"]

    if status == "Strong Match":
        final = "Likely Valid" if python_class == "Valid" else (
                 "Likely Invalid" if python_class == "Invalid" else "Manual Review Required")
    elif status == "Acceptable Match":
        final = "Likely Valid" if python_class == "Valid" else (
                 "Likely Invalid" if python_class == "Invalid" else "Manual Review Required")
    else:
        # Weak Match, Model Disagreement, or Uncertain Result -> always
        # send to a human. This is a deliberate safety choice: when the
        # two independent AIs don't clearly agree, we don't guess.
        final = "Manual Review Required"

    return final

@app.route("/api/claims/predict_full", methods=["POST"])
def predict_full():
    """
    Expects the same JSON claim body as /api/claims/predict.

    Returns:
    {
        "python_model": { "predicted_class": ..., "confidence_scores": {...} },
        "teachable_machine": { "predicted_class": ..., "confidence_scores": {...} },
        "comparison": {
            "classes_match": ...,
            "confidence_difference": ...,
            "consistency_status": ...
        },
        "final_decision": "Likely Valid" | "Likely Invalid" | "Manual Review Required"
    }
    """
    claim_data = request.get_json()

    if not claim_data:
        return jsonify({"error": "No claim data provided"}), 400

    try:
        # ---- Step 1: Python model prediction ----
        python_result = predict_claim(claim_data)

        # ---- Step 2: Generate the Claim Summary Card image ----
        # We need a claim_id on the card even if the frontend didn't send
        # one -- fall back to a placeholder so card generation never breaks.
        card_claim = {**claim_data, "claim_id": claim_data.get("claim_id", "PENDING")}
        card_image = generate_card(card_claim, variation=0)

        # ---- Step 3: Teachable Machine prediction from that image ----
        tm_result = predict_card_image(card_image)

        # ---- Step 4: Compare both predictions ----
        comparison = compare_predictions(python_result, tm_result)

        # ---- Step 5: Final combined decision ----
        final_decision = decide_final_result(python_result, comparison)

        return jsonify({
            "python_model": python_result,
            "teachable_machine": tm_result,
            "comparison": comparison,
            "final_decision": final_decision,
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] Unexpected error in /predict_full: {e}")
        return jsonify({"error": "Something went wrong while processing the claim."}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
