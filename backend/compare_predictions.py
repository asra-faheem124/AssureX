"""
AssureX Claim Engine - Model Comparison Logic
---------------------------------------------------
Takes the two independent predictions (Python model + Teachable Machine)
and compares them: do they agree on the class? How different are their
confidence scores? This produces the "Model Consistency Status" your
SRS requires: Strong Match / Acceptable Match / Weak Match /
Model Disagreement / Uncertain Result.

These thresholds are configurable on purpose (not hard-coded deep in
the logic) -- your SRS explicitly lists "changing a confidence threshold"
as a possible surprise modification evaluators may ask for, so keeping
them as named constants up top makes that a 10-second change, not a
code rewrite.
"""

# ---- Configurable thresholds ----
MIN_CONFIDENCE_FOR_CONFIDENT = 0.60   # below this, a prediction is "uncertain"
STRONG_MATCH_MAX_DIFF = 0.10           # confidence difference <= this -> Strong Match
ACCEPTABLE_MATCH_MAX_DIFF = 0.25       # confidence difference <= this -> Acceptable Match
WEAK_MATCH_MAX_DIFF = 0.45             # confidence difference <= this -> Weak Match
                                         # anything above WEAK_MATCH_MAX_DIFF -> Model Disagreement


def compare_predictions(python_result: dict, tm_result: dict) -> dict:
    """
    python_result and tm_result are both dictionaries shaped like:
        {"predicted_class": "Valid", "confidence_scores": {...}}
    (this is exactly what predictor.predict_claim() and
    teachable_machine_predictor.predict_card_image() both return --
    we deliberately made them match so this function doesn't need to
    know which model produced which result)

    Returns:
        {
            "classes_match": True/False,
            "python_top_class": "Valid",
            "python_top_confidence": 0.91,
            "tm_top_class": "Valid",
            "tm_top_confidence": 0.85,
            "confidence_difference": 0.06,
            "consistency_status": "Strong Match"
        }
    """
    python_class = python_result["predicted_class"]
    tm_class = tm_result["predicted_class"]

    python_top_conf = python_result["confidence_scores"][python_class]
    tm_top_conf = tm_result["confidence_scores"][tm_class]

    classes_match = python_class == tm_class
    confidence_difference = round(abs(python_top_conf - tm_top_conf), 4)

    # ---- Decide the consistency status ----
    # Logic, in plain words:
    #   1. If either model isn't even confident in ITS OWN answer -> Uncertain
    #   2. If the two models disagree on the class entirely -> Model Disagreement
    #   3. If they agree, how close are their confidence scores?
    if python_top_conf < MIN_CONFIDENCE_FOR_CONFIDENT or tm_top_conf < MIN_CONFIDENCE_FOR_CONFIDENT:
        status = "Uncertain Result"
    elif not classes_match:
        status = "Model Disagreement"
    elif confidence_difference <= STRONG_MATCH_MAX_DIFF:
        status = "Strong Match"
    elif confidence_difference <= ACCEPTABLE_MATCH_MAX_DIFF:
        status = "Acceptable Match"
    elif confidence_difference <= WEAK_MATCH_MAX_DIFF:
        status = "Weak Match"
    else:
        status = "Model Disagreement"

    return {
        "classes_match": classes_match,
        "python_top_class": python_class,
        "python_top_confidence": python_top_conf,
        "tm_top_class": tm_class,
        "tm_top_confidence": tm_top_conf,
        "confidence_difference": confidence_difference,
        "consistency_status": status,
    }


# ---------------------------------------------------------------------
# Quick self-test with a few made-up example scenarios
# ---------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        ("Both agree, both confident", 
         {"predicted_class": "Valid", "confidence_scores": {"Valid": 0.95, "Invalid": 0.03, "ManualReview": 0.02}},
         {"predicted_class": "Valid", "confidence_scores": {"Valid": 0.90, "Invalid": 0.05, "ManualReview": 0.05}}),

        ("Both agree, but confidence gap is large",
         {"predicted_class": "Invalid", "confidence_scores": {"Valid": 0.05, "Invalid": 0.90, "ManualReview": 0.05}},
         {"predicted_class": "Invalid", "confidence_scores": {"Valid": 0.30, "Invalid": 0.62, "ManualReview": 0.08}}),

        ("Models disagree entirely",
         {"predicted_class": "Valid", "confidence_scores": {"Valid": 0.80, "Invalid": 0.10, "ManualReview": 0.10}},
         {"predicted_class": "Invalid", "confidence_scores": {"Valid": 0.15, "Invalid": 0.75, "ManualReview": 0.10}}),

        ("Both models are unsure",
         {"predicted_class": "ManualReview", "confidence_scores": {"Valid": 0.35, "Invalid": 0.20, "ManualReview": 0.45}},
         {"predicted_class": "ManualReview", "confidence_scores": {"Valid": 0.30, "Invalid": 0.25, "ManualReview": 0.45}}),
    ]

    for description, python_result, tm_result in scenarios:
        comparison = compare_predictions(python_result, tm_result)
        print(f"\n{description}")
        print(f"  -> {comparison}")
