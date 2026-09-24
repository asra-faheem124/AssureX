"""
AssureX Claim Engine - Python Model Training Script
----------------------------------------------------
This script does 6 things, in order:

1. Loads the training/validation/test CSVs we generated earlier.
2. Converts text columns (like "Yes"/"No", "Active"/"Expired") into numbers,
   because ML models can only understand numbers, not words.
3. Trains THREE different algorithms and compares them (the SRS requires
   comparing at least 3 algorithms before picking one).
4. Evaluates each one on the validation set (accuracy, precision, recall, F1).
5. Picks the best-performing algorithm.
6. Saves the trained model + the encoders to disk, so the Flask backend can
   load them later and make real predictions.

Run it with:
    python train_model.py

Output files (in ../model/ folder):
    model.pkl              -> the trained model itself
    encoders.pkl            -> the encoders used to convert text -> numbers
    feature_columns.pkl     -> the exact list/order of columns the model expects
    model_report.txt        -> a human-readable comparison report (for your project report!)
"""

import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

# ---------------------------------------------------------------------
# 0. File paths -- adjust these if your folders are named differently
# ---------------------------------------------------------------------
DATA_DIR = "../data"
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "claims_train.csv")
VAL_PATH = os.path.join(DATA_DIR, "claims_val.csv")
TEST_PATH = os.path.join(DATA_DIR, "claims_test.csv")

# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test rows")

# Columns we will NOT feed to the model (they're identifiers/dates, not
# meaningful patterns for classification)
DROP_COLS = ["claim_id", "purchase_date", "claim_submission_date"]

# Columns that are text/categories and need to be converted to numbers
CATEGORICAL_COLS = ["product_category", "warranty_status", "fault_type",
                     "authorized_repair", "has_receipt", "has_warranty_card",
                     "has_product_image", "has_serial_evidence", "serial_match",
                     "excluded_damage", "duplicate_claim"]

TARGET_COL = "class_label"

# ---------------------------------------------------------------------
# 2. Encode categorical (text) columns into numbers
# ---------------------------------------------------------------------
# We fit each encoder ONLY on the training data, then reuse it on val/test.
# This mirrors how it must work in production: the model never "sees" the
# val/test answers while learning.

encoders = {}

def encode_columns(df, fit=False):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if fit:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc
        else:
            enc = encoders[col]
            df[col] = enc.transform(df[col])
    return df

train_enc = encode_columns(train_df, fit=True)
val_enc = encode_columns(val_df, fit=False)
test_enc = encode_columns(test_df, fit=False)

# Encode the target label (Valid/Invalid/ManualReview -> 0/1/2) too
target_encoder = LabelEncoder()
train_enc[TARGET_COL] = target_encoder.fit_transform(train_enc[TARGET_COL])
val_enc[TARGET_COL] = target_encoder.transform(val_enc[TARGET_COL])
test_enc[TARGET_COL] = target_encoder.transform(test_enc[TARGET_COL])
encoders["class_label"] = target_encoder

# ---------------------------------------------------------------------
# 3. Split into features (X) and answer (y)
# ---------------------------------------------------------------------
feature_cols = [c for c in train_enc.columns if c not in DROP_COLS + [TARGET_COL]]

X_train, y_train = train_enc[feature_cols], train_enc[TARGET_COL]
X_val, y_val = val_enc[feature_cols], val_enc[TARGET_COL]
X_test, y_test = test_enc[feature_cols], test_enc[TARGET_COL]

print(f"Features used ({len(feature_cols)}): {feature_cols}")

# ---------------------------------------------------------------------
# 4. Train & compare 3 algorithms on the VALIDATION set
# ---------------------------------------------------------------------
candidates = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
report_lines = []
report_lines.append("AssureX Claim Engine - Model Comparison Report\n")
report_lines.append("=" * 55 + "\n")

for name, clf in candidates.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="macro")
    rec = recall_score(y_val, preds, average="macro")
    f1 = f1_score(y_val, preds, average="macro")

    results[name] = {"model": clf, "accuracy": acc, "f1": f1}

    report_lines.append(f"\nAlgorithm: {name}")
    report_lines.append(f"  Validation Accuracy : {acc:.4f}")
    report_lines.append(f"  Validation Precision: {prec:.4f}")
    report_lines.append(f"  Validation Recall   : {rec:.4f}")
    report_lines.append(f"  Validation F1-score : {f1:.4f}")

    print(f"{name}: accuracy={acc:.4f}, f1={f1:.4f}")

# ---------------------------------------------------------------------
# 5. Pick the best model (highest F1-score on validation set)
# ---------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]["f1"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name}")
report_lines.append(f"\n\nSELECTED MODEL: {best_name}\n")

# ---------------------------------------------------------------------
# 6. Final evaluation of the BEST model on the TEST set (unseen data)
# ---------------------------------------------------------------------
test_preds = best_model.predict(X_test)

test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="macro")
cm = confusion_matrix(y_test, test_preds)
class_report = classification_report(
    y_test, test_preds, target_names=target_encoder.classes_
)

report_lines.append(f"\nFinal Test Set Evaluation ({best_name})")
report_lines.append(f"  Test Accuracy: {test_acc:.4f}")
report_lines.append(f"  Test F1-score: {test_f1:.4f}")
report_lines.append(f"\nConfusion Matrix (rows=actual, cols=predicted):")
report_lines.append(f"  Classes order: {list(target_encoder.classes_)}")
report_lines.append(str(cm))
report_lines.append("\nClassification Report:\n")
report_lines.append(class_report)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test F1-score: {test_f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\n", class_report)

# ---------------------------------------------------------------------
# 7. Save everything the Flask backend will need later
# ---------------------------------------------------------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))

with open(os.path.join(MODEL_DIR, "model_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print(f"\nSaved model.pkl, encoders.pkl, feature_columns.pkl, model_report.txt in {MODEL_DIR}/")
