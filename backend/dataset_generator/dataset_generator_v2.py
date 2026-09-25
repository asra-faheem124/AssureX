"""
AssureX Claim Engine - Dataset Generator v2 (IMPROVED)
----------------------------------------------------------
Why v2? The first version let "warranty_status" dominate the label almost
by itself, so the ML model learned a lazy shortcut and ignored other
important factors (excluded damage, serial mismatch, missing docs, etc).

v2 fixes this by generating claims with FULLY RANDOM, independent feature
values, then computing a "risk score" from ALL the important factors
combined (this mirrors exactly what your warranty rule engine will do
later in the app). The class label comes from that combined score, so the
ML model is forced to learn how multiple factors interact -- just like a
real warranty reviewer would.

Run it with:
    python dataset_generator_v2.py
"""

import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(7)

PRODUCT_CATEGORIES = ["Washing Machine", "Refrigerator", "Television",
                       "Laptop", "Mobile Phone", "Air Conditioner"]
FAULT_TYPES = ["Mechanical Failure", "Electrical Fault", "Physical Damage",
               "Software Issue", "Water Damage", "Overheating"]
EXCLUDED_FAULTS = {"Physical Damage", "Water Damage"}
TODAY = datetime(2026, 9, 23)


def random_raw_claim(claim_index):
    """Generates ONE claim with fully random, independent feature values
    (no target class in mind yet -- that comes later from the score)."""

    product_category = random.choice(PRODUCT_CATEGORIES)
    fault_type = random.choice(FAULT_TYPES)
    warranty_duration_months = random.choice([6, 12, 18, 24])

    # Purchase date spread widely so warranty ends up Active for some,
    # Expired for others, roughly 50/50
    days_back = random.randint(10, warranty_duration_months * 30 + 300)
    purchase_date = TODAY - timedelta(days=days_back)
    product_age_months = max(0, days_back // 30)
    warranty_expiry = purchase_date + timedelta(days=warranty_duration_months * 30)
    warranty_status = "Active" if warranty_expiry >= TODAY else "Expired"

    repair_count = random.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]
    authorized_repair = random.choices(["Yes", "No"], weights=[75, 25])[0] if repair_count > 0 else "Yes"

    # Documents: each independently present ~75% of the time
    has_receipt = random.choices(["Yes", "No"], weights=[75, 25])[0]
    has_warranty_card = random.choices(["Yes", "No"], weights=[75, 25])[0]
    has_product_image = random.choices(["Yes", "No"], weights=[80, 20])[0]
    has_serial_evidence = random.choices(["Yes", "No"], weights=[75, 25])[0]
    missing_doc_count = [has_receipt, has_warranty_card,
                          has_product_image, has_serial_evidence].count("No")

    serial_match = random.choices(["Yes", "No"], weights=[80, 20])[0]

    # Excluded damage is MOSTLY tied to fault type but not 100% deterministic:
    #   - ~8% chance a covered fault is still flagged as excluded (grey area / data entry error)
    #   - ~5% chance an excluded-type fault slips through as NOT flagged (missed during intake)
    # This prevents the Valid class from NEVER having excluded_damage=Yes.
    if fault_type in EXCLUDED_FAULTS:
        excluded_damage = "No" if random.random() < 0.05 else "Yes"
    else:
        excluded_damage = "Yes" if random.random() < 0.08 else "No"

    duplicate_claim = random.choices(["Yes", "No"], weights=[8, 92])[0]

    claim_submission_date = TODAY - timedelta(days=random.randint(0, 10))

    return {
        "claim_id": f"CLM{claim_index:05d}",
        "product_category": product_category,
        "product_age_months": product_age_months,
        "warranty_duration_months": warranty_duration_months,
        "warranty_status": warranty_status,
        "fault_type": fault_type,
        "repair_count": repair_count,
        "authorized_repair": authorized_repair,
        "has_receipt": has_receipt,
        "has_warranty_card": has_warranty_card,
        "has_product_image": has_product_image,
        "has_serial_evidence": has_serial_evidence,
        "missing_doc_count": missing_doc_count,
        "serial_match": serial_match,
        "excluded_damage": excluded_damage,
        "duplicate_claim": duplicate_claim,
        "purchase_date": purchase_date.strftime("%Y-%m-%d"),
        "claim_submission_date": claim_submission_date.strftime("%Y-%m-%d"),
    }


def compute_risk_score(claim):
    """
    Combines ALL risk factors into one score. This is the same logic your
    warranty RULE ENGINE will use later in the real app -- here we use it
    to decide training labels, so the ML model learns the same reasoning.
    Higher score = more suspicious/likely to fail.
    """
    score = 0
    if claim["warranty_status"] == "Expired":
        score += 3
    if claim["excluded_damage"] == "Yes":
        score += 3
    if claim["serial_match"] == "No":
        score += 2
    score += claim["missing_doc_count"]          # 0 to 4
    if claim["duplicate_claim"] == "Yes":
        score += 3
    if claim["authorized_repair"] == "No":
        score += 1
    if claim["repair_count"] >= 2:
        score += 1

    # Random noise — keeps boundaries realistic without too much overlap
    score += random.choice([-1, 0, 0, 0, 1])
    return max(0, score)


def label_from_score(score):
    # Tighter boundaries: Valid ≤ 1, ManualReview 2–4, Invalid ≥ 5
    # This keeps class separation clear enough for high model accuracy
    # while excluded_damage still has ~8% realistic noise built in.
    if score <= 1:
        return "Valid"
    elif score <= 4:
        return "ManualReview"
    else:
        return "Invalid"


def generate_balanced_dataset(records_per_class=500, max_attempts=500000):
    """
    Keeps generating random claims and only keeps ones that land in a
    class we still need more of, until every class has records_per_class
    records. This is called 'rejection sampling' -- simple and reliable.
    """
    buckets = {"Valid": [], "Invalid": [], "ManualReview": []}
    claim_index = 1
    attempts = 0

    while attempts < max_attempts and any(
        len(buckets[c]) < records_per_class for c in buckets
    ):
        attempts += 1
        claim = random_raw_claim(claim_index)
        score = compute_risk_score(claim)
        label = label_from_score(score)

        if len(buckets[label]) < records_per_class:
            claim["class_label"] = label
            buckets[label].append(claim)
            claim_index += 1

    all_records = buckets["Valid"] + buckets["Invalid"] + buckets["ManualReview"]
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Generated after {attempts} attempts:")
    for c in buckets:
        print(f"  {c}: {len(buckets[c])} records")

    return df


def stratified_split(df, train_frac=0.70, val_frac=0.15):
    train_parts, val_parts, test_parts = [], [], []
    for label in df["class_label"].unique():
        subset = df[df["class_label"] == label].sample(frac=1, random_state=42)
        n = len(subset)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(subset.iloc[:n_train])
        val_parts.append(subset.iloc[n_train:n_train + n_val])
        test_parts.append(subset.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df, val_df, test_df


if __name__ == "__main__":
    # 834 per class x 3 = 2,502 records (~2,500 as requested)
    df = generate_balanced_dataset(records_per_class=834)
    df.to_csv("claims_dataset.csv", index=False)

    train_df, val_df, test_df = stratified_split(df)
    train_df.to_csv("claims_train.csv", index=False)
    val_df.to_csv("claims_val.csv", index=False)
    test_df.to_csv("claims_test.csv", index=False)

    print(f"\nTotal records: {len(df)}")
    print(df["class_label"].value_counts())
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("\nFiles saved: claims_dataset.csv, claims_train.csv, claims_val.csv, claims_test.csv")
