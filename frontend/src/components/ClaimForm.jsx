import React, { useState } from "react";

// ---------------------------------------------------------------------
// CHANGE THIS if your Flask server runs on a different address/port
// ---------------------------------------------------------------------
const API_URL = "http://127.0.0.1:5000/api/claims/predict";

// These are the exact dropdown choices matching what the model was
// trained on. Keeping them as dropdowns (instead of free text) means the
// user can never accidentally type a word the model doesn't recognize.
const PRODUCT_CATEGORIES = ["Washing Machine", "Refrigerator", "Television",
  "Laptop", "Mobile Phone", "Air Conditioner"];
const FAULT_TYPES = ["Mechanical Failure", "Electrical Fault", "Physical Damage",
  "Software Issue", "Water Damage", "Overheating"];
const YES_NO = ["Yes", "No"];
const WARRANTY_STATUS = ["Active", "Expired"];

// The initial/default values for a brand-new form
const initialFormState = {
  product_category: PRODUCT_CATEGORIES[0],
  product_age_months: 6,
  warranty_duration_months: 12,
  warranty_status: "Active",
  fault_type: FAULT_TYPES[0],
  repair_count: 0,
  authorized_repair: "Yes",
  has_receipt: "Yes",
  has_warranty_card: "Yes",
  has_product_image: "Yes",
  has_serial_evidence: "Yes",
  missing_doc_count: 0,
  serial_match: "Yes",
  excluded_damage: "No",
  duplicate_claim: "No",
};

function ClaimForm() {
  // "formData" holds everything the user has typed/selected so far
  const [formData, setFormData] = useState(initialFormState);

  // "result" holds the prediction response once we get one back
  const [result, setResult] = useState(null);

  // "loading" tracks whether we're currently waiting for Flask to respond
  const [loading, setLoading] = useState(false);

  // "error" holds a message if something went wrong
  const [error, setError] = useState(null);

  // Runs every time ANY input in the form changes.
  // "name" comes from the input's `name` attribute, "value" from what
  // the user typed/selected.
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev, // keep all the other fields as they were
      [name]: type === "number" ? Number(value) : value,
    }));
  };

  // Automatically keep missing_doc_count in sync with the 4 document
  // toggles, so the user doesn't have to calculate it by hand.
  const recalculateMissingDocs = (data) => {
    const docFields = ["has_receipt", "has_warranty_card",
      "has_product_image", "has_serial_evidence"];
    const missingCount = docFields.filter((f) => data[f] === "No").length;
    return { ...data, missing_doc_count: missingCount };
  };

  const handleDocChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => recalculateMissingDocs({ ...prev, [name]: value }));
  };

  // Runs when the user clicks "Submit Claim"
  const handleSubmit = async (e) => {
    e.preventDefault(); // stop the browser from doing a full page reload
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        // Flask sent back an error (e.g. missing field) -- show it
        setError(data.error || "Something went wrong.");
      } else {
        setResult(data);
      }
    } catch (err) {
      // This happens if Flask isn't running at all, or CORS is blocked
      setError("Could not reach the server. Is Flask running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  // Small helper to pick a color per predicted class -- just for the UI
  const classColor = {
    Valid: "#1e7e34",
    Invalid: "#c62828",
    ManualReview: "#e0a800",
  };

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: 20, fontFamily: "Arial, sans-serif" }}>
      <h2>Submit a Warranty Claim</h2>

      <form onSubmit={handleSubmit}>
        <FormRow label="Product Category">
          <select name="product_category" value={formData.product_category} onChange={handleChange}>
            {PRODUCT_CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </FormRow>

        <FormRow label="Product Age (months)">
          <input type="number" name="product_age_months" min="0"
            value={formData.product_age_months} onChange={handleChange} />
        </FormRow>

        <FormRow label="Warranty Duration (months)">
          <input type="number" name="warranty_duration_months" min="1"
            value={formData.warranty_duration_months} onChange={handleChange} />
        </FormRow>

        <FormRow label="Warranty Status">
          <select name="warranty_status" value={formData.warranty_status} onChange={handleChange}>
            {WARRANTY_STATUS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </FormRow>

        <FormRow label="Fault Type">
          <select name="fault_type" value={formData.fault_type} onChange={handleChange}>
            {FAULT_TYPES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </FormRow>

        <FormRow label="Excluded Damage?">
          <select name="excluded_damage" value={formData.excluded_damage} onChange={handleChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Repair Count">
          <input type="number" name="repair_count" min="0"
            value={formData.repair_count} onChange={handleChange} />
        </FormRow>

        <FormRow label="Authorized Repair?">
          <select name="authorized_repair" value={formData.authorized_repair} onChange={handleChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Has Receipt?">
          <select name="has_receipt" value={formData.has_receipt} onChange={handleDocChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Has Warranty Card?">
          <select name="has_warranty_card" value={formData.has_warranty_card} onChange={handleDocChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Has Product Image?">
          <select name="has_product_image" value={formData.has_product_image} onChange={handleDocChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Has Serial Evidence?">
          <select name="has_serial_evidence" value={formData.has_serial_evidence} onChange={handleDocChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Serial Number Matches?">
          <select name="serial_match" value={formData.serial_match} onChange={handleChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <FormRow label="Duplicate Claim Suspected?">
          <select name="duplicate_claim" value={formData.duplicate_claim} onChange={handleChange}>
            {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </FormRow>

        <p style={{ fontSize: 13, color: "#666" }}>
          Missing documents detected automatically: <b>{formData.missing_doc_count}</b>
        </p>

        <button type="submit" disabled={loading} style={{
          padding: "10px 24px", fontSize: 16, cursor: "pointer",
          backgroundColor: "#2b6cb0", color: "white", border: "none", borderRadius: 4,
        }}>
          {loading ? "Analyzing Claim..." : "Submit Claim"}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: 20, padding: 12, background: "#fdecea", color: "#c62828", borderRadius: 4 }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 24, padding: 20, border: "1px solid #ddd", borderRadius: 8 }}>
          <h3 style={{ margin: 0 }}>
            Prediction:{" "}
            <span style={{ color: classColor[result.predicted_class] }}>
              {result.predicted_class}
            </span>
          </h3>

          <div style={{ marginTop: 16 }}>
            {Object.entries(result.confidence_scores).map(([label, score]) => (
              <ConfidenceBar key={label} label={label} score={score} color={classColor[label]} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Small reusable component: a label on the left, the input on the right
function FormRow({ label, children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
      <label style={{ width: 220, fontSize: 14 }}>{label}</label>
      {children}
    </div>
  );
}

// Small reusable component: a horizontal bar showing a confidence score
function ConfidenceBar({ label, score, color }) {
  const percent = Math.round(score * 100);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div style={{ background: "#eee", borderRadius: 4, height: 10 }}>
        <div style={{
          width: `${percent}%`, background: color, height: "100%",
          borderRadius: 4, transition: "width 0.3s ease",
        }} />
      </div>
    </div>
  );
}

export default ClaimForm;
