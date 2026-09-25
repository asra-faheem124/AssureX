# AssureX Claim Engine

**AI-Powered Warranty Claim Validation System**  
Category: NextWave AI and ML | Theme: AI-Powered Document Ops

---

## What It Does

AssureX is a web-based warranty claim validation application that uses two independent AI models to evaluate claims and produce a final decision:

1. **Python Classification Model** (Random Forest) — trained on 1,500 structured claim records
2. **Google Teachable Machine Model** — trained on visual Claim Summary Card images
3. **Warranty Rule Engine** — validates business rules (expiry, exclusions, documents, serial numbers)
4. **Final Decision** — combines all three: `Likely Valid`, `Likely Invalid`, or `Manual Review Required`

---

## Project Structure

```
AssureX/
├── backend/
│   ├── app.py                        # Flask API — main entry point
│   ├── compare_predictions.py        # Model comparison logic
│   ├── model/
│   │   ├── train_model.py            # ML training script (3 algorithms compared)
│   │   ├── predictor.py              # Python model inference
│   │   ├── model.pkl                 # Saved Random Forest model
│   │   ├── encoders.pkl              # Label encoders
│   │   └── feature_columns.pkl       # Feature column order
│   ├── model_tm/
│   │   ├── teachable_machine_predictor_tflite.py
│   │   ├── model_unquant.tflite      # Exported TFLite model
│   │   └── labels.txt                # Class labels
│   ├── card_generator/
│   │   ├── card_generator.py         # Claim Summary Card generator
│   │   └── generate_training_cards.py
│   ├── dataset_generator/
│   │   └── dataset_generator_v2.py   # Generates the 1,500-record dataset
│   └── data/
│       ├── claims_dataset.csv        # Full 1,500-record dataset
│       ├── claims_train.csv          # 70% training split
│       ├── claims_val.csv            # 15% validation split
│       ├── claims_test.csv           # 15% test split
│       └── claim_cards/              # Claim Summary Card images (train/val/test)
├── frontend/
│   └── src/
│       └── components/
│           └── ClaimForm.jsx         # React claim submission form
├── requirements.txt
├── AI_USAGE.md
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for the React frontend)
- pip

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/asra-faheem124/AssureX.git
cd AssureX
git checkout Maheen
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies
```bash
cd frontend
npm install
```

---

## Running the Application

### Start the Flask backend
```bash
cd backend
python app.py
```
Backend runs at: `http://127.0.0.1:5000`

### Start the React frontend (separate terminal)
```bash
cd frontend
npm run dev
```
Frontend runs at: `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/health` | JSON health check |
| POST | `/api/claims/predict_full` | Full AI pipeline — returns both model predictions, comparison, and final decision |

### Example request body for `/api/claims/predict_full`
```json
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
```

---

## Retraining the Python Model

```bash
cd backend/model
python train_model.py
```

Trains 3 algorithms (Random Forest, Logistic Regression, SVM), selects the best, and saves `model.pkl`.

---

## Regenerating the Dataset

```bash
cd backend/dataset_generator
python dataset_generator_v2.py
```

---

## Regenerating Claim Summary Cards

```bash
cd backend/card_generator
python generate_training_cards.py
```

---

## Model Performance

| Model | Test Accuracy |
|-------|--------------|
| Random Forest (selected) | **90%** |
| Logistic Regression | 85% |
| SVM | 65% |

---

## Dataset

- **Total records:** 1,500 (500 Valid, 500 Invalid, 500 Manual Review)
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Training images:** 2,100 (2 variations × 1,050 training claims)

---

## Team

| Name | Role |
|------|------|
| Maheen | ML Model, Card Generator, Backend Pipeline |
| [Add team members] | [Add roles] |

---

## Deployment

The application is deployed at: *[Add deployment URL]*

Evaluator credentials: *[Add credentials]*

---

## Known Limitations

- Google Teachable Machine model accuracy is lower than the Python model due to limited image variation diversity
- OCR receipt scanning is planned but not yet implemented
- The application currently uses structured form input; future versions will support document upload and auto-extraction
