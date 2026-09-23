from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "AssureX Backend is Working!"


@app.route("/claim")
def get_claim():

    claim = {
        "claim_id": "CLM001",
        "product": "Samsung Washing Machine",
        "fault": "Motor Failure",
        "warranty_active": True,
        "receipt_available": True,
        "serial_match": True
    }

    return jsonify(claim)


if __name__ == "__main__":
    app.run(debug=True)