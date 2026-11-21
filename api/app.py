
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-based fraud prediction service",
    version="1.0"
)

# Load trained model artifacts
artifacts = joblib.load("artifacts/fraud_model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
threshold = artifacts["threshold"]


class Transaction(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    input_data = transaction.features

    missing = set(features) - set(input_data)
    extra = set(input_data) - set(features)

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": list(missing),
                "unexpected_features": list(extra)
            }
        )

    # Preserve training feature order
    df = pd.DataFrame(
        [[input_data[col] for col in features]],
        columns=features
    )

    # Scale Amount and Time
    df[["Amount", "Time"]] = scaler.transform(
        df[["Amount", "Time"]]
    )

    probability = float(model.predict_proba(df)[0, 1])

    prediction = int(probability >= threshold)

    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Genuine",
        "threshold": threshold
    }
