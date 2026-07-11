print(">>> LOADED api/app.py <<<")

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn using an XGBoost model.",
    version="1.0.0"
)

# Load the trained model
model = joblib.load("models/churn_model.joblib")


class Customer(BaseModel):
    CreditScore: int
    Geography: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!"
    }


@app.post("/predict")
def predict(customer: Customer):

    input_data = pd.DataFrame([customer.dict()])

    prediction = model.predict(input_data)

    result = "Customer Will Churn" if prediction[0] == 1 else "Customer Will Stay"

    return {
        "prediction": result
    }