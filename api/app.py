print(">>> LOADED api/app.py <<<")

import download_from_s3
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a bank customer will churn using an XGBoost model.",
    version="1.0.0"
)

# Load trained model
model = joblib.load("models/churn_model.joblib")


class Customer(BaseModel):
    CreditScore: int = Field(example=650)
    Geography: int = Field(
        example=1,
        description="0 = France, 1 = Germany, 2 = Spain"
    )
    Gender: int = Field(
        example=1,
        description="0 = Female, 1 = Male"
    )
    Age: int = Field(example=35)
    Tenure: int = Field(example=5)
    Balance: float = Field(example=75000.50)
    NumOfProducts: int = Field(example=2)
    HasCrCard: int = Field(
        example=1,
        description="0 = No, 1 = Yes"
    )
    IsActiveMember: int = Field(
        example=1,
        description="0 = No, 1 = Yes"
    )
    EstimatedSalary: float = Field(example=50000.75)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running successfully!"
    }


@app.post("/predict")
def predict(customer: Customer):

    input_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(input_data)

    result = (
        "Customer Will Churn"
        if prediction[0] == 1
        else "Customer Will Stay"
    )

    return {
        "prediction": result
    }