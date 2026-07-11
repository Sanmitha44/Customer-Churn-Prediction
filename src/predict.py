import joblib
import pandas as pd


model = joblib.load("../models/churn_model.joblib")


def predict(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    if prediction[0] == 1:
        return "Customer Will Leave"

    return "Customer Will Stay"