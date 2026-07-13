import os
import joblib
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import load_data, preprocess_data


# Load Dataset
df = load_data("../data/customer_churn.csv")

# Preprocess Dataset
X_train, X_test, y_train, y_test = preprocess_data(df)

# Create models folder if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Set MLflow Experiment
mlflow.set_experiment("Customer_Churn_Prediction")

with mlflow.start_run():

    # Create Model
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Print Metrics
    print("\n===== Model Performance =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Log Parameters
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("eval_metric", "logloss")

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log Model
    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model"
    )

    # Save Model
    joblib.dump(model, "../models/churn_model.joblib")

print("\nModel Saved Successfully!")


