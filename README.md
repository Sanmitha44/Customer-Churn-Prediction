# 🚀 Customer Churn Prediction API

A Machine Learning REST API built using **FastAPI**, **XGBoost**, **Docker**, and **AWS Elastic Beanstalk** for predicting customer churn.

---

## 📌 Project Overview

This project predicts whether a customer is likely to churn based on customer information using a trained XGBoost classification model.

The application is deployed as a REST API using FastAPI and Docker on AWS Elastic Beanstalk.

---

## 🛠 Tech Stack

- Python
- FastAPI
- XGBoost
- Scikit-learn
- Pandas
- MLflow
- Joblib
- Docker
- AWS Elastic Beanstalk
- Swagger UI

---

## 📂 Project Structure

```
Customer-Churn-Prediction
│
├── api/
│   └── app.py
│
├── models/
│   └── churn_model.joblib
│
├── data/
│
├── src/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .dockerignore
```

---

## 🚀 Features

- Customer churn prediction
- REST API using FastAPI
- Interactive Swagger documentation
- Docker containerization
- AWS Elastic Beanstalk deployment
- MLflow integrated for model management

---

## 📥 API Endpoint

### Predict Customer Churn

**POST**

```
/predict
```

### Sample Request

```json
{
  "CreditScore": 650,
  "Geography": 1,
  "Gender": 1,
  "Age": 35,
  "Tenure": 5,
  "Balance": 75000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

---

### Sample Response

```json
{
    "prediction": "Customer Will Stay"
}
```

or

```json
{
    "prediction": "Customer Will Churn"
}
```

---

## 🌐 Deployment

The application is containerized using Docker and deployed on **AWS Elastic Beanstalk**.

Interactive API documentation is available through Swagger UI.

---

## 🧠 Machine Learning Model

- Algorithm: XGBoost Classifier
- Task: Binary Classification
- Output:
  - Customer Will Stay
  - Customer Will Churn

---

## 📖 Future Improvements

- Probability score prediction
- User Authentication
- Database integration
- Batch prediction
- CI/CD Pipeline using GitHub Actions

---

## 👩‍💻 Author

**Sanmitha Kulal**

Computer Science (Data Science)

Alva's Institute of Engineering and Technology

GitHub: https://github.com/Sanmitha44
