# Credit Card Fraud Detection & Risk Scoring

## Project Overview
A Machine Learning project that detects potentially fraudulent credit card transactions using classification algorithms.

## Problem Statement
Credit card fraud detection is a highly imbalanced classification problem. The goal is to identify fraudulent transactions while minimizing false alarms.

## Dataset
- Source: Kaggle Credit Card Fraud Detection
- Features: 30 input features
- Target: `Class`
- 0 = Genuine transaction
- 1 = Fraudulent transaction

Note: The dataset used in this project is a partial dataset containing 173,756 records after removing one row with a missing target value.

## Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM

## Techniques Used
- Data preprocessing
- Class imbalance handling
- Feature scaling
- Cross-validation
- Hyperparameter tuning
- Threshold tuning
- SHAP explainability

## Evaluation Metrics
- Precision
- Recall
- F1-score
- PR-AUC
- Accuracy

## Best Model Candidate
Tuned XGBoost achieved a comparative holdout PR-AUC of 0.8792. LightGBM achieved the highest F1-score of 0.8392 among the compared models.

These results are from the current experimental evaluation; an untouched final test set is still needed for independent validation.

## Technology Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, FastAPI, Docker.

## Project Status
Machine learning experiments completed. API and Docker execution verification pending.

## Disclaimer
This project is for educational and research purposes only. It is not a production-ready financial fraud detection service.
