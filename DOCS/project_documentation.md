# Healthcare Insurance Analytics System

## 1. Introduction

The Healthcare Insurance Analytics System is a Python-based data analytics application developed to analyse healthcare insurance expenses and customer feedback.

The system combines machine learning, sentiment analysis and association rule mining to provide useful insights through an interactive Streamlit dashboard.

## 2. Problem Statement

Healthcare insurance companies handle large amounts of customer, policy and expense information.

Analysing this information manually makes it difficult to estimate insurance expenses, understand customer satisfaction and identify patterns in claim costs.

This project addresses these challenges using data analytics and machine learning techniques.

## 3. Objectives

- Predict healthcare insurance expenses using customer and policy information.
- Identify important factors influencing insurance expenses.
- Analyse customer feedback and classify sentiment.
- Identify common customer complaints.
- Discover relationships between policy characteristics and claim-cost categories.
- Present analytical results through an interactive dashboard.

## 4. System Modules

### Module 1: Insurance Expense Prediction

Uses Random Forest Regression to estimate insurance expenses based on:
- Age
- BMI
- Number of dependents
- Smoking status
- Policy type
- Network type
- Deductible amount
- Coverage limit
- Previous claim count

### Module 2: Customer Feedback Analysis

Uses TF-IDF and Logistic Regression to classify customer feedback into:
- Positive
- Negative
- Neutral

Also identifies common complaints related to claim delays, customer support, claim rejection, high premiums and policy issues.

### Module 3: Pattern Mining

Uses the Apriori algorithm to discover frequent relationships between:
- Policy types
- Smoking status
- Age groups
- Insurance cost categories

Association rules are evaluated using support, confidence and lift.

### Module 4: Model Evaluation

Evaluates the insurance expense prediction model using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Compares Random Forest Regression with Linear Regression.

### Module 5: Interactive Dashboard

The Streamlit dashboard provides:
- Insurance expense overview
- Individual expense prediction
- Customer feedback analysis
- Pattern mining results
- Model evaluation charts

## 5. Software Requirements

- Python 3.13
- Visual Studio Code
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLxtend

## 6. Dataset Description

The project uses two CSV datasets.

### insurance_customers.csv

Contains customer demographic information, insurance policy details, claim information and insurance expenses.

### customer_feedback.csv

Contains customer feedback text and sentiment labels.

## 7. Expected Results

The system generates:
- Estimated annual insurance expenses in Indian Rupees.
- Important insurance cost-related features.
- Customer sentiment distribution.
- Common customer complaint categories.
- Policy and cost association rules.
- Regression model evaluation metrics.

## 8. Conclusion

The Healthcare Insurance Analytics System demonstrates how machine learning and data mining techniques can be combined to analyse insurance expenses and customer satisfaction.

The interactive dashboard makes the analytical results easier to understand and explore.

## 9. Limitations

- Model performance depends on the quality of the available dataset.
- Sentiment analysis may not correctly interpret sarcasm or complex expressions.
- Association rules indicate relationships, not causation.
- Predictions are educational estimates and should not be treated as actual insurance quotations.