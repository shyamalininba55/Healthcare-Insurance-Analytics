# Project Methodology

## 1. Methodology Overview

The Healthcare Insurance Analytics System follows a data analytics workflow involving data collection, preprocessing, feature engineering, model development, evaluation and dashboard visualization.

## 2. Data Collection

Two datasets are used:

1. Insurance customer dataset containing demographic, policy, claim and expense information.
2. Customer feedback dataset containing feedback text and sentiment labels.

## 3. Data Preprocessing

The following preprocessing operations are performed:

- Load CSV files using Pandas.
- Handle missing values using SimpleImputer.
- Convert categorical variables into numerical representations using OneHotEncoder.
- Scale numerical variables using StandardScaler.
- Clean customer feedback text and sentiment labels.

## 4. Feature Engineering

Additional features are created to improve insurance expense analysis.

### Age Group

Customers are grouped into:
- 18–30
- 31–45
- 46–60
- 60+

### BMI Group

Customers are classified into:
- Underweight
- Normal
- Overweight
- Obese

### Smoking and BMI Interaction

Combines smoking status and BMI to represent their interaction.

### Claim Frequency Group

Customers are classified according to previous claim counts:
- No claims
- 1–2 claims
- 3+ claims

## 5. Insurance Expense Prediction

### Algorithm: Random Forest Regression

Random Forest Regression is used to predict insurance expenses.

The dataset is divided into:
- Training dataset: 80%
- Testing dataset: 20%

The model uses 250 decision trees with a maximum depth of 12.

### Baseline Comparison

Linear Regression is implemented as a baseline model to compare predictive performance.

## 6. Customer Feedback Sentiment Analysis

### TF-IDF

Term Frequency-Inverse Document Frequency converts customer feedback text into numerical features.

### Logistic Regression

Logistic Regression classifies feedback into Positive, Negative and Neutral categories.

A phrase-based correction is also used for certain common expressions.

Model accuracy and classification metrics are used to assess performance.

## 7. Pattern Mining

### Apriori Algorithm

The Apriori algorithm identifies frequently occurring combinations of insurance policy characteristics and cost categories.

### Association Rule Metrics

- Support: Frequency of an item combination.
- Confidence: Strength of a rule.
- Lift: Strength of an association compared with independent occurrence.

## 8. Model Evaluation

The regression model is evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted expenses.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

### R² Score

Measures how much variation in insurance expenses is explained by the model.

## 9. Visualization

Streamlit provides the user interface.

Matplotlib and Seaborn are used for charts, distributions and model evaluation visualizations.

## 10. Execution Workflow

1. Load insurance and feedback datasets.
2. Preprocess the customer data.
3. Perform feature engineering.
4. Train the Random Forest regression model.
5. Train the Linear Regression baseline.
6. Train the sentiment classification model.
7. Perform association rule mining.
8. Evaluate model performance.
9. Display results through the Streamlit dashboard.

## 11. Limitations

The results depend on the quality and representativeness of the supplied data. The sentiment correction rules cover only selected expressions, and the system is intended for educational analytics rather than real-world insurance decisions.