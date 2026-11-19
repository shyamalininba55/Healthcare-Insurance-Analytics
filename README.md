# Healthcare Insurance Analytics System

## 1. Project Overview

The Healthcare Insurance Analytics System is a Python-based application that predicts healthcare insurance expenses, analyses customer feedback and discovers relationships between insurance policies and claim-cost categories.

The application provides an interactive dashboard using Streamlit.

## 2. Objectives

- Predict expected insurance expenses using customer and policy details.
- Identify important factors influencing insurance costs.
- Classify customer feedback into Positive, Negative and Neutral.
- Identify common customer complaints and satisfaction trends.
- Discover policy and claim-cost relationships using association rule mining.
- Evaluate predictive model performance.

## 3. Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- MLxtend
- TF-IDF
- Logistic Regression
- Random Forest Regression
- Linear Regression
- Apriori Algorithm

## 4. Project Modules

### Module 1: Insurance Expense Prediction
Uses Random Forest Regression to estimate annual insurance expenses based on customer demographics, medical information and policy details.

### Module 2: Customer Feedback Analysis
Uses TF-IDF and Logistic Regression for sentiment classification. Includes analysis of customer complaints and satisfaction.

### Module 3: Pattern Mining
Uses the Apriori algorithm to discover associations between policy characteristics and claim-cost categories.

### Module 4: Model Evaluation
Evaluates regression performance using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Compares Random Forest with Linear Regression.

### Module 5: Interactive Dashboard
Displays insurance expense trends, customer feedback analysis, predictions, discovered patterns and evaluation charts.

## 5. Project Structure

```text
Healthcare_Insurance_Analytics/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── insurance_customers.csv
│   └── customer_feedback.csv
│
└── docs/
    ├── project_documentation.md
    └── methodology.md
```

## 6. Installation

Install Python 3.13 or a compatible Python version.

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## 7. Execution

Run the application:

```bash
python -m streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## 8. Expected Outputs

- Predicted insurance expenses in Indian Rupees.
- Insurance cost-related feature importance.
- Positive, Negative and Neutral customer feedback analysis.
- Common customer complaints.
- Policy and claim-cost association rules.
- Regression evaluation metrics.
- Interactive analytical dashboard.

## 9. Dataset

The application uses two CSV files:

1. insurance_customers.csv – Customer demographics, policy details and insurance expenses.
2. customer_feedback.csv – Customer feedback text and sentiment labels.

Add the actual dataset source or explain how the synthetic data was generated here.

## 10. Limitations

- Predictions depend on the quality and representativeness of the training data.
- Sentiment classification may not correctly interpret sarcasm or complex language.
- Association rules identify relationships, not causation.
- Predictions are educational estimates and not actual insurance quotations.

## 11. Conclusion

The project combines predictive analytics, sentiment analysis and pattern mining in an interactive healthcare insurance analytics application to support understanding of insurance expenses and customer satisfaction.