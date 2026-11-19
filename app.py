import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer

from mlxtend.frequent_patterns import apriori, association_rules


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Healthcare Insurance Analytics",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Insurance Analytics System")
st.caption("Insurance Expense Prediction and Customer Feedback Analytics")
st.info(
    "Educational project using the supplied dataset. "
    "Predictions are estimates, not actual insurance quotations."
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def inr(value):
    return f"₹{value:,.2f}"


def get_sentiment_rule(text):
    """
    Small rule-based correction for clear, common phrases.
    The ML model is used when the wording is not clearly covered.
    """
    text = str(text).lower().strip()

    positive_phrases = [
        "good", "great", "excellent", "very helpful",
        "helpful", "friendly", "fast", "quick",
        "satisfied", "happy", "wonderful", "easy",
        "love", "awesome", "supportive", "appreciate",
        "smooth", "efficient", "professional", "fantastic",
        "thank you", "thanks"
    ]

    negative_phrases = [
        "bad", "poor", "terrible", "worst",
        "delay", "delayed", "rejected", "unhappy",
        "angry", "disappointed", "slow", "rude",
        "fraud", "problem", "complaint", "confusing",
        "not helpful", "not satisfied", "waste",
        "pathetic", "awful", "denied"
    ]

    # Handle common negation phrases before checking individual words.
    negative_patterns = [
        "not good", "not great", "not helpful",
        "not happy", "not satisfied", "not excellent",
        "never helpful", "poor service"
    ]

    positive_patterns = [
        "not bad", "no problem", "not disappointed"
    ]

    if any(phrase in text for phrase in negative_patterns):
        return "Negative"

    if any(phrase in text for phrase in positive_patterns):
        return "Positive"

    positive_hits = sum(
        1 for phrase in positive_phrases if phrase in text
    )
    negative_hits = sum(
        1 for phrase in negative_phrases if phrase in text
    )

    if positive_hits > negative_hits:
        return "Positive"

    if negative_hits > positive_hits:
        return "Negative"

    return None


# ==================================================
# LOAD DATA
# ==================================================

DATA = Path("data")


@st.cache_data
def load_data():
    customers = pd.read_csv(DATA / "insurance_customers.csv")
    feedback_data = pd.read_csv(DATA / "customer_feedback.csv")
    return customers, feedback_data


try:
    df, feedback = load_data()
except FileNotFoundError as error:
    st.error(
        "Dataset file not found. Make sure these files exist: "
        "data/insurance_customers.csv and data/customer_feedback.csv"
    )
    st.stop()


# Validate required columns
required_customer_columns = [
    "age", "bmi", "children", "deductible", "coverage_limit",
    "claim_count", "smoker", "sex", "region", "policy_type",
    "network_type", "charges"
]

required_feedback_columns = ["feedback_text", "sentiment"]

missing_customer = [
    col for col in required_customer_columns if col not in df.columns
]
missing_feedback = [
    col for col in required_feedback_columns if col not in feedback.columns
]

if missing_customer:
    st.error(f"Missing columns in insurance_customers.csv: {missing_customer}")
    st.stop()

if missing_feedback:
    st.error(f"Missing columns in customer_feedback.csv: {missing_feedback}")
    st.stop()

# Clean text and labels
feedback["feedback_text"] = feedback["feedback_text"].fillna("").astype(str)
feedback["sentiment"] = (
    feedback["sentiment"].fillna("").astype(str).str.strip().str.capitalize()
)

valid_sentiments = ["Positive", "Negative", "Neutral"]
feedback = feedback[feedback["sentiment"].isin(valid_sentiments)].copy()
feedback = feedback[feedback["feedback_text"].str.strip() != ""]

if feedback["sentiment"].nunique() < 2:
    st.error("The feedback dataset must contain at least two sentiment classes.")
    st.stop()


# ==================================================
# FEATURE ENGINEERING
# ==================================================

df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 30, 45, 60, 100],
    labels=["18-30", "31-45", "46-60", "60+"]
).astype(str)

df["bmi_group"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
).astype(str)

df["smoker_bmi_interaction"] = (
    (df["smoker"].astype(str).str.lower() == "yes").astype(int) * df["bmi"]
)

df["claim_frequency_group"] = pd.cut(
    df["claim_count"],
    bins=[-1, 0, 2, 100],
    labels=["No claims", "1-2 claims", "3+ claims"]
).astype(str)

# Create cost categories if the dataset does not already contain them.
if "claim_cost_category" not in df.columns:
    df["claim_cost_category"] = pd.qcut(
        df["charges"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    ).astype(str)


# ==================================================
# REGRESSION MODEL
# ==================================================

num_cols = [
    "age",
    "bmi",
    "children",
    "deductible",
    "coverage_limit",
    "claim_count",
    "smoker_bmi_interaction"
]

cat_cols = [
    "sex",
    "smoker",
    "region",
    "policy_type",
    "network_type",
    "age_group",
    "bmi_group",
    "claim_frequency_group"
]

X = df[num_cols + cat_cols]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)


# ==================================================
# BASELINE MODEL
# ==================================================

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
baseline_r2 = r2_score(y_test, baseline_predictions)


# ==================================================
# DASHBOARD TABS
# ==================================================

tabs = st.tabs([
    "Overview",
    "Cost Prediction",
    "Customer Feedback",
    "Pattern Mining",
    "Model Evaluation"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with tabs[0]:
    st.header("Insurance Analytics Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Average Insurance Expense", inr(df["charges"].mean()))

    negative_percentage = (
        (feedback["sentiment"] == "Negative").mean() * 100
    )
    col3.metric("Negative Feedback", f"{negative_percentage:.1f}%")
    col4.metric("Model R² Score", f"{r2:.3f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Insurance Expense Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["charges"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Annual Insurance Expense (₹)")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Expense by Smoking Status")
        fig, ax = plt.subplots()
        sns.barplot(
            data=df,
            x="smoker",
            y="charges",
            errorbar=None,
            ax=ax
        )
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Average Expense (₹)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Insurance Expense by Policy Type")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="policy_type", y="charges", ax=ax)
    ax.set_xlabel("Policy Type")
    ax.set_ylabel("Insurance Expense (₹)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Customer Dataset")
    st.dataframe(df.head(10), use_container_width=True)


# ==================================================
# TAB 2: COST PREDICTION
# ==================================================

with tabs[1]:
    st.header("Insurance Expense Prediction")
    st.write(
        "Enter customer and policy details to predict "
        "the expected annual insurance expense."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age_in = st.number_input("Age", 18, 100, 35)
        sex_in = st.selectbox("Sex", ["female", "male"])
        bmi_in = st.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
        children_in = st.number_input("Dependents", 0, 10, 1)

    with c2:
        smoker_in = st.selectbox("Smoker", ["no", "yes"])
        region_in = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"]
        )
        policy_in = st.selectbox(
            "Policy Type",
            ["Basic", "Standard", "Premium"]
        )
        network_in = st.selectbox("Network", ["HMO", "PPO", "EPO"])

    with c3:
        # These choices are rounded INR amounts.
        # They should be consistent with the units used in your training dataset.
        ded_in = st.selectbox(
            "Deductible (₹)",
            [500, 1000, 5000, 10000, 50000]
        )
        cov_in = st.selectbox(
            "Coverage Limit (₹)",
            [50000, 100000, 200000, 500000, 1000000, 2000000]
        )
        claims_in = st.number_input("Previous Claim Count", 0, 15, 1)

    if st.button("Predict Insurance Expense", type="primary"):
        age_g = pd.cut(
            [age_in],
            [17, 30, 45, 60, 100],
            labels=["18-30", "31-45", "46-60", "60+"]
        ).astype(str)[0]

        bmi_g = pd.cut(
            [bmi_in],
            [0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        ).astype(str)[0]

        cf_g = pd.cut(
            [claims_in],
            [-1, 0, 2, 100],
            labels=["No claims", "1-2 claims", "3+ claims"]
        ).astype(str)[0]

        input_data = pd.DataFrame([{
            "age": age_in,
            "bmi": bmi_in,
            "children": children_in,
            "deductible": ded_in,
            "coverage_limit": cov_in,
            "claim_count": claims_in,
            "smoker_bmi_interaction": (
                (smoker_in == "yes") * bmi_in
            ),
            "sex": sex_in,
            "smoker": smoker_in,
            "region": region_in,
            "policy_type": policy_in,
            "network_type": network_in,
            "age_group": age_g,
            "bmi_group": bmi_g,
            "claim_frequency_group": cf_g
        }])

        estimate = model.predict(input_data)[0]

        st.success(
            f"Estimated Annual Insurance Expense: {inr(estimate)}"
        )
        st.caption(
            "This is a machine learning estimate, not an actual insurance quotation."
        )


# ==================================================
# TAB 3: CUSTOMER FEEDBACK
# ==================================================

with tabs[2]:
    st.header("Customer Feedback Sentiment Analysis")

    st.subheader("Sentiment Distribution")
    counts = feedback["sentiment"].value_counts()

    sentiment_order = ["Positive", "Negative", "Neutral"]
    st.bar_chart(counts.reindex(sentiment_order, fill_value=0))

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", int(counts.get("Positive", 0)))
    col2.metric("Negative", int(counts.get("Negative", 0)))
    col3.metric("Neutral", int(counts.get("Neutral", 0)))

    # Train the text classification model.
    X_text = feedback["feedback_text"]
    y_text = feedback["sentiment"]

    X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
        X_text,
        y_text,
        test_size=0.25,
        random_state=42,
        stratify=y_text
    )

    sentiment_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    sentiment_model.fit(X_text_train, y_text_train)

    test_predictions = sentiment_model.predict(X_text_test)
    sentiment_accuracy = accuracy_score(y_text_test, test_predictions)

    st.metric(
        "Sentiment Model Accuracy",
        f"{sentiment_accuracy:.2%}"
    )

    # Show classification report to help evaluate all three classes.
    with st.expander("View sentiment model evaluation"):
        report = classification_report(
            y_text_test,
            test_predictions,
            output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose())

        cm = confusion_matrix(
            y_text_test,
            test_predictions,
            labels=sentiment_model.classes_
        )
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=sentiment_model.classes_,
            yticklabels=sentiment_model.classes_,
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    # Test user-entered feedback.
    st.subheader("Analyse New Customer Feedback")

    user_text = st.text_area(
        "Enter customer feedback",
        "Customer support was good"
    )

    if st.button("Analyse Feedback"):
        if not user_text.strip():
            st.warning("Please enter some feedback.")
        else:
            ml_result = sentiment_model.predict([user_text])[0]
            probabilities = sentiment_model.predict_proba([user_text])[0]

            # Use clear phrase rules for common expressions.
            # Otherwise, use the trained ML model.
            rule_result = get_sentiment_rule(user_text)

            if rule_result is not None:
                final_result = rule_result
                method = "Phrase-based correction for a clear expression"
            else:
                final_result = ml_result
                method = "TF-IDF and Logistic Regression model"

            if final_result == "Positive":
                st.success(f"Predicted Sentiment: {final_result}")
            elif final_result == "Negative":
                st.error(f"Predicted Sentiment: {final_result}")
            else:
                st.info(f"Predicted Sentiment: {final_result}")

            st.caption(f"Classification method: {method}")

            # These probabilities belong to the ML model prediction.
            st.write("Machine-learning model probabilities:")
            probability_df = pd.DataFrame({
                "Sentiment": sentiment_model.classes_,
                "Probability": probabilities
            })
            st.bar_chart(
                probability_df.set_index("Sentiment")
            )

    # Common complaints based on records labelled Negative.
    st.subheader("Common Customer Complaints")

    negative_feedback = feedback.loc[
        feedback["sentiment"] == "Negative",
        "feedback_text"
    ].str.lower()

    complaint_keywords = {
        "Claim Delay": ["delay", "late", "waiting"],
        "Poor Support": ["support", "service", "response"],
        "Claim Rejection": ["reject", "denied"],
        "High Premium": ["premium", "expensive", "cost"],
        "Policy Issues": ["policy", "coverage", "confusing"]
    }

    complaint_counts = {}

    for complaint, keywords in complaint_keywords.items():
        complaint_counts[complaint] = int(
            negative_feedback.apply(
                lambda text: any(word in text for word in keywords)
            ).sum()
        )

    st.bar_chart(pd.Series(complaint_counts))

    st.subheader("Customer Feedback Records")
    st.dataframe(feedback.head(15), use_container_width=True)


# ==================================================
# TAB 4: PATTERN MINING
# ==================================================

with tabs[3]:
    st.header("Policy and Claim Cost Pattern Mining")

    st.write(
        "Apriori discovers frequent relationships between "
        "policy characteristics and insurance cost categories."
    )

    basket = pd.DataFrame({
        "Policy Basic": df["policy_type"] == "Basic",
        "Policy Standard": df["policy_type"] == "Standard",
        "Policy Premium": df["policy_type"] == "Premium",
        "Smoker": df["smoker"].astype(str).str.lower() == "yes",
        "Non-smoker": df["smoker"].astype(str).str.lower() == "no",
        "Age 18-30": df["age_group"] == "18-30",
        "Age 31-45": df["age_group"] == "31-45",
        "Age 46+": df["age_group"].isin(["46-60", "60+"]),
        "Low Cost": df["claim_cost_category"] == "Low",
        "Medium Cost": df["claim_cost_category"] == "Medium",
        "High Cost": df["claim_cost_category"] == "High"
    }).astype(bool)

    frequent_items = apriori(
        basket,
        min_support=0.04,
        use_colnames=True
    )

    if not frequent_items.empty:
        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=0.35
        )

        rules = rules[
            rules["consequents"].apply(
                lambda items: any("Cost" in str(item) for item in items)
            )
        ].copy()

        if not rules.empty:
            rules["antecedents"] = rules["antecedents"].apply(
                lambda items: ", ".join(sorted(items))
            )
            rules["consequents"] = rules["consequents"].apply(
                lambda items: ", ".join(sorted(items))
            )

            st.subheader("Discovered Association Rules")
            st.dataframe(
                rules[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift"
                    ]
                ].sort_values(
                    "lift",
                    ascending=False
                ).head(15),
                use_container_width=True
            )
        else:
            st.info("No association rules met the selected thresholds.")
    else:
        st.info("No frequent itemsets were discovered.")


# ==================================================
# TAB 5: MODEL EVALUATION
# ==================================================

with tabs[4]:
    st.header("Regression Model Evaluation")
    st.write(
        "Evaluation metrics calculated using the held-out test dataset."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", inr(mae))
    col2.metric("RMSE", inr(rmse))
    col3.metric("R² Score", f"{r2:.3f}")

    st.subheader("Model Comparison")
    st.write(f"Linear Regression R²: **{baseline_r2:.3f}**")
    st.write(f"Random Forest R²: **{r2:.3f}**")

    st.subheader("Actual vs Predicted Expenses")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.6)
    ax.set_xlabel("Actual Expense (₹)")
    ax.set_ylabel("Predicted Expense (₹)")
    ax.set_title("Actual vs Predicted Insurance Expenses")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Important Cost-Related Features")

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_values = model.named_steps[
        "regressor"
    ].feature_importances_

    importance = pd.Series(
        importance_values,
        index=feature_names
    ).sort_values(ascending=False).head(12)

    st.bar_chart(importance)

    st.caption(
        "Feature importance indicates predictive contribution, not causation."
    )

    st.success("Model evaluation completed successfully.")
