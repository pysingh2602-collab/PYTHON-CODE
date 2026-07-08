import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Random Forest Classifier",
    layout="wide"
)

st.title("🌳 Random Forest Classification")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    # Select Target Column
    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # Features and Target
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical columns into numbers
    X = pd.get_dummies(X)

    # Convert target into numeric values
    y = pd.factorize(y)[0]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Number of Trees
    n_trees = st.slider(
        "Number of Trees",
        min_value=10,
        max_value=500,
        value=100
    )

    if st.button("Train Random Forest"):

        # Create Model
        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42
        )

        # Train Model
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        st.success(
            f"Accuracy : {accuracy:.2%}"
        )

        # -----------------------------
        # Classification Report
        # -----------------------------
        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(report_df)

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        st.pyplot(fig)

        # -----------------------------
        # Feature Importance
        # -----------------------------
        st.subheader(
            "Feature Importance"
        )

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance":
            model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(importance)

        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        ax2.bar(
            importance["Feature"],
            importance["Importance"]
        )

        plt.xticks(
            rotation=90
        )

        ax2.set_title(
            "Feature Importance"
        )

        st.pyplot(fig2)