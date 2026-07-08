import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, ttest_1samp
# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="CSV Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Data Explorer")
st.write(
    "Upload a CSV file to explore data types, statistics, and graphical representations."
)

# ------------------------------------
# Upload CSV File
# ------------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # ------------------------------------
    # Dataset Preview
    # ------------------------------------
    st.header("1️⃣ Dataset Preview")
    st.dataframe(df)

    # ------------------------------------
    # Dataset Shape
    # ------------------------------------
    st.header("2️⃣ Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # ------------------------------------
    # Missing Values
    # ------------------------------------
    st.header("3️⃣ Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_df)

    # ------------------------------------
    # Duplicate Rows
    # ------------------------------------
    st.header("4️⃣ Duplicate Rows")
    duplicates = df.duplicated().sum()
    st.write(f"Number of Duplicate Rows: {duplicates}")

    # ------------------------------------
    # Data Types
    # ------------------------------------
    st.header("5️⃣ Data Types")

    st.write("""
    **Meaning of Data Types**
    - int64 → Whole Numbers
    - float64 → Decimal Numbers
    - object → Text/String
    - bool → True or False
    """)

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(data_types)

    # ------------------------------------
    # Numerical Columns
    # ------------------------------------
    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select a Numerical Column",
            numeric_cols
        )

        # ------------------------------------
        # Statistical Values
        # ------------------------------------
        st.header("6️⃣ Statistical Values")

        skew_val = df[selected_col].skew()
        kurt_val = df[selected_col].kurt()

        q1 = df[selected_col].quantile(0.25)
        q2 = df[selected_col].quantile(0.50)
        q3 = df[selected_col].quantile(0.75)
        iqr = q3 - q1

        stats = pd.DataFrame({
            "Statistic": [
                "Count",
                "Mean",
                "Median",
                "Mode",
                "Minimum",
                "Maximum",
                "Variance",
                "Standard Deviation",
                "Skewness",
                "Kurtosis",
                "Q1 (25%)",
                "Q2 (50%)",
                "Q3 (75%)",
                "IQR"
            ],
            "Value": [
                df[selected_col].count(),
                df[selected_col].mean(),
                df[selected_col].median(),
                df[selected_col].mode().iloc[0]
                if not df[selected_col].mode().empty
                else "No Mode",
                df[selected_col].min(),
                df[selected_col].max(),
                df[selected_col].var(),
                df[selected_col].std(),
                skew_val,
                kurt_val,
                q1,
                q2,
                q3,
                iqr
            ]
        })

        st.dataframe(stats)

        # Download Statistics
        csv = stats.to_csv(index=False)

        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv,
            file_name="statistics.csv",
            mime="text/csv"
        )

        # ------------------------------------
        # Graphical Representation
        # ------------------------------------
        st.header("7️⃣ Graphical Representation")

        # Histogram
        st.subheader("Histogram")
        fig, ax = plt.subplots()
        ax.hist(df[selected_col], bins=10)
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Histogram of {selected_col}"
        )
        st.pyplot(fig)

        # Distribution Plot
        st.subheader("Distribution Plot")
        fig, ax = plt.subplots()
        sns.histplot(
            df[selected_col],
            kde=True,
            ax=ax
        )
        ax.set_title(
            f"Distribution of {selected_col}"
        )
        st.pyplot(fig)

        # Box Plot
        st.subheader("Box Plot")
        fig, ax = plt.subplots(
            figsize=(7, 3)
        )
        sns.boxplot(
            x=df[selected_col],
            ax=ax
        )
        st.pyplot(fig)

        # Bar Chart
        st.subheader("Bar Chart")
        fig, ax = plt.subplots()
        df[selected_col].value_counts().plot(
            kind="bar",
            ax=ax
        )
        ax.set_title(
            f"Bar Chart of {selected_col}"
        )
        st.pyplot(fig)

        # ------------------------------------
        # Correlation Matrix
        # ------------------------------------
        st.header("8️⃣ Correlation Matrix")

        corr = df.corr(
            numeric_only=True
        )

        st.dataframe(corr)

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

        # ------------------------------------
        # Interpretation
        # ------------------------------------
        st.header("9️⃣ Interpretation")

        if skew_val > 0:
            st.success(
                "The data is Positively Skewed (Right Skewed)."
            )
        elif skew_val < 0:
            st.success(
                "The data is Negatively Skewed (Left Skewed)."
            )
        else:
            st.success(
                "The data is Symmetric."
            )

        if kurt_val > 0:
            st.info(
                "The distribution is Leptokurtic (Sharp Peak)."
            )
        elif kurt_val < 0:
            st.info(
                "The distribution is Platykurtic (Flat)."
            )
        else:
            st.info(
                "The distribution is Mesokurtic (Normal)."
            )

    else:
        st.warning(
            "No numerical columns found in the dataset."
        )

else:
    st.info(
        "Please upload a CSV file to begin."
    ) 

    # ------------------------------------
# Hypothesis Testing
# ------------------------------------
st.header("🧪 Hypothesis Testing")

sample_mean = df[selected_col].mean()
sample_std = df[selected_col].std()
sample_size = len(df[selected_col].dropna())

# User input
hyp_mean = st.number_input(
    "Enter the Hypothesized Mean (μ₀)",
    value=float(sample_mean)
)

# ------------------------------
# Z-Test
# ------------------------------
st.subheader("Z-Test")

if sample_size >= 30:

    z_score = (
        (sample_mean - hyp_mean)
        / (sample_std / np.sqrt(sample_size))
    )

    p_value_z = 2 * (
        1 - norm.cdf(abs(z_score))
    )

    st.write(f"Sample Mean = {sample_mean:.2f}")
    st.write(f"Z-Statistic = {z_score:.4f}")
    st.write(f"P-Value = {p_value_z:.4f}")

    if p_value_z < 0.05:
        st.success(
            "Reject the Null Hypothesis (H₀)"
        )
    else:
        st.info(
            "Fail to Reject the Null Hypothesis (H₀)"
        )

else:
    st.warning(
        "Sample size is less than 30. Z-Test is not recommended."
    )

# ------------------------------
# One Sample T-Test
# ------------------------------
st.subheader("One Sample T-Test")

t_stat, p_value_t = ttest_1samp(
    df[selected_col].dropna(),
    popmean=hyp_mean
)

st.write(f"T-Statistic = {t_stat:.4f}")
st.write(f"P-Value = {p_value_t:.4f}")

if p_value_t < 0.05:
    st.success(
        "Reject the Null Hypothesis (H₀)"
    )
else:
    st.info(
        "Fail to Reject the Null Hypothesis (H₀)"
    )