import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, ttest_1samp, chi2_contingency

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Statistical Tests", layout="wide")

st.title("📊 Statistical Tests using Streamlit")
st.write("Perform Z-Test, T-Test and Chi-Square Test with Graphs.")

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns

    # ====================================
    # Z TEST
    # ====================================
    st.header("1️⃣ Z-Test")

    if len(numeric_cols) > 0:

        z_col = st.selectbox(
            "Select column for Z-Test",
            numeric_cols,
            key="z"
        )

        population_mean = st.number_input(
            "Population Mean",
            value=float(df[z_col].mean())
        )

        if st.button("Perform Z-Test"):

            sample = df[z_col].dropna()

            sample_mean = sample.mean()
            sample_std = sample.std()

            n = len(sample)

            z_score = (
                (sample_mean - population_mean)
                / (sample_std / np.sqrt(n))
            )

            p_value = 2 * (1 - norm.cdf(abs(z_score)))

            st.success(f"Z-Statistic = {z_score:.4f}")
            st.success(f"P-Value = {p_value:.4f}")

            # Graph
            fig, ax = plt.subplots(figsize=(8, 4))

            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x)

            ax.plot(x, y)
            ax.axvline(z_score,
                       color='red',
                       linestyle='--',
                       label=f'Z = {z_score:.2f}')

            ax.set_title("Normal Distribution Curve")
            ax.legend()

            st.pyplot(fig)

    # ====================================
    # T TEST
    # ====================================
    st.header("2️⃣ T-Test")

    if len(numeric_cols) > 0:

        t_col = st.selectbox(
            "Select column for T-Test",
            numeric_cols,
            key="t"
        )

        hypothesized_mean = st.number_input(
            "Hypothesized Mean",
            value=float(df[t_col].mean()),
            key="tmean"
        )

        if st.button("Perform T-Test"):

            sample = df[t_col].dropna()

            t_stat, p_val = ttest_1samp(
                sample,
                hypothesized_mean
            )

            st.success(f"T-Statistic = {t_stat:.4f}")
            st.success(f"P-Value = {p_val:.4f}")

            # Graph
            fig, ax = plt.subplots(figsize=(8, 4))

            sns.histplot(
                sample,
                kde=True,
                ax=ax
            )

            ax.axvline(
                hypothesized_mean,
                color='red',
                linestyle='--',
                label='Hypothesized Mean'
            )

            ax.set_title("Sample Distribution")
            ax.legend()

            st.pyplot(fig)

    # ====================================
    # CHI-SQUARE TEST
    # ====================================
    st.header("3️⃣ Chi-Square Test")

    categorical_cols = df.select_dtypes(
        include=['object']
    ).columns

    if len(categorical_cols) >= 2:

        col1 = st.selectbox(
            "Select First Categorical Column",
            categorical_cols,
            key="c1"
        )

        col2 = st.selectbox(
            "Select Second Categorical Column",
            categorical_cols,
            key="c2"
        )

        if st.button("Perform Chi-Square Test"):

            contingency = pd.crosstab(
                df[col1],
                df[col2]
            )

            chi2, p, dof, expected = (
                chi2_contingency(contingency)
            )

            st.write("Contingency Table")
            st.dataframe(contingency)

            st.success(
                f"Chi-Square Statistic = {chi2:.4f}"
            )

            st.success(
                f"P-Value = {p:.4f}"
            )

            st.success(
                f"Degrees of Freedom = {dof}"
            )

            # Heatmap
            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.heatmap(
                contingency,
                annot=True,
                cmap="Blues",
                fmt='d',
                ax=ax
            )

            ax.set_title(
                "Chi-Square Contingency Heatmap"
            )

            st.pyplot(fig)

    else:
        st.warning(
            "Dataset needs at least two categorical columns for Chi-Square Test."
        )