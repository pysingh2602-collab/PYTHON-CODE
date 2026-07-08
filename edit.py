import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    "Upload a CSV file to explore its data types, "
    "statistics of a particular column, and graphs."
)

# ------------------------------------
# Upload CSV
# ------------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV File
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
    # Data Types
    # ------------------------------------
    st.header("3️⃣ Data Types")

    st.write("""
    **Meaning of Data Types**
    - int64  → Whole Numbers
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
    # Select One Numerical Column
    # ------------------------------------
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select a Numerical Column",
            numeric_cols
        )

        # ------------------------------------
        # Statistical Values
        # ------------------------------------
        st.header("4️⃣ Statistical Values")

        stats = pd.DataFrame({
            "Statistic": [
                "Count",
                "Mean",
                "Median",
                "Mode",
                "Minimum",
                "Maximum",
                "Variance",
                "Standard Deviation"
            ],
            "Value": [
                df[selected_col].count(),
                df[selected_col].mean(),
                df[selected_col].median(),
                df[selected_col].mode().iloc[0]
                if not df[selected_col].mode().empty else "No Mode",
                df[selected_col].min(),
                df[selected_col].max(),
                df[selected_col].var(),
                df[selected_col].std()
            ]
        })

        st.dataframe(stats)
        # ------------------------------------
# Correlation Coefficient
# ------------------------------------
st.header("5️⃣ Correlation Coefficient")

if len(numeric_cols) >= 2:

    col1 = st.selectbox(
        "Select First Column",
        numeric_cols,
        key="corr1"
    )

    col2 = st.selectbox(
        "Select Second Column",
        numeric_cols,
        index=1,
        key="corr2"
    )

    correlation = df[col1].corr(df[col2])

    st.write(
        f"Correlation between **{col1}** and **{col2}** = "
        f"**{correlation:.3f}**"
    )

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[col1], df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"{col1} vs {col2}")
    st.pyplot(fig)

else:
    st.warning(
        "At least two numerical columns are required to calculate correlation."
    )
    # ------------------------------------
    # Graphical Representation
    # ------------------------------------
    st.header("5️⃣ Graphical Representation")

    # Histogram
    st.subheader("Histogram")

    fig, ax = plt.subplots()
    ax.hist(df[selected_col], bins=10)
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {selected_col}")
    st.pyplot(fig)


            # Box Plot
    st.subheader("Box Plot")

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df[selected_col], ax=ax)
    st.pyplot(fig)

        # Bar Chart
        st.subheader("Bar Chart")

        fig, ax = plt.subplots()
        df[selected_col].value_counts().plot(
            kind='bar',
            ax=ax
        )
        ax.set_title(f"Bar Chart of {selected_col}")
        st.pyplot(fig)

    else:
        st.warning(
            "No numerical columns found in the dataset."
        )