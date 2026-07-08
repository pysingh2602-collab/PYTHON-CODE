"""
Advanced CSV Data Explorer
===========================
A Streamlit dashboard for exploring, visualizing, and running statistical
hypothesis tests on any CSV file a user uploads.

Tabs:
    1. Portal              - landing page / quick orientation
    2. Data                - dataset preview, types, missing values, describe()
    3. Analysis             - distribution stats + charts for one numeric column
    4. Pandas Dashboard     - common pandas operations, runnable from the UI
    5. Statistical Tests    - Z-test, T-test, Chi-square test, ANOVA

Run with:  streamlit run geeks_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    skew,
    kurtosis,
    ttest_1samp,
    chi2_contingency,
    f_oneway,
)
from statsmodels.stats.weightstats import ztest


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# st.set_page_config() must be the very first Streamlit command that runs.
st.set_page_config(
    page_title="Advanced CSV Data Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Advanced CSV Data Explorer")
st.caption(
    "Upload a CSV file and explore it with summary stats, charts, pandas "
    "commands, and hypothesis tests \u2014 no coding required."
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# Keeping repeated logic in small helper functions makes the tab code below
# easier to read and keeps every chart / message consistent.

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame.

    This is wrapped in @st.cache_data so Streamlit only re-parses the file
    when a *different* file is uploaded, instead of every single time a
    widget elsewhere on the page triggers a script rerun.
    """
    return pd.read_csv(file)


def render_figure(fig) -> None:
    """Show a matplotlib figure in the app, then free its memory.

    Matplotlib keeps every figure object alive until it's explicitly
    closed. In a long-running Streamlit session that creates many charts,
    skipping plt.close() slowly leaks memory, so we always clean up here.
    """
    st.pyplot(fig)
    plt.close(fig)


def show_test_verdict(p_value: float, alpha: float, hypothesis_text: str) -> None:
    """Translate a p-value into a plain-English conclusion.

    p_value         - the p-value returned by a statistical test
    alpha           - the significance threshold the user chose
    hypothesis_text - a short phrase describing what's being tested, used
                       to phrase the conclusion (e.g. "the means differ")
    """
    if p_value < alpha:
        st.warning(
            f"**p = {p_value:.4f} < \u03b1 = {alpha}** \u2192 Reject the null hypothesis. "
            f"There is statistically significant evidence that {hypothesis_text}."
        )
    else:
        st.info(
            f"**p = {p_value:.4f} \u2265 \u03b1 = {alpha}** \u2192 Fail to reject the null hypothesis. "
            f"There isn't enough evidence that {hypothesis_text}."
        )


# =============================================================================
# SIDEBAR : FILE UPLOAD + QUICK ORIENTATION
# =============================================================================
with st.sidebar:
    st.header("1\ufe0f\u20e3 Upload your data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Any standard CSV works. The first row should contain column headers.",
    )

    st.divider()
    st.header("\u2139\ufe0f What each tab does")
    st.markdown(
        "- **Data** \u2014 preview rows, column types, missing values\n"
        "- **Analysis** \u2014 distribution stats & charts for one column\n"
        "- **Pandas Dashboard** \u2014 common pandas commands, click to run\n"
        "- **Statistical Tests** \u2014 Z-test, T-test, Chi-square, ANOVA"
    )


# =============================================================================
# MAIN APP LOGIC
# =============================================================================
# Everything that depends on the uploaded file lives inside this `if` block.
# (In the original script, the "Statistical Tests" tab was accidentally
# written *outside* this block, so it ran even before a file was uploaded
# and crashed with a NameError on `df`/`stat_tab`. That's fixed here by
# nesting all five tabs inside this single `if uploaded_file is not None:`.)
if uploaded_file is not None:

    # ---- Load the file, with friendly errors instead of a crash ----------
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read this CSV file: {e}")
        st.stop()

    if df.empty:
        st.warning("The uploaded CSV has no rows. Please upload a file that contains data.")
        st.stop()

    # Column groups are reused across several tabs, so compute them once.
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ==================================================
    # TABS
    # ==================================================
    portal, data_tab, dictionary_tab, analysis, pandas_tab, stat_tab = st.tabs(
        [
            "🏠 Portal",
            "📁 Data",
            "📚 Data Dictionary",
            "📊 Analysis",
            "🐼 Pandas Dashboard",
            "📈 Statistical Tests",
        ]
    )

    # ==================================================
    # TAB 1 : PORTAL
    # ==================================================
    with portal:

        st.header("Welcome 👋")
        st.success(
            f"Loaded **{uploaded_file.name}** \u2014 {df.shape[0]} rows \u00d7 {df.shape[1]} columns."
        )

        st.markdown("Here's what you can do once your data is loaded:")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.markdown(
                "- ✅ Explore your dataset (preview, types, missing values)\n"
                "- ✅ Visualize distributions (histogram, box plot, line chart)\n"
                "- ✅ Inspect correlations between numeric columns\n"
            )
        with feat_col2:
            st.markdown(
                "- ✅ Run common pandas commands without writing code\n"
                "- ✅ Test hypotheses: Z-test, T-test, Chi-square, ANOVA\n"
                "- ✅ Download sorted / grouped / cleaned data\n"
            )

    # ==================================================
    # TAB 2 : DATA
    # ==================================================
    with data_tab:

        st.header("📁 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isnull().sum().sum()))

        st.subheader("Data Types")
        st.caption(
            "A column's type determines what's possible with it later \u2014 "
            "numeric tests need numeric columns, Chi-square needs categorical ones."
        )
        st.dataframe(
            pd.DataFrame({"Column": df.columns, "Datatype": df.dtypes.astype(str)}),
            use_container_width=True,
        )

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        st.dataframe(
            pd.DataFrame(
                {"Column": df.columns, "Missing": missing.values, "Missing %": missing_pct.values}
            ),
            use_container_width=True,
        )
        if missing.sum() == 0:
            st.success("No missing values found \u2014 this dataset is complete! 🎉")
        else:
            st.caption("Columns with a high missing percentage may need cleaning before analysis.")

        st.subheader("Descriptive Statistics")
        if numeric_cols:
            st.caption("count, mean, standard deviation, min/max, and quartiles for every numeric column.")
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns to summarize.")
           # ==================================================
    # TAB 3 : DATA DICTIONARY
    # ==================================================
    with dictionary_tab:

        st.header("📚 Data Dictionary")
        st.caption(
            "A data dictionary provides detailed information about each column "
            "including datatype, missing values, unique values and sample data."
        )

        dictionary_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum().values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": [df[col].nunique() for col in df.columns],
            "Sample Value": [
                str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
                for col in df.columns
            ]
        })

        st.dataframe(
            dictionary_df,
            use_container_width=True
        )

        st.download_button(
            label="⬇️ Download Data Dictionary",
            data=dictionary_df.to_csv(index=False).encode("utf-8"),
            file_name="data_dictionary.csv",
            mime="text/csv"
        )

        selected_column = st.selectbox(
            "Select a column for detailed information",
            df.columns
        )

        st.subheader(f"Details of: {selected_column}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Datatype:**", df[selected_column].dtype)
            st.write("**Missing Values:**", df[selected_column].isnull().sum())
            st.write("**Unique Values:**", df[selected_column].nunique())

        with col2:
            st.write("**Minimum Value:**",
                     df[selected_column].min()
                     if pd.api.types.is_numeric_dtype(df[selected_column])
                     else "Not Numeric")

            st.write("**Maximum Value:**",
                     df[selected_column].max()
                     if pd.api.types.is_numeric_dtype(df[selected_column])
                     else "Not Numeric")

        st.subheader("Sample Records")

        st.dataframe(
            df[[selected_column]].head(10),
            use_container_width=True
        )

    # TAB 3 : ANALYSIS
    # ==================================================
    with analysis:

        st.header("📊 Single-Column Analysis")

        if not numeric_cols:
            st.warning("No numeric columns were found, so there's nothing to analyze in this tab.")
        else:
            selected_col = st.selectbox(
                "Select a numerical column to analyze",
                numeric_cols,
                help="Every statistic and chart below updates for whichever column you choose.",
            )

            data = df[selected_col].dropna()

            if data.empty:
                st.warning(f"'{selected_col}' has no non-missing values to analyze.")
            else:
                st.subheader("Statistical Summary")
                st.caption(
                    "Skewness: 0 means a symmetric distribution; positive/negative means it "
                    "leans right/left. Kurtosis: 0 means tails similar to a normal distribution; "
                    "higher means more extreme outliers."
                )
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Statistic": [
                                "Count", "Mean", "Median", "Minimum", "Maximum",
                                "Variance", "Standard Deviation", "Skewness", "Kurtosis",
                            ],
                            "Value": [
                                data.count(), data.mean(), data.median(), data.min(),
                                data.max(), data.var(), data.std(), skew(data), kurtosis(data),
                            ],
                        }
                    ),
                    use_container_width=True,
                )

                # ---- Histogram ----
                st.subheader("Histogram")
                st.caption("How values are distributed across ranges, with a smoothed density curve (KDE).")
                fig, ax = plt.subplots()
                sns.histplot(data, kde=True, ax=ax)
                ax.set_xlabel(selected_col)
                render_figure(fig)

                # ---- Box Plot ----
                st.subheader("Box Plot")
                st.caption("The median, the middle 50% of values (the box), and outliers (points beyond the whiskers).")
                fig, ax = plt.subplots()
                sns.boxplot(x=data, ax=ax)
                ax.set_xlabel(selected_col)
                render_figure(fig)

                # ---- Line Chart ----
                st.subheader("Line Chart")
                st.caption("Values plotted in row order \u2014 useful for spotting trends if rows represent a sequence (e.g. time).")
                fig, ax = plt.subplots()
                ax.plot(data.values)
                ax.set_xlabel("Row index")
                ax.set_ylabel(selected_col)
                render_figure(fig)

            # ---- Correlation Heatmap ----
            st.subheader("Correlation Heatmap")
            st.caption(
                "How strongly every pair of numeric columns moves together: "
                "+1 = perfectly together, -1 = perfectly opposite, 0 = unrelated."
            )
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                render_figure(fig)
            else:
                st.info("Need at least 2 numeric columns to compute correlations.")

    # ==================================================
    # TAB 4 : PANDAS DASHBOARD
    # ==================================================
    with pandas_tab:

        st.header("🐼 Pandas Commands Dashboard")
        st.caption("Each panel shows the pandas code being run, plus its live output \u2014 a quick way to learn the syntax.")

        with st.expander("📄 Data Preview \u2014 df.head() / df.tail() / df.shape"):
            st.code("df.head()")
            st.dataframe(df.head(), use_container_width=True)
            st.code("df.tail()")
            st.dataframe(df.tail(), use_container_width=True)
            st.code("df.shape")
            st.write(df.shape)

        with st.expander("ℹ️ Data Information \u2014 df.columns / df.dtypes"):
            st.code("df.columns")
            st.write(list(df.columns))
            st.code("df.dtypes")
            st.dataframe(df.dtypes.astype(str), use_container_width=True)

        with st.expander("📊 Statistics \u2014 df.describe() / df.mean()"):
            st.code("df.describe()")
            st.dataframe(df.describe(), use_container_width=True)
            if numeric_cols:
                st.code("df.mean(numeric_only=True)")
                st.dataframe(df.mean(numeric_only=True), use_container_width=True)
            else:
                st.info("No numeric columns to average.")

        with st.expander("❓ Missing Values \u2014 df.isnull().sum()"):
            st.code("df.isnull().sum()")
            st.dataframe(df.isnull().sum(), use_container_width=True)

        with st.expander("🔄 Sorting \u2014 df.sort_values()"):
            sort_col = st.selectbox("Column to sort by", df.columns, key="sort_col")
            ascending = st.checkbox("Ascending order", value=True, key="sort_asc")
            st.code(f'df.sort_values("{sort_col}", ascending={ascending})')

            try:
                sorted_df = df.sort_values(sort_col, ascending=ascending)
                st.dataframe(sorted_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download sorted data as CSV",
                    sorted_df.to_csv(index=False).encode("utf-8"),
                    file_name="sorted_data.csv",
                    mime="text/csv",
                    key="download_sorted",
                )
            except Exception as e:
                st.warning(f"Couldn't sort by this column: {e}")

        with st.expander("📂 GroupBy \u2014 df.groupby().mean()"):
            if len(cat_cols) == 0 or len(numeric_cols) == 0:
                st.info(
                    "GroupBy needs at least one categorical (text) column and one numeric "
                    "column \u2014 this dataset doesn't have both."
                )
            else:
                group_col = st.selectbox("Group by (categorical)", cat_cols, key="group_col")
                value_col = st.selectbox("Aggregate (numeric)", numeric_cols, key="value_col")

                result = df.groupby(group_col)[value_col].mean().reset_index()
                st.code(f'df.groupby("{group_col}")["{value_col}"].mean()')
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "⬇️ Download grouped result as CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="grouped_data.csv",
                    mime="text/csv",
                    key="download_grouped",
                )

        with st.expander("🔥 Correlation \u2014 df.corr()"):
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                st.dataframe(corr, use_container_width=True)
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                render_figure(fig)
            else:
                st.info("Need at least 2 numeric columns to compute correlations.")

        with st.expander("🎯 loc[] and iloc[] \u2014 row selection"):
            st.caption("Pick any row position to inspect with .iloc[].")
            row_idx = st.number_input(
                "Row position",
                min_value=0,
                max_value=max(len(df) - 1, 0),
                value=0,
                step=1,
                key="loc_idx",
            )
            st.code(f"df.iloc[[{row_idx}]]")
            st.dataframe(df.iloc[[row_idx]], use_container_width=True)

        with st.expander("💾 Export \u2014 download your data"):
            st.caption("These buttons export the current dataframe as-is.")
            st.download_button(
                "⬇️ Download as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="data.csv",
                mime="text/csv",
                key="download_csv_full",
            )
            st.download_button(
                "⬇️ Download as JSON",
                df.to_json(orient="records").encode("utf-8"),
                file_name="data.json",
                mime="application/json",
                key="download_json_full",
            )

    # ==================================================
    # TAB 5 : STATISTICAL TESTS
    # ==================================================
    # NOTE: this whole tab now lives *inside* `if uploaded_file is not None:`,
    # properly indented as part of the tabs created above. That's the fix
    # for the original bug: previously this block sat at the top level of
    # the script (outside the if/else entirely), so Python tried to run it
    # unconditionally and crashed with a NameError before any file was
    # uploaded, since `df` and `stat_tab` didn't exist yet.
    with stat_tab:

        st.header("📈 Statistical Tests Dashboard")
        st.caption(
            "Pick a test, choose your columns, and the app computes the statistic, "
            "the p-value, and a plain-English verdict."
        )

        alpha = st.slider(
            "Significance level (\u03b1)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="The threshold below which a p-value counts as 'statistically significant'. 0.05 is the conventional default.",
        )

        st.divider()

        # -----------------------------------------------------------
        # Z-TEST \u2014 compares a sample mean to a known population mean.
        # Best suited to larger samples (commonly n > 30).
        # -----------------------------------------------------------
        st.subheader("1\ufe0f\u20e3 Z-Test \u2014 sample mean vs. a population mean")
        st.caption("Null hypothesis: the column's true mean equals the population mean you enter.")

        if not numeric_cols:
            st.info("No numeric columns available for a Z-test.")
        else:
            z_col = st.selectbox("Numeric column", numeric_cols, key="z_col")
            z_data = df[z_col].dropna()

            pop_mean = st.number_input(
                "Population mean to test against", value=float(z_data.mean()), key="z_mean"
            )

            if st.button("Run Z-Test"):
                try:
                    z_stat, p = ztest(z_data, value=pop_mean)

                    res_c1, res_c2 = st.columns(2)
                    res_c1.success(f"Z statistic = {z_stat:.4f}")
                    res_c2.success(f"P value = {p:.4f}")
                    show_test_verdict(p, alpha, f"the mean of '{z_col}' differs from {pop_mean}")

                    fig, ax = plt.subplots()
                    sns.histplot(z_data, kde=True, ax=ax)
                    ax.axvline(pop_mean, color="red", linestyle="--", label="Population mean")
                    ax.legend()
                    render_figure(fig)
                except Exception as e:
                    st.error(f"Z-test failed: {e}")

        st.divider()

        # -----------------------------------------------------------
        # T-TEST \u2014 like the Z-test, but appropriate for smaller samples
        # where the population standard deviation isn't known.
        # -----------------------------------------------------------
        st.subheader("2\ufe0f\u20e3 T-Test \u2014 sample mean vs. a hypothesized mean")
        st.caption("Null hypothesis: the column's true mean equals the value you enter. Works well even with small samples.")

        if not numeric_cols:
            st.info("No numeric columns available for a T-test.")
        else:
            t_col = st.selectbox("Numeric column", numeric_cols, key="t_col")
            t_data = df[t_col].dropna()

            test_mean = st.number_input(
                "Hypothesized mean", value=float(t_data.mean()), key="t_mean"
            )

            if st.button("Run T-Test"):
                try:
                    t_stat, p = ttest_1samp(t_data, popmean=test_mean)

                    res_c1, res_c2 = st.columns(2)
                    res_c1.success(f"T statistic = {t_stat:.4f}")
                    res_c2.success(f"P value = {p:.4f}")
                    show_test_verdict(p, alpha, f"the mean of '{t_col}' differs from {test_mean}")

                    fig, ax = plt.subplots()
                    sns.histplot(t_data, kde=True, ax=ax)
                    ax.axvline(test_mean, color="red", linestyle="--", label="Hypothesized mean")
                    ax.legend()
                    render_figure(fig)
                except Exception as e:
                    st.error(f"T-test failed: {e}")

        st.divider()

        # -----------------------------------------------------------
        # CHI-SQUARE TEST \u2014 tests whether two categorical columns are
        # independent of one another.
        # -----------------------------------------------------------
        st.subheader("3\ufe0f\u20e3 Chi-Square Test \u2014 independence of two categories")
        st.caption("Null hypothesis: the two categorical columns are independent (knowing one tells you nothing about the other).")

        if len(cat_cols) < 2:
            st.info("Need at least 2 categorical (text) columns for a Chi-square test.")
        else:
            chi_col1 = st.selectbox("First category", cat_cols, key="chi_col1")
            remaining_cats = [c for c in cat_cols if c != chi_col1]
            chi_col2 = st.selectbox("Second category", remaining_cats, key="chi_col2")

            if st.button("Run Chi-Square Test"):
                try:
                    table = pd.crosstab(df[chi_col1], df[chi_col2])
                    chi2_stat, p, dof, expected = chi2_contingency(table)

                    res_c1, res_c2 = st.columns(2)
                    res_c1.success(f"Chi-Square = {chi2_stat:.4f}")
                    res_c2.success(f"P value = {p:.4f}")
                    show_test_verdict(p, alpha, f"'{chi_col1}' and '{chi_col2}' are related")

                    if (expected < 5).any():
                        st.warning(
                            "Some expected cell counts are below 5 \u2014 the Chi-square "
                            "approximation may be unreliable for this combination of columns."
                        )

                    st.caption("Observed counts (rows = first category, columns = second category):")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(table, annot=True, cmap="Blues", fmt="d", ax=ax)
                    render_figure(fig)
                except Exception as e:
                    st.error(f"Chi-square test failed: {e}")

        st.divider()

        # -----------------------------------------------------------
        # ANOVA \u2014 tests whether the mean of a numeric column differs
        # across the groups defined by a categorical column.
        # -----------------------------------------------------------
        st.subheader("4\ufe0f\u20e3 ANOVA \u2014 compare means across groups")
        st.caption("Null hypothesis: every group shares the same mean for the chosen numeric column.")

        if not cat_cols or not numeric_cols:
            st.info("ANOVA needs at least one categorical column and one numeric column.")
        else:
            anova_group_col = st.selectbox("Group by (categorical)", cat_cols, key="anova_group")
            anova_value_col = st.selectbox("Numeric column to compare", numeric_cols, key="anova_value")

            if st.button("Run ANOVA"):
                groups = [
                    df[df[anova_group_col] == g][anova_value_col].dropna()
                    for g in df[anova_group_col].dropna().unique()
                ]
                groups = [g for g in groups if len(g) > 0]

                if len(groups) < 2:
                    st.warning(f"'{anova_group_col}' needs at least 2 non-empty groups to run ANOVA.")
                else:
                    try:
                        f_stat, p = f_oneway(*groups)

                        res_c1, res_c2 = st.columns(2)
                        res_c1.success(f"F statistic = {f_stat:.4f}")
                        res_c2.success(f"P value = {p:.4f}")
                        show_test_verdict(
                            p, alpha,
                            f"'{anova_value_col}' differs across groups of '{anova_group_col}'",
                        )

                        fig, ax = plt.subplots()
                        sns.boxplot(x=anova_group_col, y=anova_value_col, data=df, ax=ax)
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                        render_figure(fig)
                    except Exception as e:
                        st.error(f"ANOVA failed: {e}")

        st.divider()

        # -----------------------------------------------------------
        # CORRELATION \u2014 a quick numeric reference alongside the tests.
        # -----------------------------------------------------------
        st.subheader("5\ufe0f\u20e3 Correlation Overview")
        st.caption("How strongly each pair of numeric columns moves together.")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            st.dataframe(corr, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            render_figure(fig)
        else:
            st.info("Need at least 2 numeric columns to compute correlations.")

else:
    st.info("👆 Upload a CSV file in the sidebar to get started.")