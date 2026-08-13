import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="E-Commerce Project",
    page_icon="🛒",
    layout="wide"
)


# =========================
# CLEANING FUNCTION
# =========================
def clean_data(df):

    raw_df = df.copy()

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Hidden missing values
    missing_values = [
        "", " ", "NA", "N/A", "na", "n/a",
        "Null", "null", "NULL",
        "None", "none", "-"
    ]

    df = df.replace(missing_values, np.nan)

    # Clean text
    text_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in text_cols:
        df[col] = df[col].apply(
            lambda x: x.strip().lower()
            if isinstance(x, str) else x
        )

    # Numeric columns
    numeric_cols = [
        "customer_age",
        "unit_price_egp",
        "quantity",
        "discount_percent",
        "expected_delivery_days",
        "actual_delivery_days",
        "rating",
        "stock_available_before_sale",
        "lead_time_days",
        "competitor_price_egp"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    for col in ["order_date", "return_request_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Returned
    if "returned" in df.columns:
        df["returned"] = df["returned"].map({
            "yes": "Yes",
            "y": "Yes",
            "1": "Yes",
            "true": "Yes",
            "no": "No",
            "n": "No",
            "0": "No",
            "false": "No"
        })

    # Gender
    if "customer_gender" in df.columns:
        df["customer_gender"] = df["customer_gender"].map({
            "m": "Male",
            "male": "Male",
            "man": "Male",
            "f": "Female",
            "female": "Female",
            "woman": "Female"
        })

    # Standard categorical columns
    categorical_cols = [
        "branch", "sales_channel", "product_domain",
        "category", "brand", "payment_method",
        "marketing_source", "campaign_name",
        "courier", "delivery_status",
        "return_reason_category", "refund_method",
        "supplier", "weather", "season"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.title()

    # Fill missing categories
    for col in ["customer_area", "branch", "sales_channel"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "campaign_name" in df.columns:
        df["campaign_name"] = df["campaign_name"].fillna("No Campaign")

    # Duplicates
    duplicates_before = int(df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)

    duplicates_after = int(df.duplicated().sum())

    # Duplicate Order IDs
    if "order_id" in df.columns:
        duplicate_orders = df[
            df.duplicated("order_id", keep=False)
        ]
    else:
        duplicate_orders = pd.DataFrame()

    # Business rules
    if "customer_age" in df.columns:
        df.loc[
            (df["customer_age"] < 10) |
            (df["customer_age"] > 100),
            "customer_age"
        ] = np.nan

    for col in ["unit_price_egp", "competitor_price_egp"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    if "quantity" in df.columns:
        df.loc[df["quantity"] <= 0, "quantity"] = np.nan

    for col in [
        "expected_delivery_days",
        "actual_delivery_days",
        "stock_available_before_sale",
        "lead_time_days"
    ]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    if "rating" in df.columns:
        df.loc[
            (df["rating"] < 1) |
            (df["rating"] > 5),
            "rating"
        ] = np.nan

    # Return date validation
    if {"order_date", "return_request_date"}.issubset(df.columns):
        df.loc[
            df["return_request_date"] < df["order_date"],
            "return_request_date"
        ] = pd.NaT

    # Discount
    if "discount_percent" in df.columns:

        max_discount = df["discount_percent"].max(skipna=True)

        if pd.notna(max_discount) and max_discount > 1:
            df["discount_percent"] /= 100

        df.loc[
            (df["discount_percent"] < 0) |
            (df["discount_percent"] > 1),
            "discount_percent"
        ] = np.nan


    # =========================
    # FEATURE ENGINEERING
    # =========================

    if {"unit_price_egp", "quantity"}.issubset(df.columns):
        df["gross_revenue_egp"] = (
            df["unit_price_egp"] * df["quantity"]
        )

    if {"gross_revenue_egp", "discount_percent"}.issubset(df.columns):
        df["discount_amount_egp"] = (
            df["gross_revenue_egp"] *
            df["discount_percent"]
        )

    if {"gross_revenue_egp", "discount_amount_egp"}.issubset(df.columns):
        df["net_revenue_egp"] = (
            df["gross_revenue_egp"] -
            df["discount_amount_egp"]
        )

    if {
        "actual_delivery_days",
        "expected_delivery_days"
    }.issubset(df.columns):

        df["delivery_delay_days"] = (
            df["actual_delivery_days"] -
            df["expected_delivery_days"]
        )

        df["on_time_flag"] = (
            df["delivery_delay_days"] <= 0
        )

    if {
        "unit_price_egp",
        "competitor_price_egp"
    }.issubset(df.columns):

        df["price_gap_egp"] = (
            df["unit_price_egp"] -
            df["competitor_price_egp"]
        )

        df["price_gap_percent"] = np.where(
            df["competitor_price_egp"].notna()
            & (df["competitor_price_egp"] != 0),

            (
                df["price_gap_egp"] /
                df["competitor_price_egp"]
            ) * 100,

            np.nan
        )

    if "customer_age" in df.columns:

        df["customer_age_group"] = pd.cut(
            df["customer_age"],
            bins=[0, 18, 25, 35, 45, 55, 65, 100],
            labels=[
                "Under 18",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65+"
            ],
            right=False
        )

    if "order_date" in df.columns:

        df["order_month"] = df["order_date"].dt.month
        df["order_quarter"] = df["order_date"].dt.quarter
        df["order_weekday"] = df["order_date"].dt.day_name()

    if "order_time" in df.columns:

        temp_time = pd.to_datetime(
            df["order_time"].astype("string"),
            errors="coerce"
        )

        df["order_hour"] = temp_time.dt.hour

    if {"order_date", "return_request_date"}.issubset(df.columns):

        df["return_request_lag_days"] = (
            df["return_request_date"] -
            df["order_date"]
        ).dt.days

    if "returned" in df.columns:

        df["returned_flag"] = df["returned"].map({
            "Yes": 1,
            "No": 0
        })


    # Missing summary
    missing_summary = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": df.isna().mean() * 100
    }).sort_values("Missing %", ascending=False)

    report = {
        "Original Rows": len(raw_df),
        "Cleaned Rows": len(df),
        "Rows Removed": len(raw_df) - len(df),
        "Duplicates Before": duplicates_before,
        "Duplicates After": duplicates_after
    }

    return df, raw_df, missing_summary, duplicate_orders, report


# =========================
# HOME
# =========================
def home():

    st.title("🛒 E-Commerce Data Analysis")

    st.write(
        """
        End-to-End Data Science Project covering
        customers, sales, products, marketing,
        delivery, returns, inventory and Machine Learning.
        """
    )

    st.subheader("Application Sections")

    st.write("""
    🧹 Data Cleaning & Feature Engineering

    📊 EDA

    📈 Visualizations

    🤖 Machine Learning Prediction
    """)


# =========================
# CLEANING PAGE
# =========================
def cleaning_page():

    st.title("🧹 Data Cleaning")

    uploaded_file = st.file_uploader(
        "Upload Sales_Dataset.xlsx",
        type=["xlsx"]
    )

    if uploaded_file is None:
        st.info("Upload the Excel dataset to start.")
        return

    try:
        original_df = pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

        (
            clean_df,
            raw_df,
            missing_summary,
            duplicate_orders,
            report
        ) = clean_data(original_df)

    except Exception as e:
        st.error(f"Error: {e}")
        return

    st.success("Data cleaned successfully ✅")


    # Metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original Rows",
        report["Original Rows"]
    )

    c2.metric(
        "Cleaned Rows",
        report["Cleaned Rows"]
    )

    c3.metric(
        "Rows Removed",
        report["Rows Removed"]
    )

    c4.metric(
        "Final Columns",
        clean_df.shape[1]
    )


    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Raw Data",
        "Cleaned Data",
        "Missing Values",
        "Duplicates"
    ])


    with tab1:

        st.subheader("Raw Dataset")

        st.dataframe(
            raw_df,
            use_container_width=True
        )


    with tab2:

        st.subheader("Cleaned Dataset")

        st.dataframe(
            clean_df,
            use_container_width=True
        )


    with tab3:

        st.subheader("Missing Values")

        st.dataframe(
            missing_summary,
            use_container_width=True
        )


    with tab4:

        st.metric(
            "Duplicates Before",
            report["Duplicates Before"]
        )

        st.metric(
            "Duplicates After",
            report["Duplicates After"]
        )

        if duplicate_orders.empty:

            st.success(
                "No duplicated Order_ID found."
            )

        else:

            st.warning(
                f"{len(duplicate_orders)} duplicated Order_ID rows found."
            )

            st.dataframe(
                duplicate_orders,
                use_container_width=True
            )


    # Created Features
    st.subheader("⚙️ Feature Engineering")

    features = [
        "gross_revenue_egp",
        "discount_amount_egp",
        "net_revenue_egp",
        "delivery_delay_days",
        "on_time_flag",
        "price_gap_egp",
        "price_gap_percent",
        "customer_age_group",
        "order_month",
        "order_quarter",
        "order_weekday",
        "order_hour",
        "return_request_lag_days",
        "returned_flag"
    ]

    created = [
        x for x in features
        if x in clean_df.columns
    ]

    st.write(created)


    # Download
    csv = clean_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download Clean Dataset",
        csv,
        "Project1_Clean_Orders.csv",
        "text/csv"
    )

# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_data(clean_df):
    df = clean_df.copy()

    # Missing values summary
    missing_summary = pd.DataFrame({
        "Dtype": df.dtypes.astype(str),
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2)
    }).sort_values("Missing %", ascending=False)

    # Duplicate checks
    duplicates = int(df.duplicated().sum())

    duplicate_order_ids = 0
    if "order_id" in df.columns:
        duplicate_order_ids = int(
            df.duplicated("order_id", keep=False).sum()
        )

    # Return consistency
    inconsistent_no = 0
    inconsistent_yes = 0

    return_cols = [
        "refund_method",
        "return_reason_category",
        "return_complaint_text",
        "return_request_date"
    ]

    if {"returned", "refund_method"}.issubset(df.columns):
        inconsistent_no = int(
            (
                (df["returned"] == "No")
                & df["refund_method"].notna()
            ).sum()
        )

    if "returned" in df.columns and all(
        col in df.columns for col in return_cols
    ):
        inconsistent_yes = int(
            (
                (df["returned"] == "Yes")
                & df[return_cols].isna().all(axis=1)
            ).sum()
        )

    # Courier check
    missing_courier = 0

    if "courier" in df.columns:
        missing_courier = int(
            df["courier"].isna().sum()
        )

    # =====================================================
    # OUTLIER DETECTION USING IQR
    # =====================================================

    outlier_columns = [
        "net_revenue_egp",
        "unit_price_egp",
        "quantity",
        "actual_delivery_days",
        "lead_time_days"
    ]

    outlier_results = []

    for col in outlier_columns:

        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        flag_name = f"{col}_outlier_flag"

        df[flag_name] = (
            (df[col] < lower_bound)
            | (df[col] > upper_bound)
        )

        outlier_results.append({
            "Column": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outliers": int(df[flag_name].sum())
        })

    outlier_summary = pd.DataFrame(outlier_results)

    # =====================================================
    # EXTRA FEATURES FROM PREPROCESSING NOTEBOOK
    # =====================================================

    if "price_gap_percent" in df.columns:

        df["price_position"] = pd.cut(
            df["price_gap_percent"],
            bins=[
                -float("inf"),
                -2,
                2,
                float("inf")
            ],
            labels=[
                "Below Competitor",
                "Similar to Competitor",
                "Above Competitor"
            ]
        )

    if {
        "stock_available_before_sale",
        "quantity"
    }.issubset(df.columns):

        df["low_stock_flag"] = (
            df["stock_available_before_sale"]
            < df["quantity"]
        )

    report = {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Duplicates": duplicates,
        "Duplicate Order IDs": duplicate_order_ids,
        "Inconsistent No": inconsistent_no,
        "Inconsistent Yes": inconsistent_yes,
        "Missing Courier": missing_courier
    }

    return (
        df,
        missing_summary,
        outlier_summary,
        report
    )


# =========================================================
# PREPROCESSING PAGE
# =========================================================

def preprocessing_page():

    st.title("⚙️ Data Preprocessing")

    st.write(
        "This section performs missing-value auditing, "
        "consistency checks, IQR outlier detection "
        "and additional feature engineering."
    )

    # Use cleaned data automatically
    if "clean_df" in st.session_state:

        clean_df = st.session_state["clean_df"].copy()

        st.success(
            "Using cleaned data from Data Cleaning ✅"
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload Project1_Clean_Orders.csv",
            type=["csv"],
            key="preprocessing_file"
        )

        if uploaded_file is None:

            st.info(
                "Please run Data Cleaning first "
                "or upload the cleaned CSV file."
            )

            return

        clean_df = pd.read_csv(uploaded_file)

    try:

        (
            preprocessed_df,
            missing_summary,
            outlier_summary,
            report
        ) = preprocess_data(clean_df)

        st.session_state[
            "preprocessed_df"
        ] = preprocessed_df

    except Exception as e:

        st.error(
            f"Preprocessing Error: {e}"
        )

        return

    st.success(
        "Preprocessing completed successfully ✅"
    )

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        report["Rows"]
    )

    col2.metric(
        "Columns",
        report["Columns"]
    )

    col3.metric(
        "Duplicates",
        report["Duplicates"]
    )

    col4.metric(
        "Duplicate Order IDs",
        report["Duplicate Order IDs"]
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Missing Values",
        "Consistency Checks",
        "Outliers",
        "Final Data"
    ])

    # -----------------------------
    # Missing Values
    # -----------------------------

    with tab1:

        st.subheader(
            "Missing Value Audit"
        )

        st.dataframe(
            missing_summary,
            use_container_width=True
        )

        st.info(
            "Return-related missing values are kept "
            "when the order was not returned. "
            "Courier can also be missing for "
            "orders picked up from the store."
        )

    # -----------------------------
    # Consistency
    # -----------------------------

    with tab2:

        st.subheader(
            "Consistency Checks"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Returned No + Refund",
            report["Inconsistent No"]
        )

        c2.metric(
            "Returned Yes + Missing Return Data",
            report["Inconsistent Yes"]
        )

        c3.metric(
            "Missing Courier",
            report["Missing Courier"]
        )

    # -----------------------------
    # Outliers
    # -----------------------------

    with tab3:

        st.subheader(
            "IQR Outlier Detection"
        )

        st.write(
            "Outliers are flagged, not removed."
        )

        st.dataframe(
            outlier_summary,
            use_container_width=True,
            hide_index=True
        )

        if not outlier_summary.empty:

            chart_data = (
                outlier_summary
                .set_index("Column")[["Outliers"]]
            )

            st.bar_chart(
                chart_data
            )

        st.subheader(
            "Created Outlier Flags"
        )

        flag_columns = [
            "net_revenue_egp_outlier_flag",
            "unit_price_egp_outlier_flag",
            "quantity_outlier_flag",
            "actual_delivery_days_outlier_flag",
            "lead_time_days_outlier_flag"
        ]

        existing_flags = [
            col for col in flag_columns
            if col in preprocessed_df.columns
        ]

        st.write(existing_flags)

    # -----------------------------
    # Final Data
    # -----------------------------

    with tab4:

        st.subheader(
            "Final Preprocessed Dataset"
        )

        st.dataframe(
            preprocessed_df,
            use_container_width=True
        )

        st.subheader(
            "New Features"
        )

        new_features = [
            "net_revenue_egp_outlier_flag",
            "unit_price_egp_outlier_flag",
            "quantity_outlier_flag",
            "actual_delivery_days_outlier_flag",
            "lead_time_days_outlier_flag",
            "price_position",
            "low_stock_flag"
        ]

        existing_features = [
            col for col in new_features
            if col in preprocessed_df.columns
        ]

        st.write(
            existing_features
        )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv = (
        preprocessed_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    st.download_button(
        "⬇️ Download Preprocessed Dataset",
        csv,
        "Project1_Preprocessed_Features.csv",
        "text/csv"
    )
# =========================================================
# EDA PAGE
# =========================================================

def eda_page():

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.write(
        "Univariate EDA for identifiers, dates & time, "
        "categorical variables, numeric variables, "
        "and return complaint text."
    )

    # =====================================================
    # GET PREPROCESSED DATA
    # =====================================================

    if "preprocessed_df" in st.session_state:

        df = st.session_state["preprocessed_df"].copy()

        st.success(
            "Using data from Preprocessing ✅"
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload Project1_Preprocessed_Features.csv",
            type=["csv"],
            key="eda_file"
        )

        if uploaded_file is None:

            st.info(
                "Please run Preprocessing first "
                "or upload Project1_Preprocessed_Features.csv."
            )

            return

        df = pd.read_csv(uploaded_file)

    # =====================================================
    # EXTRA LIBRARIES
    # =====================================================

    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter

    # =====================================================
    # BASIC DATASET INFORMATION
    # =====================================================

    st.success("EDA dataset loaded successfully ✅")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Identifiers",
        "Dates & Time",
        "Categorical",
        "Numeric",
        "Return Complaints"
    ])

    # =====================================================
    # 1. IDENTIFIERS
    # =====================================================

    with tab1:

        st.subheader("Identifier Univariate EDA")

        id_cols = [
            "order_id",
            "customer_id",
            "product_id"
        ]

        id_cols = [
            col for col in id_cols
            if col in df.columns
        ]

        identifier_summary = []

        for col in id_cols:

            total_count = len(df[col])

            unique_count = df[col].nunique()

            duplicate_count = (
                total_count - unique_count
            )

            identifier_summary.append({
                "Identifier": col,
                "Total Rows": total_count,
                "Unique Keys": unique_count,
                "Duplicates": duplicate_count
            })

        st.dataframe(
            pd.DataFrame(identifier_summary),
            width="stretch",
            hide_index=True
        )

        if id_cols:

            selected_id = st.selectbox(
                "Select Identifier",
                id_cols,
                key="eda_identifier"
            )

            st.write(
                f"Top Repetitions: {selected_id}"
            )

            top_freq = (
                df[selected_id]
                .value_counts()
                .head(10)
            )

            top_freq_df = pd.DataFrame({
                selected_id: top_freq.index.astype(str),
                "Order Count": top_freq.values
            })

            st.dataframe(
                top_freq_df,
                width="stretch",
                hide_index=True
            )

            # Original notebook specifically visualized
            # Top 10 Product IDs
            if "product_id" in df.columns:

                st.subheader(
                    "Top 10 Most Frequent Product IDs"
                )

                top_products = (
                    df["product_id"]
                    .value_counts()
                    .head(10)
                )

                fig, ax = plt.subplots(
                    figsize=(10, 4)
                )

                sns.barplot(
                    x=top_products.index.astype(str),
                    y=top_products.values,
                    ax=ax
                )

                ax.set_title(
                    "Top 10 Most Frequent Product IDs",
                    fontsize=12,
                    fontweight="bold"
                )

                ax.set_xlabel("Product ID")
                ax.set_ylabel("Order Count")

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

    # =====================================================
    # 2. DATES & TIME
    # =====================================================

    with tab2:

        st.subheader(
            "Dates & Time Univariate EDA"
        )

        if "order_date" in df.columns:

            df["order_date"] = pd.to_datetime(
                df["order_date"],
                errors="coerce"
            )

            order_min = df["order_date"].min()
            order_max = df["order_date"].max()

            st.write(
                f"**Order Date Range:** "
                f"{order_min.date() if pd.notna(order_min) else 'N/A'} "
                f"to "
                f"{order_max.date() if pd.notna(order_max) else 'N/A'}"
            )

        if "return_request_date" in df.columns:

            df["return_request_date"] = pd.to_datetime(
                df["return_request_date"],
                errors="coerce"
            )

            return_min = (
                df["return_request_date"].min()
            )

            return_max = (
                df["return_request_date"].max()
            )

            st.write(
                f"**Return Date Range:** "
                f"{return_min.date() if pd.notna(return_min) else 'N/A'} "
                f"to "
                f"{return_max.date() if pd.notna(return_max) else 'N/A'}"
            )

        # Monthly Order Trend
        if "order_month" in df.columns:

            st.subheader(
                "Monthly Order Trend Distribution"
            )

            monthly_orders = (
                df["order_month"]
                .value_counts()
                .sort_index()
            )

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                monthly_orders.index,
                monthly_orders.values,
                marker="o",
                linewidth=2
            )

            ax.set_title(
                "Monthly Order Trend Distribution",
                fontsize=12,
                fontweight="bold"
            )

            ax.set_xlabel("Month")
            ax.set_ylabel("Number of Orders")
            ax.set_xticks(range(1, 13))
            ax.grid(
                True,
                linestyle="--",
                alpha=0.6
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        # Hourly Order Distribution
        if "order_hour" in df.columns:

            st.subheader(
                "Hourly Order Frequency Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            sns.countplot(
                data=df,
                x="order_hour",
                ax=ax
            )

            ax.set_title(
                "Hourly Order Frequency Distribution",
                fontsize=12,
                fontweight="bold"
            )

            ax.set_xlabel(
                "Hour of Day (0-23)"
            )

            ax.set_ylabel(
                "Order Volume"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

    # =====================================================
    # 3. CATEGORICAL VARIABLES
    # =====================================================

    with tab3:

        st.subheader(
            "Categorical Univariate EDA"
        )

        cat_cols = [
            "customer_gender",
            "customer_area",
            "branch",
            "sales_channel",
            "product_domain",
            "category",
            "brand",
            "payment_method",
            "marketing_source",
            "campaign_name",
            "courier",
            "delivery_status",
            "returned",
            "return_reason_category",
            "refund_method",
            "supplier",
            "weather",
            "season",
            "customer_age_group",
            "price_position",
            "on_time_flag",
            "low_stock_flag"
        ]

        cat_cols = [
            col for col in cat_cols
            if col in df.columns
        ]

        if cat_cols:

            selected_cat = st.selectbox(
                "Select Categorical Variable",
                cat_cols,
                key="eda_category"
            )

            counts = (
                df[selected_cat]
                .value_counts(
                    dropna=False
                )
            )

            percents = (
                df[selected_cat]
                .value_counts(
                    dropna=False,
                    normalize=True
                )
                * 100
            )

            summary = pd.DataFrame({
                "Category": counts.index.astype(str),
                "Count": counts.values,
                "Percentage (%)": percents.values.round(2)
            })

            st.write(
                f"Categorical Summary: {selected_cat}"
            )

            st.dataframe(
                summary,
                width="stretch",
                hide_index=True
            )

            top_cat = counts.head(10)

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            sns.barplot(
                y=top_cat.index.astype(str),
                x=top_cat.values,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {selected_cat} "
                "(Top Classes)",
                fontsize=12,
                fontweight="bold"
            )

            ax.set_xlabel("Frequency")
            ax.set_ylabel(selected_cat)

            for index, value in enumerate(
                top_cat.values
            ):

                ax.text(
                    value,
                    index,
                    f" {value} "
                    f"({value / len(df) * 100:.1f}%)",
                    va="center"
                )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

    # =====================================================
    # 4. NUMERIC VARIABLES
    # =====================================================

    with tab4:

        st.subheader(
            "Numeric Variables Univariate EDA"
        )

        num_cols = [
            "customer_age",
            "unit_price_egp",
            "quantity",
            "discount_percent",
            "expected_delivery_days",
            "actual_delivery_days",
            "rating",
            "stock_available_before_sale",
            "lead_time_days",
            "competitor_price_egp",
            "gross_revenue_egp",
            "discount_amount_egp",
            "net_revenue_egp",
            "delivery_delay_days",
            "price_gap_egp",
            "price_gap_percent",
            "return_request_lag_days"
        ]

        num_cols = [
            col for col in num_cols
            if col in df.columns
        ]

        numeric_summary = []

        for col in num_cols:

            s = pd.to_numeric(
                df[col],
                errors="coerce"
            ).dropna()

            if len(s) == 0:
                continue

            numeric_summary.append({
                "Column": col,
                "Count": int(s.count()),
                "Missing": int(
                    df[col].isna().sum()
                ),
                "Mean": round(
                    s.mean(),
                    2
                ),
                "Std": round(
                    s.std(),
                    2
                ),
                "Median": round(
                    s.median(),
                    2
                ),
                "Min": round(
                    s.min(),
                    2
                ),
                "Max": round(
                    s.max(),
                    2
                ),
                "Skewness": round(
                    s.skew(),
                    2
                )
            })

        numeric_summary_df = pd.DataFrame(
            numeric_summary
        )

        st.dataframe(
            numeric_summary_df,
            width="stretch",
            hide_index=True
        )

        if num_cols:

            selected_num = st.selectbox(
                "Select Numeric Variable",
                num_cols,
                key="eda_numeric"
            )

            numeric_data = pd.to_numeric(
                df[selected_num],
                errors="coerce"
            ).dropna()

            if not numeric_data.empty:

                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(12, 4)
                )

                # Histogram + KDE
                sns.histplot(
                    numeric_data,
                    kde=True,
                    bins=30,
                    ax=axes[0]
                )

                axes[0].set_title(
                    f"Histogram & KDE: "
                    f"{selected_num}",
                    fontsize=11,
                    fontweight="bold"
                )

                axes[0].set_xlabel(
                    selected_num
                )

                axes[0].set_ylabel(
                    "Frequency"
                )

                # Box Plot
                sns.boxplot(
                    x=numeric_data,
                    ax=axes[1]
                )

                axes[1].set_title(
                    f"Box Plot (Outliers Check): "
                    f"{selected_num}",
                    fontsize=11,
                    fontweight="bold"
                )

                axes[1].set_xlabel(
                    selected_num
                )

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

    # =====================================================
    # 5. RETURN COMPLAINT TEXT ANALYSIS
    # =====================================================

    with tab5:

        st.subheader(
            "Return Complaint Text Analysis"
        )

        if "return_complaint_text" not in df.columns:

            st.info(
                "return_complaint_text column "
                "is not available."
            )

        else:

            text_data = (
                df["return_complaint_text"]
                .dropna()
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Complaint Records",
                len(text_data)
            )

            c2.metric(
                "Missing Complaint Records",
                int(
                    df["return_complaint_text"]
                    .isna()
                    .sum()
                )
            )

            if len(text_data) > 0:

                words = [
                    w.lower().strip(
                        ".,!?:;-"
                    )
                    for text in text_data
                    for w in str(text).split()
                    if len(w) > 2
                ]

                top_words_df = pd.DataFrame(
                    Counter(words).most_common(15),
                    columns=[
                        "Word",
                        "Frequency"
                    ]
                )

                # Arabic display support
                try:

                    import arabic_reshaper
                    from bidi.algorithm import get_display

                    def fix_arabic(text):

                        if isinstance(
                            text,
                            str
                        ):

                            reshaped = (
                                arabic_reshaper
                                .reshape(text)
                            )

                            return get_display(
                                reshaped
                            )

                        return text

                    top_words_df[
                        "Word Display"
                    ] = (
                        top_words_df["Word"]
                        .apply(fix_arabic)
                    )

                except ImportError:

                    top_words_df[
                        "Word Display"
                    ] = top_words_df["Word"]

                st.write(
                    "Top 15 Most Frequent Words "
                    "in Return Complaints"
                )

                st.dataframe(
                    top_words_df[
                        [
                            "Word",
                            "Frequency"
                        ]
                    ],
                    width="stretch",
                    hide_index=True
                )

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                sns.barplot(
                    data=top_words_df,
                    x="Frequency",
                    y="Word Display",
                    ax=ax
                )

                ax.set_title(
                    "Top 15 Most Frequent Words "
                    "in Return Complaints",
                    fontsize=12,
                    fontweight="bold"
                )

                ax.set_xlabel("Frequency")
                ax.set_ylabel("Word")

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

# =========================================================
# VISUALIZATIONS PAGE - ABDULLAH (1-15)
# =========================================================

def visualizations_page():

    st.title("📈 Project Visualizations")

    st.write(
        "This section contains Abdullah's 15 visualizations. "
        "Dina's remaining 15 visualizations will be added later."
    )

    # Get preprocessed data
    if "preprocessed_df" in st.session_state:
        df = st.session_state["preprocessed_df"].copy()
        st.success("Using data from Preprocessing ✅")

    else:
        uploaded_file = st.file_uploader(
            "Upload Project1_Preprocessed_Features.csv",
            type=["csv"],
            key="visualizations_file"
        )

        if uploaded_file is None:
            st.info(
                "Please run Preprocessing first or upload "
                "Project1_Preprocessed_Features.csv."
            )
            return

        df = pd.read_csv(uploaded_file)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("tab10")

    visualizations = [
        "1. Customer Gender vs Net Revenue",
        "2. Customer Age Group vs Average Order Value",
        "3. Customer Area vs Number of Orders",
        "4. Customer Area vs Return Rate",
        "5. Branch vs Net Revenue",
        "6. Sales Channel vs Net Revenue",
        "7. Product Domain vs Net Revenue",
        "8. Category vs Quantity Sold",
        "9. Brand vs Net Revenue",
        "10. Product Name vs Customer Rating",
        "11. Unit Price vs Quantity",
        "12. Discount Percent vs Quantity",
        "13. Discount Percent vs Net Revenue",
        "14. Unit Price vs Competitor Price",
        "15. Price Gap vs Quantity Sold"
    ]

    selected = st.selectbox(
        "Choose a visualization",
        visualizations
    )

    # 1
    if selected.startswith("1."):

        gender_perf = df.groupby("customer_gender").agg(
            total_net_revenue=("net_revenue_egp", "sum"),
            avg_order_value=("net_revenue_egp", "mean"),
            order_count=("order_id", "count")
        ).reset_index()

        st.subheader("Customer Gender vs Net Revenue")
        st.dataframe(gender_perf, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(
            data=gender_perf,
            x="customer_gender",
            y="total_net_revenue",
            ax=axes[0]
        )

        axes[0].set_title("Total Net Revenue by Gender")
        axes[0].set_ylabel("Total Net Revenue (EGP)")

        sns.barplot(
            data=gender_perf,
            x="customer_gender",
            y="avg_order_value",
            ax=axes[1]
        )

        axes[1].set_title("Average Order Value by Gender")
        axes[1].set_ylabel("AOV (EGP)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 2
    elif selected.startswith("2."):

        age_perf = df.groupby(
            "customer_age_group",
            observed=False
        ).agg(
            avg_order_value=("net_revenue_egp", "mean"),
            sample_size=("order_id", "count")
        ).reset_index()

        st.subheader("Customer Age Group vs Average Order Value")
        st.dataframe(age_perf, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=age_perf,
            x="customer_age_group",
            y="avg_order_value",
            ax=ax
        )

        ax.set_title("Average Order Value by Customer Age Group")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Average Order Value (EGP)")

        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 3
    elif selected.startswith("3."):

        area_orders = df["customer_area"].value_counts().reset_index()
        area_orders.columns = ["customer_area", "order_count"]

        area_orders["percentage"] = (
            area_orders["order_count"]
            / area_orders["order_count"].sum()
        ) * 100

        top_areas = area_orders.head(10)

        st.subheader("Customer Area vs Number of Orders")
        st.dataframe(top_areas, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=top_areas,
            y="customer_area",
            x="order_count",
            ax=ax
        )

        ax.set_title("Top 10 Customer Areas by Number of Orders")
        ax.set_xlabel("Order Count")
        ax.set_ylabel("Customer Area")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 4
    elif selected.startswith("4."):

        area_returns = df.groupby("customer_area").agg(
            order_count=("order_id", "count"),
            return_rate=("returned_flag", "mean")
        ).reset_index()

        area_filtered = area_returns[
            area_returns["order_count"] >= 30
        ].sort_values(
            "return_rate",
            ascending=False
        )

        top_areas = area_filtered.head(10)

        st.subheader("Customer Area vs Return Rate")
        st.caption("Minimum threshold = 30 orders")
        st.dataframe(top_areas, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=top_areas,
            y="customer_area",
            x="return_rate",
            ax=ax
        )

        ax.set_title(
            "Top Customer Areas by Return Rate (Min 30 Orders)"
        )

        ax.set_xlabel("Return Rate")
        ax.set_ylabel("Customer Area")

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: f"{x:.0%}"
            )
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 5
    elif selected.startswith("5."):

        branch_perf = df.groupby("branch").agg(
            total_net_revenue=("net_revenue_egp", "sum"),
            order_count=("order_id", "count"),
            avg_order_value=("net_revenue_egp", "mean")
        ).reset_index().sort_values(
            "total_net_revenue",
            ascending=False
        )

        st.subheader("Branch vs Net Revenue")
        st.dataframe(branch_perf, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=branch_perf,
            x="branch",
            y="total_net_revenue",
            ax=ax
        )

        ax.set_title("Total Net Revenue by Branch")
        ax.set_xlabel("Branch")
        ax.set_ylabel("Total Net Revenue (EGP)")

        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 6
    elif selected.startswith("6."):

        channel_perf = df.groupby("sales_channel").agg(
            total_net_revenue=("net_revenue_egp", "sum"),
            order_count=("order_id", "count"),
            avg_order_value=("net_revenue_egp", "mean")
        ).reset_index()

        st.subheader("Sales Channel vs Net Revenue")
        st.dataframe(channel_perf, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=channel_perf,
            x="sales_channel",
            y="total_net_revenue",
            ax=ax
        )

        ax.set_title("Total Net Revenue by Sales Channel")
        ax.set_xlabel("Sales Channel")
        ax.set_ylabel("Total Net Revenue (EGP)")

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():,.0f}",
                (
                    p.get_x() + p.get_width() / 2,
                    p.get_height()
                ),
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 7
    elif selected.startswith("7."):

        domain_perf = df.groupby("product_domain").agg(
            total_net_revenue=("net_revenue_egp", "sum"),
            total_quantity=("quantity", "sum")
        ).reset_index().sort_values(
            "total_net_revenue",
            ascending=False
        )

        st.subheader("Product Domain vs Net Revenue")
        st.dataframe(domain_perf, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=domain_perf,
            x="product_domain",
            y="total_net_revenue",
            ax=ax
        )

        ax.set_title("Total Net Revenue by Product Domain")
        ax.set_xlabel("Product Domain")
        ax.set_ylabel("Net Revenue (EGP)")

        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 8
    elif selected.startswith("8."):

        cat_perf = df.groupby("category").agg(
            total_quantity=("quantity", "sum"),
            total_net_revenue=("net_revenue_egp", "sum")
        ).reset_index().sort_values(
            "total_quantity",
            ascending=False
        )

        st.subheader("Category vs Quantity Sold")
        st.dataframe(cat_perf, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=cat_perf,
            y="category",
            x="total_quantity",
            ax=ax
        )

        ax.set_title(
            "Total Quantity Sold by Product Category"
        )

        ax.set_xlabel("Quantity Sold")
        ax.set_ylabel("Category")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 9
    elif selected.startswith("9."):

        brand_perf = df.groupby("brand").agg(
            total_net_revenue=("net_revenue_egp", "sum"),
            avg_rating=("rating", "mean"),
            order_count=("order_id", "count")
        ).reset_index()

        top_brands = brand_perf[
            brand_perf["order_count"] >= 10
        ].sort_values(
            "total_net_revenue",
            ascending=False
        ).head(10)

        st.subheader("Brand vs Net Revenue")
        st.caption("Minimum 10 orders")
        st.dataframe(top_brands, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=top_brands,
            y="brand",
            x="total_net_revenue",
            ax=ax
        )

        ax.set_title(
            "Top 10 Brands by Net Revenue"
        )

        ax.set_xlabel("Net Revenue (EGP)")
        ax.set_ylabel("Brand")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 10
    elif selected.startswith("10."):

        prod_rating = df.groupby("product_name").agg(
            avg_rating=("rating", "mean"),
            review_count=("rating", "count")
        ).reset_index()

        top_rated_prods = prod_rating[
            prod_rating["review_count"] >= 15
        ].sort_values(
            "avg_rating",
            ascending=False
        ).head(10)

        st.subheader("Product Name vs Customer Rating")
        st.caption("Minimum 15 reviews")
        st.dataframe(top_rated_prods, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=top_rated_prods,
            y="product_name",
            x="avg_rating",
            ax=ax
        )

        ax.set_title(
            "Top Rated Products (Min 15 Reviews)"
        )

        ax.set_xlabel("Average Rating")
        ax.set_ylabel("Product Name")
        ax.set_xlim(1, 5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 11
    elif selected.startswith("11."):

        st.subheader("Unit Price vs Quantity")

        st.dataframe(
            df[
                ["unit_price_egp", "quantity"]
            ].describe()
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.scatterplot(
            data=df,
            x="unit_price_egp",
            y="quantity",
            alpha=0.5,
            ax=ax
        )

        ax.set_title("Unit Price vs Order Quantity")
        ax.set_xlabel("Unit Price (EGP)")
        ax.set_ylabel("Quantity")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 12
    elif selected.startswith("12."):

        discount_summary = (
            df.groupby("discount_percent")["quantity"]
            .agg(
                ["mean", "median", "count"]
            )
            .reset_index()
        )

        st.subheader("Discount Percent vs Quantity")
        st.dataframe(discount_summary, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.scatterplot(
            data=df,
            x="discount_percent",
            y="quantity",
            alpha=0.4,
            ax=ax
        )

        ax.set_title(
            "Discount Percent vs Quantity Sold"
        )

        ax.set_xlabel("Discount Percent")
        ax.set_ylabel("Quantity")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 13
    elif selected.startswith("13."):

        corr_val = df[
            "discount_percent"
        ].corr(
            df["net_revenue_egp"]
        )

        st.subheader(
            "Discount Percent vs Net Revenue"
        )

        st.metric(
            "Correlation",
            f"{corr_val:.4f}"
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.regplot(
            data=df,
            x="discount_percent",
            y="net_revenue_egp",
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
            ax=ax
        )

        ax.set_title(
            "Discount Percent vs Net Revenue"
        )

        ax.set_xlabel("Discount Percent")
        ax.set_ylabel("Net Revenue (EGP)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 14
    elif selected.startswith("14."):

        comp_df = df.dropna(
            subset=[
                "competitor_price_egp",
                "unit_price_egp"
            ]
        )

        st.subheader(
            "Unit Price vs Competitor Price"
        )

        st.dataframe(
            comp_df[
                [
                    "unit_price_egp",
                    "competitor_price_egp"
                ]
            ].describe()
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        sns.scatterplot(
            data=comp_df,
            x="competitor_price_egp",
            y="unit_price_egp",
            alpha=0.5,
            ax=ax
        )

        max_val = max(
            comp_df["competitor_price_egp"].max(),
            comp_df["unit_price_egp"].max()
        )

        ax.plot(
            [0, max_val],
            [0, max_val],
            "r--",
            label="Equal Price Baseline"
        )

        ax.set_title(
            "Unit Price vs Competitor Price"
        )

        ax.set_xlabel(
            "Competitor Price (EGP)"
        )

        ax.set_ylabel(
            "Unit Price (EGP)"
        )

        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 15
    elif selected.startswith("15."):

        vis_df = df.copy()

        vis_df["price_position"] = np.select(
            [
                vis_df["price_gap_egp"] < 0,
                vis_df["price_gap_egp"] == 0,
                vis_df["price_gap_egp"] > 0
            ],
            [
                "Below Competitor",
                "Equal",
                "Above Competitor"
            ],
            default="Unknown"
        )

        price_pos_summary = (
            vis_df.groupby(
                "price_position"
            )["quantity"]
            .agg(
                ["sum", "mean", "count"]
            )
            .reset_index()
        )

        st.subheader(
            "Price Gap vs Quantity Sold"
        )

        st.dataframe(
            price_pos_summary,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=price_pos_summary,
            x="price_position",
            y="mean",
            ax=ax
        )

        ax.set_title(
            "Average Order Quantity by Competitive Price Position"
        )

        ax.set_xlabel(
            "Price Position Relative to Competitor"
        )

        ax.set_ylabel(
            "Average Order Quantity"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)                
# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Cleaning",
        "Preprocessing",
        "EDA",
        "Visualizations",
        "Machine Learning"
    ]
)
if page == "Home":
    home()

elif page == "Data Cleaning":
    cleaning_page()

elif page == "Preprocessing":
    preprocessing_page()

elif page == "EDA":
    eda_page()

elif page == "Visualizations":
  visualizations_page()

elif page == "Machine Learning":
    st.title("🤖 Machine Learning")
    st.info("Machine Learning will be added later.")


# =========================
#