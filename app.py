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
    st.title("📊 EDA")
    st.info("EDA will be added later.")

elif page == "Visualizations":
    st.title("📈 Visualizations")
    st.info("Visualizations will be added later.")

elif page == "Machine Learning":
    st.title("🤖 Machine Learning")
    st.info("Machine Learning will be added later.")


# =========================
#