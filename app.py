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


# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Cleaning",
        "EDA",
        "Visualizations",
        "Machine Learning"
    ]
)
if page == "Home":
    home()

elif page == "Data Cleaning":
    cleaning_page()

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