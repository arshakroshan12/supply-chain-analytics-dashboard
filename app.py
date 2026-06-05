import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)

# ----------------------------------
# LOAD DATA & MODEL
# ----------------------------------
df = pd.read_csv("walmart_processed.csv")
model = joblib.load("xgboost_model.pkl")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.title("📦 Supply Chain Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Store Analysis",
        "Forecasting",
        "Model Insights"
    ]
)

# ==================================
# OVERVIEW PAGE
# ==================================
if page == "Overview":

    st.title("📦 Supply Chain Analytics Dashboard")

    total_sales = df["Weekly_Sales"].sum()
    avg_sales = df["Weekly_Sales"].mean()
    best_store = df.groupby("Store")["Weekly_Sales"].sum().idxmax()
    num_stores = df["Store"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "📊 Avg Weekly Sales",
        f"${avg_sales:,.0f}"
    )

    col3.metric(
        "🏆 Best Store",
        int(best_store)
    )

    col4.metric(
        "🏪 Stores",
        num_stores
    )

    st.divider()

    st.subheader("📈 Weekly Sales Trend")

    sales_trend = df.groupby("Date")["Weekly_Sales"].sum()

    st.line_chart(sales_trend)

# ==================================
# STORE ANALYSIS PAGE
# ==================================
elif page == "Store Analysis":

    st.title("🏪 Store Analysis")

    selected_store = st.selectbox(
        "Select Store",
        sorted(df["Store"].unique())
    )

    store_data = df[df["Store"] == selected_store]

    store_sales = store_data["Weekly_Sales"].sum()

    st.metric(
        "Store Total Sales",
        f"${store_sales:,.0f}"
    )

    st.subheader(
        f"Sales Trend - Store {selected_store}"
    )

    store_trend = store_data.groupby(
        "Date"
    )["Weekly_Sales"].sum()

    st.line_chart(store_trend)

    st.subheader("Recent Records")

    st.dataframe(
        store_data.tail(10),
        use_container_width=True
    )

# ==================================
# FORECASTING PAGE
# ==================================
elif page == "Forecasting":

    st.title("🔮 Sales Forecasting")

    col1, col2 = st.columns(2)

    with col1:

        store = st.number_input(
            "Store",
            min_value=1,
            max_value=45,
            value=20
        )

        holiday = st.selectbox(
            "Holiday Week",
            [0, 1]
        )

        temp = st.slider(
            "Temperature",
            0.0,
            100.0,
            60.0
        )

        fuel = st.slider(
            "Fuel Price",
            2.0,
            5.0,
            3.5
        )

    with col2:

        cpi = st.slider(
            "CPI",
            120.0,
            230.0,
            180.0
        )

        unemployment = st.slider(
            "Unemployment",
            3.0,
            15.0,
            8.0
        )

        year = st.selectbox(
            "Year",
            [2010, 2011, 2012]
        )

        month = st.slider(
            "Month",
            1,
            12,
            6
        )

        week = st.slider(
            "Week",
            1,
            52,
            25
        )

    if st.button("🚀 Predict Weekly Sales"):

        input_data = pd.DataFrame({
            "Store": [store],
            "Holiday_Flag": [holiday],
            "Temperature": [temp],
            "Fuel_Price": [fuel],
            "CPI": [cpi],
            "Unemployment": [unemployment],
            "Year": [year],
            "Month": [month],
            "Week": [week]
        })

        prediction = model.predict(input_data)[0]

        st.metric(
            "Predicted Weekly Sales",
            f"${prediction:,.0f}"
        )

# ==================================
# MODEL INSIGHTS PAGE
# ==================================
elif page == "Model Insights":

    st.title("🎯 Model Insights")

    st.subheader("Feature Importance")

    feature_names = [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Week"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "MAE": [62007, 59128],
        "RMSE": [113945, 94826],
        "R²": [0.960, 0.972]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Processed Dataset",
        data=csv,
        file_name="walmart_processed.csv",
        mime="text/csv"
    )

