import streamlit as st
import pandas as pd
import pickle
from prophet.plot import plot_plotly

st.set_page_config(page_title="AI Sales Forecasting Dashboard", layout="wide")

st.title("📈 AI Sales Forecasting Dashboard")
st.write("Real-world retail sales forecasting using Machine Learning")

# Load trained model
with open("model/sales_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)

# Upload files
sales_file = st.file_uploader("Upload 1_target_ts.csv", type=["csv"])
price_file = st.file_uploader("Upload 2_related_ts.csv", type=["csv"])

if sales_file and price_file:
    # Sales file HAS headers
    sales = pd.read_csv(sales_file)

    # Price file has NO headers → define manually (IMPORTANT FIX)
    price = pd.read_csv(
        price_file,
        header=None,
        names=["item", "org", "date", "unit_price"]
    )

    # Select item & org
    item = st.selectbox("Select Item", sales["item"].unique())
    org = st.selectbox("Select Org Unit", sales["org"].unique())

    sales = sales[(sales["item"] == item) & (sales["org"] == org)]
    price = price[(price["item"] == item) & (price["org"] == org)]

    # Merge datasets
    df = pd.merge(
        sales,
        price,
        on=["item", "org", "date"],
        how="left"
    )

    # Prophet format
    df = df.rename(columns={
        "date": "ds",
        "quantity": "y"
    })

    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", "y", "unit_price"]]

    st.subheader("📊 Historical Sales Data")
    st.dataframe(df)

    months = st.slider("Forecast Months", 1, 24, 6)

    # Future dataframe
    future = model.make_future_dataframe(periods=months, freq="MS")

    # Assume last price continues
    future["unit_price"] = df["unit_price"].iloc[-1]

    forecast = model.predict(future)

    st.subheader("📉 Sales Forecast")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Forecast Table")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

else:
    st.info("Please upload both CSV files to proceed.")
