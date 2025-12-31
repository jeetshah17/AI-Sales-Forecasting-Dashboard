import pandas as pd
from prophet import Prophet
import pickle
import os

BASE_PATH = r"D:\AI-sales-forecasting-dashboard"

# Load sales data (has headers)
sales = pd.read_csv(os.path.join(BASE_PATH, "1_target_ts.csv"))

# Load price data (NO headers → define manually)
price = pd.read_csv(
    os.path.join(BASE_PATH, "2_related_ts.csv"),
    header=None,
    names=["item", "org", "date", "unit_price"]
)

# Select first item & org
ITEM_ID = sales["item"].iloc[0]
ORG_ID = sales["org"].iloc[0]

sales = sales[(sales["item"] == ITEM_ID) & (sales["org"] == ORG_ID)]
price = price[(price["item"] == ITEM_ID) & (price["org"] == ORG_ID)]

# Merge datasets
df = pd.merge(
    sales,
    price,
    on=["item", "org", "date"],
    how="left"
)

# Prepare Prophet format
df = df.rename(columns={
    "date": "ds",
    "quantity": "y"
})

df["ds"] = pd.to_datetime(df["ds"])
df = df[["ds", "y", "unit_price"]]

# Train model
model = Prophet()
model.add_regressor("unit_price")
model.fit(df)

# Save model
model_path = os.path.join(BASE_PATH, "model", "sales_forecast_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)


print("✅ Model trained successfully")
print("📦 Model saved at:", model_path)
