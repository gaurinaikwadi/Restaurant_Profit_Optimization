import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("data/SkyCity Auckland Restaurants & Bars.csv")

# Create Target Variable
df["TotalNetProfit"] = (
    df["InStoreNetProfit"]
    + df["UberEatsNetProfit"]
    + df["DoorDashNetProfit"]
    + df["SelfDeliveryNetProfit"]
)

# Features
features = [
    "GrowthFactor",
    "AOV",
    "MonthlyOrders",
    "InStoreOrders",
    "UberEatsOrders",
    "DoorDashOrders",
    "SelfDeliveryOrders",
    "CommissionRate",
    "DeliveryRadiusKM",
    "DeliveryCostPerOrder",
    "InStoreShare",
    "UE_share",
    "DD_share",
    "SD_share"
]

X = df[features]
y = df["TotalNetProfit"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

# Save Model
pickle.dump(model, open("models/profit_model.pkl", "wb"))

print("\nModel Saved Successfully")

importance = model.feature_importances_

plt.figure(figsize=(10,5))

plt.barh(features, importance)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

print("Feature Importance Saved")