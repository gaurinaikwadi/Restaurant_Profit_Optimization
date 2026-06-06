import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/SkyCity Auckland Restaurants & Bars.csv")

df["TotalNetProfit"] = (
    df["InStoreNetProfit"]
    + df["UberEatsNetProfit"]
    + df["DoorDashNetProfit"]
    + df["SelfDeliveryNetProfit"]
)

plt.figure(figsize=(8,5))
plt.hist(df["TotalNetProfit"], bins=30)
plt.title("Profit Distribution")
plt.savefig("profit_distribution.png")

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True))
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")