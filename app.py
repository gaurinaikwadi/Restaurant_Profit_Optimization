import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


st.info(
    "This dashboard predicts restaurant profit based on order volume, channel mix, delivery costs, and commission rates."
)


st.markdown("""
<style>

.main {
    background-color: white;
}

h1 {
    color: #1F4E79;
}

[data-testid="stMetricValue"] {
    color: #0F9D58;
    font-size: 28px;
}

[data-testid="stSidebar"] {
    background-color: #F5F7FA;
}

</style>
""", unsafe_allow_html=True)



# Load Model
model = pickle.load(
    open("models/profit_model.pkl", "rb")
)

st.set_page_config(
    page_title="SkyCity Restaurant Profit Optimizer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<h1>🍽️ Restaurant Profit Optimization Dashboard</h1>
<h3 style='text-align:center;color:#AAAAAA;'>
Predictive Modeling & Profit Optimization for Multi-Channel Restaurant Operations
</h3>
""", unsafe_allow_html=True)


st.sidebar.header("Input Parameters")

growth = st.sidebar.slider(
    "Growth Factor",
    0.8,
    1.5,
    1.0
)

aov = st.sidebar.slider(
    "Average Order Value",
    10.0,
    100.0,
    35.0
)

orders = st.sidebar.slider(
    "Monthly Orders",
    100,
    10000,
    3000
)

commission = st.sidebar.slider(
    "Commission Rate",
    0.05,
    0.40,
    0.25
)

radius = st.sidebar.slider(
    "Delivery Radius KM",
    1.0,
    20.0,
    5.0
)

delivery_cost = st.sidebar.slider(
    "Delivery Cost Per Order",
    1.0,
    10.0,
    3.0
)

instore = st.sidebar.slider(
    "InStore Share",
    0.0,
    1.0,
    0.40
)

ue = st.sidebar.slider(
    "Uber Eats Share",
    0.0,
    1.0,
    0.30
)

dd = st.sidebar.slider(
    "DoorDash Share",
    0.0,
    1.0,
    0.20
)

sd = st.sidebar.slider(
    "Self Delivery Share",
    0.0,
    1.0,
    0.10
)

instore_orders = st.sidebar.slider(
    "InStore Orders",
    0,
    10000,
    1000
)

uber_orders = st.sidebar.slider(
    "Uber Eats Orders",
    0,
    10000,
    800
)

doordash_orders = st.sidebar.slider(
    "DoorDash Orders",
    0,
    10000,
    600
)

self_orders = st.sidebar.slider(
    "Self Delivery Orders",
    0,
    10000,
    400
)

data = pd.DataFrame({
    "GrowthFactor":[growth],
    "AOV":[aov],
    "MonthlyOrders":[orders],

    "InStoreOrders":[instore_orders],
    "UberEatsOrders":[uber_orders],
    "DoorDashOrders":[doordash_orders],
    "SelfDeliveryOrders":[self_orders],

    "CommissionRate":[commission],
    "DeliveryRadiusKM":[radius],
    "DeliveryCostPerOrder":[delivery_cost],

    "InStoreShare":[instore],
    "UE_share":[ue],
    "DD_share":[dd],
    "SD_share":[sd]
})

prediction = model.predict(data)[0]


col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"💰 Predicted Profit\n\n${prediction:,.2f}")

with col2:
    st.success(f"📈 Commission Rate\n\n{commission*100:.0f}%")

with col3:
    st.warning(f"📦 Monthly Orders\n\n{orders:,}")

    
    
st.subheader("Input Summary")
st.dataframe(
    data,
    use_container_width=True
)

# Download Button
csv = data.to_csv(index=False)

st.download_button(
    label="📥 Download Input Data",
    data=csv,
    file_name="restaurant_input.csv",
    mime="text/csv"
)



st.subheader("📊 Channel Share Distribution")

chart_data = pd.DataFrame({
    "Channel": ["InStore", "Uber Eats", "DoorDash", "Self Delivery"],
    "Share": [instore, ue, dd, sd]
})

fig = px.pie(
    chart_data,
    values="Share",
    names="Channel",
    hole=0.45,
    title="Restaurant Channel Mix",
    color="Channel"
)

fig.update_layout(
    height=550,
    font_size=16
)

fig.update_traces(
    marker=dict(
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )
)


col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Channel Mix")
    st.write(f"🏪 InStore: {instore*100:.1f}%")
    st.write(f"🛵 Uber Eats: {ue*100:.1f}%")
    st.write(f"🚚 DoorDash: {dd*100:.1f}%")
    st.write(f"📦 Self Delivery: {sd*100:.1f}%")
    

st.subheader("What-If Analysis")

base_profit = prediction

new_commission = commission + 0.05

scenario_data = data.copy()

scenario_data["CommissionRate"] = new_commission

new_profit = model.predict(scenario_data)[0]

difference = new_profit - base_profit


if difference > 0:
    st.success(
        f"Profit increases by ${difference:,.2f} when commission increases by 5%"
    )
else:
    st.error(
        f"Profit decreases by ${abs(difference):,.2f} when commission increases by 5%"
    )

    

st.subheader("💡 Business Recommendation")

if ue > 0.50:
    st.warning(
        "High Uber Eats dependency may reduce profits because of commission charges."
    )

elif sd > 0.30:
    st.success(
        "Strong Self Delivery share improves margins and profitability."
    )

else:
    st.info(
        "Current channel mix appears balanced."
    )


st.subheader("📈 Key Insights")

st.markdown(f"""
- Predicted Profit: **${prediction:,.2f}**
- Largest Channel: **{
    max(
        [('InStore', instore),
         ('Uber Eats', ue),
         ('DoorDash', dd),
         ('Self Delivery', sd)],
        key=lambda x: x[1]
    )[0]
}**
- Commission Rate: **{commission*100:.0f}%**
- Monthly Orders: **{orders:,}**
""")



st.markdown("---")

st.markdown(
    "Developed by Gaur Naikwadi | Internship Project | 2026"
)