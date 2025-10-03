import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import math

# Load trained model
model = joblib.load("best_model.pkl")

st.title("🚚 Amazon Delivery Time Prediction")

# Inputs
agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)
store_lat = st.number_input("Store Latitude", value=28.6)
store_long = st.number_input("Store Longitude", value=77.2)
drop_lat = st.number_input("Drop Latitude", value=28.7)
drop_long = st.number_input("Drop Longitude", value=77.3)

vehicle = st.selectbox("Vehicle", ["bike", "car", "scooter"])
weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
area = st.selectbox("Area", ["Urban", "Suburban", "Rural"])
category = st.selectbox("Category", ["Food", "Grocery", "Electronics"])

order_date = st.date_input("Order Date")
order_time = st.time_input("Order Time")

# ----- Feature Engineering (must match training pipeline) -----
order_datetime = datetime.combine(order_date, order_time)
order_month = order_datetime.month
order_day = order_datetime.day

# Haversine Distance (if you used it in training)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

distance_km = haversine(store_lat, store_long, drop_lat, drop_long)

# Create input dataframe with ALL columns
input_df = pd.DataFrame([{
    "Vehicle": vehicle,
    "Weather": weather,
    "Traffic": traffic,
    "Area": area,
    "Category": category,
    "Agent_Age": agent_age,
    "Agent_Rating": agent_rating,
    "Store_Latitude": store_lat,
    "Store_Longitude": store_long,
    "Drop_Latitude": drop_lat,
    "Drop_Longitude": drop_long,
    "Order_Month": order_month,
    "Order_Day": order_day,
    "Distance_km": distance_km
}])

# Prediction
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_df)[0]
    st.success(f"📦 Estimated Delivery Time: {prediction:.2f} hours")
