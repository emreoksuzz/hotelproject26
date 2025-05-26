import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime
from joblib import load

import os
from joblib import load

# Bu .py dosyasının bulunduğu dizini bulur
current_dir = os.path.dirname(os.path.abspath(__file__))

# .pkl dosyasını onun içinden yükler
model_path = os.path.join(current_dir, "xgboost_model28.pkl")
model = load(model_path)

st.set_page_config(
    page_title="Hotel Cancellation Prediction Project",
    page_icon="🏨",
    menu_items={"Get help": "mailto:emreoksuzbusiness@outlook.com"}
)

st.title("🏨 Hotel Cancellation Prediction")
st.markdown("Predict the probability that a hotel booking will be canceled based on booking details.")
st.image("https://press.sripanwa.com/assets/uploads/data_img/77724-204-one-bedroom-luxury-villa.jpg")

# 📥 Sidebar inputları
st.sidebar.header("🔧 Booking Details")

lead_time = st.sidebar.number_input("Lead Time", 0, 750, 100)
weekend_nights = st.sidebar.number_input("Weekend Nights", 0, 19, 1)
week_nights = st.sidebar.number_input("Week Nights", 0, 30, 2)
adults = st.sidebar.number_input("Adults", 0, 55, 2)
children = st.sidebar.number_input("Children", 0, 10, 0)
adr = st.sidebar.number_input("Average Daily Rate (ADR)", 0.0, 600.0, 100.0)
parking = st.sidebar.selectbox("Car Parking Spaces", [0, 1, 2, 3])
special_requests = st.sidebar.number_input("Special Requests", 0, 5, 0)
previous_cancellations = st.sidebar.number_input("Previous Cancellations", min_value=0, value=0)
previous_bookings_not_canceled = st.sidebar.number_input("Previous Bookings (Not Canceled)", min_value=0, value=0)
booking_changes = st.sidebar.number_input("Booking Changes", min_value=0, value=0)
days_in_waiting_list = st.sidebar.number_input("Days in Waiting List", min_value=0, value=0)
# Kategorik inputlar
meal = st.sidebar.selectbox("Meal Plan", ["BB", "FB", "HB", "SC", "Undefined"])
market = st.sidebar.selectbox("Market Segment", ["Aviation", "Complementary", "Corporate", "Direct", "Groups", "Offline TA/TO", "Online TA", "Undefined"])
deposit = st.sidebar.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
customer = st.sidebar.selectbox("Customer Type", ["Contract", "Group", "Transient", "Transient-Party"])
season = st.sidebar.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])

# 🔨 Feature dictionary (model ile birebir aynı)
input_data = {
    'lead_time': [lead_time],
    'stays_in_weekend_nights': [weekend_nights],
    'stays_in_week_nights': [week_nights],
    'adults': [adults],
    'children': [children],
    'previous_cancellations': [previous_cancellations],
    'previous_bookings_not_canceled': [previous_bookings_not_canceled],
    'booking_changes': [booking_changes],
    'days_in_waiting_list': [days_in_waiting_list],
    'adr': [adr],
    'required_car_parking_spaces': [parking],
    'total_of_special_requests': [special_requests],
    'meal_BB': [1 if meal == "BB" else 0],
    'meal_FB': [1 if meal == "FB" else 0],
    'meal_HB': [1 if meal == "HB" else 0],
    'meal_SC': [1 if meal == "SC" else 0],
    'meal_Undefined': [1 if meal == "Undefined" else 0],
    'market_segment_Aviation': [1 if market == "Aviation" else 0],
    'market_segment_Complementary': [1 if market == "Complementary" else 0],
    'market_segment_Corporate': [1 if market == "Corporate" else 0],
    'market_segment_Direct': [1 if market == "Direct" else 0],
    'market_segment_Groups': [1 if market == "Groups" else 0],
    'market_segment_Offline TA/TO': [1 if market == "Offline TA/TO" else 0],
    'market_segment_Online TA': [1 if market == "Online TA" else 0],
    'market_segment_Undefined': [1 if market == "Undefined" else 0],
    'deposit_type_No Deposit': [1 if deposit == "No Deposit" else 0],
    'deposit_type_Non Refund': [1 if deposit == "Non Refund" else 0],
    'deposit_type_Refundable': [1 if deposit == "Refundable" else 0],
    'customer_type_Contract': [1 if customer == "Contract" else 0],
    'customer_type_Group': [1 if customer == "Group" else 0],
    'customer_type_Transient': [1 if customer == "Transient" else 0],
    'customer_type_Transient-Party': [1 if customer == "Transient-Party" else 0],
    'season_Fall': [1 if season == "Fall" else 0],
    'season_Spring': [1 if season == "Spring" else 0],
    'season_Summer': [1 if season == "Summer" else 0],
    'season_Winter': [1 if season == "Winter" else 0]
}

input_df = pd.DataFrame(input_data)

# 🔮 Tahmin ve çıktı
if st.sidebar.button("Submit"):
    if (previous_cancellations == 1 and previous_bookings_not_canceled == 1) or (previous_cancellations == 0 and previous_bookings_not_canceled == 0):
        st.error("⚠️ A customer cannot have both 'Previous Cancellations' and 'Previous Bookings Not Canceled' set to 1 or both to 0.")
    else:
        proba = model.predict_proba(input_df)[0][1]
        percent = round(proba * 100, 2)
        prediction = "Canceled" if proba > 0.5 else "Not Canceled"

        st.subheader("Prediction Result")
        st.table(pd.DataFrame({
            "Date": [date.today()],
            "Time": [datetime.now().strftime("%H:%M:%S")],
            "Prediction": [prediction],
            "Cancel Probability": [f"%{percent}"]
        }))

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Canceled", "Not Canceled"], [percent, 100 - percent], color=["#E74C3C", "#2ECC71"], width=0.5)

        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        ax.set_title("🔍 Cancellation Probability", fontsize=14, pad=15)

        # İçeri yazı yerleştir
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height / 2),  # barın ortası
                        ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold')

        # Stil sadeleştirme
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)
