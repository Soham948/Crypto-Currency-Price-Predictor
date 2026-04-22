import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Multi Asset Price Predictor", page_icon="📊", layout="wide")

st.title("📊 Multi-Asset Price Prediction Dashboard")

# -------------------------
# Asset Selection
# -------------------------
asset = st.sidebar.selectbox(
    "Select Asset",
    ["Ethereum", "Gold", "Silver", "Crude Oil"]
)

# -------------------------
# Asset Config
# -------------------------
if asset == "Ethereum":
    symbol = "ETH-USD"
    model_path = "ethereum_price_model.keras"
    widget_symbol = "BINANCE:ETHUSDT"

elif asset == "Gold":
    symbol = "GC=F"
    model_path = "gold_price_model.h5"
    widget_symbol = "OANDA:XAUUSD"

elif asset == "Silver":
    symbol = "SI=F"
    model_path = "silver_price_model.keras"
    widget_symbol = "OANDA:XAGUSD"

elif asset == "Crude Oil":
    symbol = "CL=F"
    model_path = "crude_oil_model.keras"
    widget_symbol = "NYMEX:CL1!"

# -------------------------
# Live Price Widget
# -------------------------
st.subheader(f"📡 Live {asset} Price")

st.components.v1.html(f"""
<div style="height:500px">
  <iframe src="https://s.tradingview.com/widgetembed/?symbol={widget_symbol}&interval=D&theme=light"
  width="100%" height="500" frameborder="0"></iframe>
</div>
""", height=500)

st.info("📌 Live price is real-time, predictions are based on LSTM model.")

# -------------------------
# Load Model
# -------------------------
try:
    model = load_model(model_path)
    st.success(f"✅ {asset} model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------
# Fetch Data
# -------------------------
data = yf.download(symbol, period="10y")

if data.empty:
    st.error("⚠️ No data found.")
    st.stop()

closing_price = data[['Close']]

# -------------------------
# Dataset Preview
# -------------------------
st.subheader("📊 Dataset Preview")
st.write(data.tail())

# -------------------------
# Historical Chart
# -------------------------
st.subheader(f"📈 {asset} Historical Price")

fig1 = plt.figure(figsize=(10,4))
plt.plot(closing_price.index, closing_price['Close'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{asset} Closing Price")
st.pyplot(fig1)

# -------------------------
# Scaling
# -------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(closing_price)

# -------------------------
# Prepare Data
# -------------------------
X, y = [], []
base_days = 100

for i in range(base_days, len(scaled_data)):
    X.append(scaled_data[i-base_days:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# -------------------------
# Train/Test Split
# -------------------------
train_size = int(len(X)*0.9)

X_test = X[train_size:]
y_test = y[train_size:]

# -------------------------
# Predictions
# -------------------------
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# -------------------------
# Actual vs Predicted
# -------------------------
st.subheader("📉 Actual vs Predicted")

fig2 = plt.figure(figsize=(12,5))
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Price")
st.pyplot(fig2)

# -------------------------
# Error Plot
# -------------------------
st.subheader("📉 Prediction Error")

error = y_test_actual.flatten() - predictions.flatten()

fig3 = plt.figure(figsize=(10,4))
plt.plot(error)
plt.title("Error Over Time")
st.pyplot(fig3)

# -------------------------
# Future Prediction
# -------------------------
st.subheader("🔮 Future Prediction")

days = st.slider("Select number of days", 1, 30, 10)

last_100 = scaled_data[-100:].reshape(1,100,1)
future_predictions = []

for _ in range(days):
    next_day = model.predict(last_100)
    future_predictions.append(next_day[0][0])
    last_100 = np.append(last_100[:,1:,:], next_day.reshape(1,1,1), axis=1)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

# -------------------------
# Future Table
# -------------------------
future_df = pd.DataFrame(future_predictions, columns=["Predicted Price"])
st.dataframe(future_df)

# -------------------------
# Future Chart
# -------------------------
fig4 = plt.figure(figsize=(10,4))
plt.plot(future_df["Predicted Price"], marker="o")
plt.title(f"{asset} Future Prediction")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
st.pyplot(fig4)

st.success("✅ Dashboard Ready!")