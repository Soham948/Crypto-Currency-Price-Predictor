import streamlit as st

st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="💰", layout="wide")

st.title("💰 Bitcoin Price Predictor using LSTM")
st.markdown("""
Welcome to the **Bitcoin Price Prediction App**!  
This tool uses a trained **LSTM (Long Short-Term Memory)** model to forecast Bitcoin closing prices.

### 🔍 App Sections:
1. **📊 Data Visualization** — View live Bitcoin historical data and charts.  
2. **🤖 Model Prediction** — Predict Bitcoin’s future prices using our LSTM model.  
3. **ℹ️ About** — Learn more about how this project works.
""")

st.info("👉 Use the left sidebar to navigate between pages.")
