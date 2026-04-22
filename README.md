🚀 Features
📡 Real-time data fetching using yfinance
🧠 LSTM-based deep learning model for time-series prediction
📈 Supports multiple assets:
Bitcoin (BTC)
Ethereum (ETH)
Gold
Silver
Crude Oil



📊 Visualization of historical and predicted trends
🌐 Interactive web interface using Streamlit
🔄 Dynamic asset selection for prediction
🛠️ Tech Stack
Frontend & Backend: Streamlit
Language: Python
Libraries:
NumPy
Pandas
Matplotlib
Scikit-learn
TensorFlow / Keras
yfinance




📂 Project Structure
Multi-Asset-Predictor/
│
├── app.py                  # Streamlit app
├── model.h5               # Trained LSTM model
├── data/                  # Stored datasets (optional)
├── notebooks/             # EDA & model training
├── utils.py               # Preprocessing & helper functions
├── requirements.txt       # Dependencies
└── README.md              # Documentation




⚙️ How It Works
Historical data for selected assets is fetched using yfinance
Data is cleaned and normalized for training
Time-series sequences are created
LSTM model learns patterns from historical price movements
Model predicts future prices
Results are displayed with graphs in Streamlit



📊 Model Details
Model Type: LSTM (Recurrent Neural Network)
Input: Historical price sequences
Output: Future price prediction
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam




💻 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/multi-asset-price-predictor.git
cd multi-asset-price-predictor
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run app.py

🎯 Use Cases
Financial market trend analysis
Cryptocurrency and commodity forecasting
Learning deep learning for time-series data
Portfolio project for data science
⚠️ Disclaimer

This project is for educational purposes only and should not be used for real-world trading or financial decisions.

🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.
