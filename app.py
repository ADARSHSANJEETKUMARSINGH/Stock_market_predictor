from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 📌 Model & Scaler Load Karna
model = joblib.load("stock_price_predictor.pkl")  # Model file load karna
scaler = joblib.load("scaler.pkl")  # Agar scaler ko bhi save kiya hai

@app.route('/')
def home():
    return "Stock Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # JSON Input Le Raha Hai
    features = np.array(data["features"]).reshape(1, -1)  # Data ko Numpy Array me Convert Karna
    features_scaled = scaler.transform(features)  # Normalize Karna
    prediction = model.predict(features_scaled)  # Prediction Lena
    
    return jsonify({"predicted_price": prediction[0]})  # JSON Response

if __name__ == "__main__":
    app.run(debug=True)  # Local Server Run Karna
