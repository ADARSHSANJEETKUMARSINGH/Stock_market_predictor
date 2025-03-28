import joblib
import numpy as np

# Model aur Scaler Load Karna
model = joblib.load("stock_price_predictor.pkl")
scaler = joblib.load("scaler.pkl")

# 📌 Scaler aur Model ke Expected Features Count Check Kar
print("Scaler requires", scaler.n_features_in_, "features")
print("Model requires", model.n_features_in_, "features")

# ✅ Sahi Number of Features Ka Example Input
example_input = np.array([[125.5, 300000, 0.85, 0.02, 1.2, 0.5, -0.3, 2000000]])  # 🛑 Yeh 8 Features Hona Chahiye
example_input_scaled = scaler.transform(example_input)  # Scaling Apply Karna

# 📌 Prediction
prediction = model.predict(example_input_scaled)
print("Predicted Stock Price:", prediction[0])
