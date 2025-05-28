import os
import pickle
from tensorflow import keras

# 模型與 Scaler 的路徑
MODEL_PATH = "models/lstm_fall_detection_model_N01.h5"
SCALER_PATH = "models/scaler_N01.pkl"

# 載入模型與 Scaler
def load_fall_model():
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler