import os
import pickle
from tensorflow import keras

# 模型與 Scaler 的路徑
MODEL_PATH = "models/fall_detection_modelN01.keras"
SCALER_PATH = "models/scaler.pkl"

# 載入模型與 Scaler
def load_fall_model():
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler