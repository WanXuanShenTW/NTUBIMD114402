import numpy as np

class KalmanFilter:
    def __init__(self):
        # 初始化卡爾曼濾波器的狀態
        self.state = None  # 狀態向量
        self.covariance = None  # 協方差矩陣
        self.process_noise = np.eye(34) * 0.01  # 過程噪聲（假設 17 個關鍵點，每個點有 x, y）
        self.measurement_noise = np.eye(34) * 0.1  # 測量噪聲
        self.transition_matrix = np.eye(34)  # 狀態轉移矩陣
        self.measurement_matrix = np.eye(34)  # 測量矩陣

    def predict(self):
        # 預測下一狀態
        if self.state is None:
            return None
        self.state = self.transition_matrix @ self.state
        self.covariance = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.process_noise
        return self.state

    def update(self, measurement):
        # 更新狀態
        if self.state is None:
            # 如果是第一次更新，直接初始化狀態
            self.state = np.array(measurement).flatten()
            self.covariance = np.eye(34)
            return self.state

        # 計算卡爾曼增益
        S = self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + self.measurement_noise
        K = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(S)

        # 更新狀態和協方差
        measurement = np.array(measurement).flatten()
        self.state = self.state + K @ (measurement - self.measurement_matrix @ self.state)
        self.covariance = (np.eye(34) - K @ self.measurement_matrix) @ self.covariance
        return self.state

    def predict_and_update(self, measurement):
    # 預測並更新
        if self.state is None:
            # 如果是第一次更新，直接初始化狀態
            self.state = np.array(measurement).flatten()
            self.covariance = np.eye(34)
            return self.state

        try:
            self.predict()
            return self.update(measurement)
        except Exception as e:
            print(f"Error in Kalman filter update: {e}")
            return self.state  # 若出現錯誤，直接返回當前狀態
