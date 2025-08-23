import cv2
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import asyncio

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 載入環境變數
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_skeleton_points(image):
    """
    從影像中提取骨架點（非同步版本）。
    """
    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except Exception as e:
            raise ValueError("無法轉換影像為 numpy.ndarray") from e

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("影像必須為 3-channel 的 numpy.ndarray")

    # 將影像轉換為 RGB 格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, pose.process, rgb_image)
    except Exception as e:
        raise RuntimeError(f"骨架解析失敗：{e}")

    if results.pose_landmarks:
        return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
    return []

async def normalize_skeleton_data(skeleton_data, scaler, time_steps=120):
    """
    將骨架數據標準化並轉換為模型輸入格式（非同步版本）。
    """
    if len(skeleton_data) < time_steps:
        padding = [[(0, 0, 0)] * 33] * (time_steps - len(skeleton_data))
        skeleton_data = padding + skeleton_data
    else:
        skeleton_data = skeleton_data[-time_steps:]

    accel = []
    for i in range(1, len(skeleton_data)):
        frame_accel = [((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                       for p1, p2 in zip(skeleton_data[i - 1], skeleton_data[i])]
        accel.append(frame_accel)

    if len(accel) < time_steps:
        accel = [[0] * 33] * (time_steps - len(accel)) + accel

    combo = []
    for i in range(time_steps):
        skeleton_frame = np.array(skeleton_data[i]).flatten()
        accel_frame = np.array(accel[i])
        combo.append(np.hstack((skeleton_frame, accel_frame)))

    combo = np.array(combo).reshape(1, time_steps, -1)
    flat = combo.reshape(1, -1)

    loop = asyncio.get_event_loop()
    scaled = await loop.run_in_executor(None, scaler.transform, flat)
    return scaled.reshape(1, time_steps, -1)

async def generate_daily_summary():
    """
    生成每日新聞摘要（非同步版本）。
    """
    prompt = (
        "爺爺奶奶，早安。以下是今日三則適合台灣長者收聽的重要新聞。\n"
        "請用繁體中文清楚易懂地描述，避免艱深詞彙，主題可包含天氣、健康、食安、生活資訊等。\n"
        "每則新聞請用一段落呈現，段落間空一行，適合轉為語音播放。\n"
        "不需要列出編號，例如 '1.', '2.' 等。\n"
        "也不需要過度強調長者身份，可以自然稱呼為『您』。"
    )

    try:
        response = await client.chat.completions.acreate(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一位台灣新聞播報員，善於口語化講解新聞內容給長輩聽。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        raise RuntimeError(f"生成新聞摘要失敗：{e}")