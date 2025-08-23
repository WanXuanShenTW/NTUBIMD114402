import numpy as np
import asyncio

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