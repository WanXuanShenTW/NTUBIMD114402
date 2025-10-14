import json
import cv2
import numpy as np
import time

# ============= 設定區 =============
JSON_PATH = "data/ws_frames/user_3/202510141147.json"  # ← 你的 JSON 檔案路徑
FPS = 10                      # ← 播放幀率，可自由修改
CANVAS_SIZE = (640, 480)      # 視窗大小 (w, h)
POINT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 255, 0)
THICKNESS = 2

# ============= YOLOv8 Pose 連線關係 (COCO 17點) =============
POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# ============= 載入資料 =============
with open(JSON_PATH, "r", encoding="utf-8") as f:
    frames = json.load(f)

if not isinstance(frames, list):
    raise ValueError("❌ JSON 格式錯誤，應為骨架幀列表 (list)。")

print(f"✅ 載入 {len(frames)} 幀骨架資料，開始以 {FPS} FPS 播放。")

frame_interval = 1.0 / FPS

# ============= 播放骨架動畫 =============
for i, frame in enumerate(frames):
    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)

    keypoints_list = frame.get("keypoints", [])
    if not keypoints_list:
        continue

    for person_kps in keypoints_list:
        # YOLOv8 格式：[ [x, y], [x, y], ... ] 共17點
        kps = [(int(x), int(y)) for x, y in person_kps if x > 0 and y > 0]

        # 畫連線
        for (a, b) in POSE_PAIRS:
            if a < len(kps) and b < len(kps):
                cv2.line(canvas, kps[a], kps[b], LINE_COLOR, THICKNESS)

        # 畫關節點
        for (x, y) in kps:
            cv2.circle(canvas, (x, y), 3, POINT_COLOR, -1)

    cv2.putText(canvas, f"Frame {i+1}/{len(frames)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Skeleton Player", canvas)

    key = cv2.waitKey(int(frame_interval * 1000)) & 0xFF
    if key == 27:  # Esc 結束
        break

cv2.destroyAllWindows()
print("🏁 播放結束")
