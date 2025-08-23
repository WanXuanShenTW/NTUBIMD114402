import os
import shutil
import tempfile
import datetime
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List
from ..service.fall_event_service import (
    get_fall_event_video_filename_by_record_id,
    get_video_filename_for_fall_event,
    add_fall_event_with_video
)

fall_router = APIRouter()

TEMP_VIDEO_DIR = "sources/tmp"
VIDEOS_DIR = "sources/fall_videos"

# 清空 tmp 資料夾
def clear_temp_directory():
    if os.path.exists(TEMP_VIDEO_DIR):
        shutil.rmtree(TEMP_VIDEO_DIR)  # 刪除整個資料夾
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)  # 重新建立空的資料夾

# 在應用程式啟動時清空暫存區
clear_temp_directory()

@fall_router.post("/fall_video")
async def detect_fall_video(
    user_id: str = Form(..., description="使用者 ID"),
    video: UploadFile = Form(..., description="上傳的 MP4 影片檔案")
):
    """
    處理使用者上傳的影片，進行跌倒偵測與骨架點提取。
    """
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="檔案格式不符，請上傳 MP4 影片")

    try:
        # 將上傳的影片存成 temporary 檔
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=TEMP_VIDEO_DIR) as tmp:
            video_path = tmp.name
            tmp.write(await video.read())

        # 取得影片的 FPS 與幀數
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="影片讀取失敗")
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 模擬處理邏輯（實際邏輯應該調用服務層函數）
        prediction_result = "fall" if frame_count > 100 else "non-fall"

        # 儲存影片檔案
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        video_filename = f"{user_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        video_path = os.path.join(VIDEOS_DIR, video_filename)
        shutil.move(tmp.name, video_path)

        # 儲存到資料庫
        await add_fall_event_with_video(
            user_id=user_id,
            location="客廳",
            pose_before_fall="走路中",
            video_filename=video_filename
        )

        return {"id": user_id, "result": prediction_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@fall_router.get("/fall_videos_data")
async def get_merged_fall_videos(
    elder_id: int = Query(..., description="長者的使用者 ID"),
    caregiver_id: int = Query(..., description="照護者的使用者 ID"),
    start_date: Optional[str] = Query(None, description="起始日期 (yyyy-mm-dd)"),
    end_date: Optional[str] = Query(None, description="結束日期 (yyyy-mm-dd)"),
    limit: int = Query(5, description="返回的最大筆數 (1-5)")
):
    """
    查詢合併後的跌倒影片資料（只回傳資料庫資料，不傳送影片檔案）。
    """
    try:
        # 處理時間區間
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1) if end_date else None
        limit = max(1, min(limit, 5))

        data_list = await get_video_filename_for_fall_event(elder_id, caregiver_id, start, end, limit)
        for row in data_list:
            if isinstance(row["detected_time"], datetime.datetime):
                row["detected_time"] = row["detected_time"].strftime("%Y-%m-%d %H:%M:%S")
        return data_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@fall_router.get("/fall_video_file")
async def get_merged_fall_video_file(record_id: int = Query(..., description="資料庫中的 record_id")):
    """
    取得合併後的跌倒影片檔案。
    """
    try:
        video_filename = await get_fall_event_video_filename_by_record_id(record_id)
        if not video_filename:
            raise HTTPException(status_code=404, detail="找不到影片")

        video_path = os.path.join(VIDEOS_DIR, video_filename)
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="影片檔案不存在於伺服器")

        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=video_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")