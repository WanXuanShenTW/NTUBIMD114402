import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from ..db import Database

gait_router = APIRouter()
GAIT_VIDEO_FOLDER = os.path.join("static", "gait_instability_videos")
os.makedirs(GAIT_VIDEO_FOLDER, exist_ok=True)

@gait_router.post("/add_gait_instability")
async def add_gait_instability(
    user_id: str = Form(..., description="使用者 ID"),
    detected_time: str = Form(..., description="偵測時間 (yyyy-MM-dd HH:mm:ss)"),
    video: UploadFile = Form(..., description="上傳的 MP4 影片檔案")
):
    """
    新增步態不穩事件，包含影片上傳與資料庫儲存。
    """
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="檔案格式不符，請上傳 MP4 影片")

    try:
        # 解析偵測時間
        detected_time_obj = datetime.strptime(detected_time, "%Y-%m-%d %H:%M:%S")
        timestamp = detected_time_obj.strftime('%Y%m%d_%H%M%S')
        filename = f"user{user_id}_{timestamp}.mp4"
        save_path = os.path.join(GAIT_VIDEO_FOLDER, filename)

        # 儲存影片檔案
        with open(save_path, "wb") as f:
            f.write(await video.read())

        # 儲存資料到資料庫
        async with Database.connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    INSERT INTO gait_instability_records (user_id, detected_time, video_filename)
                    VALUES (%s, %s, %s)
                """
                await cur.execute(query, (user_id, detected_time_obj, filename))
                await conn.commit()

        return JSONResponse(
            content={"message": "步態不穩事件已新增成功", "video_filename": filename},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")