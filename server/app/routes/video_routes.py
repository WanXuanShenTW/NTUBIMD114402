from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
from ..db import Database

video_router = APIRouter()

@video_router.get("/video-events")
async def get_video_events(
    user_id: int = Query(..., description="使用者 ID"),
    start_date: str = Query(..., description="起始日期 (yyyy-mm-dd)"),
    end_date: str = Query(..., description="結束日期 (yyyy-mm-dd)")
):
    """
    查詢使用者的影片事件。
    """
    try:
        # 解析日期
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        # 查詢資料庫
        async with Database.connection() as conn:
            async with conn.cursor() as cursor:
                query = """
                    SELECT 'fall' AS event_type, user_id, detected_time AS start_time, video_filename
                    FROM fall_events
                    WHERE user_id = %s AND video_filename IS NOT NULL AND detected_time BETWEEN %s AND %s
                    UNION ALL
                    SELECT 'leave_bed' AS event_type, user_id, leave_time AS start_time, video_filename
                    FROM leave_bed_events
                    WHERE user_id = %s AND is_abnormal = 1 AND video_filename IS NOT NULL AND leave_time BETWEEN %s AND %s
                    ORDER BY start_time DESC
                """
                values = (user_id, start, end, user_id, start, end)
                await cursor.execute(query, values)
                rows = await cursor.fetchall()

        # 格式化結果
        results = [
            {
                "event_type": row[0],
                "user_id": row[1],
                "start_time": row[2].strftime("%Y-%m-%d %H:%M:%S"),
                "video_filename": row[3]
            }
            for row in rows
        ]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")