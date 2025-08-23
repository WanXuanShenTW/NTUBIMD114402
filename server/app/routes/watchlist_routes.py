from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional
from ..service.video_watchlist_service import (
    add_video_to_watchlist,
    remove_video_from_watchlist,
    get_watchlist_video_data_by_id
)

watchlist_router = APIRouter()

class AddToWatchlistRequest(BaseModel):
    user_id: int
    record_id: int
    video_type: str

class DeleteFromWatchlistRequest(BaseModel):
    user_id: int
    record_id: int
    video_type: str

@watchlist_router.get("/watchlist")
async def get_watchlist_data(
    user_id: int = Query(..., description="使用者 ID"),
    video_type: str = Query(..., description="影片類型")
):
    """
    取得使用者的觀看清單資料。
    """
    try:
        watchlist_data = await get_watchlist_video_data_by_id(user_id, video_type)
        if not watchlist_data:
            return {"message": "沒有找到任何收藏"}
        return watchlist_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@watchlist_router.post("/watchlist")
async def add_to_watchlist(data: AddToWatchlistRequest):
    """
    新增影片到觀看清單。
    """
    try:
        await add_video_to_watchlist(data.record_id, data.user_id, data.video_type)
        return {
            "message": "新增收藏成功",
            "user_id": data.user_id,
            "record_id": data.record_id,
            "video_type": data.video_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@watchlist_router.delete("/watchlist")
async def delete_from_watchlist(data: DeleteFromWatchlistRequest):
    """
    從觀看清單中刪除影片。
    """
    try:
        success = await remove_video_from_watchlist(data.user_id, data.record_id, data.video_type)
        if success:
            return {
                "message": "刪除收藏成功",
                "user_id": data.user_id,
                "record_id": data.record_id,
                "video_type": data.video_type
            }
        else:
            raise HTTPException(status_code=404, detail="找不到符合條件的收藏")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")