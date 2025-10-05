from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from datetime import datetime

from app.exceptions import NotFoundError

from ..service.fall_event_service import (
    get_fall_event_records_by_time_range
)
from ..utils.response_util import make_json_response

fall_event_router = APIRouter(tags=["跌倒事件"])

@fall_event_router.get("/fall_event/records")
async def get_fall_records(
    user_id: int = Query(..., description="使用者ID"),
    start_time: datetime = Query(None, description="開始時間 (格式: YYYY-MM-DD HH:MM:SS)"),
    end_time: datetime = Query(None, description="結束時間 (格式: YYYY-MM-DD HH:MM:SS)")
):
    """
        查詢跌倒事件紀錄，支援時間區段篩選
    """
    try:
        records = await get_fall_event_records_by_time_range(user_id, start_time, end_time)
        return await make_json_response(code=200, data={"records": records}, message="查詢成功")
    except NotFoundError as ne:
        return await make_json_response(code=200, message="尚為空值", success=True)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)