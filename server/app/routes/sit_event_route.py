from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from ..service.sit_event_service import (
    add_sit_event,
    get_daily_sit_event_records,
    get_weekly_sit_event_records
)
from ..utils.response_util import make_json_response
from ..exceptions import NotFoundError, DatabaseError

sit_event_router = APIRouter(tags=["久坐紀錄"])

class AddSitEventRequest(BaseModel):
    user_id: int
    start_at: str
    end_at: str

@sit_event_router.post("/sit-event")
async def create_sit_event(data: AddSitEventRequest):
    """
        新增久坐事件
    """
    try:
        record_id = await add_sit_event(data.user_id, data.start_at, data.end_at)
        return await make_json_response(data={"record_id": record_id}, message="久坐事件新增成功", code=200)
    except DatabaseError as e:
        return await make_json_response(success=False, message=str(e), code=500)
    except Exception as e:
        return await make_json_response(success=False, message=str(e), code=500)

@sit_event_router.get("/sit-event/daily")
async def get_daily_sit_events(user_id: int = Query(..., description="使用者 ID"), date: str = Query(..., description="日期 (格式: YYYY-MM-DD)")):
    """
        抓取指定日期的久坐事件
    """
    try:
        result = await get_daily_sit_event_records(user_id, date)
        return await make_json_response(data=result, message="成功取得當天久坐事件記錄", code=200)
    except NotFoundError as e:
        return await make_json_response(success=False, message=str(e), code=404)
    except Exception as e:
        return await make_json_response(success=False, message=str(e), code=500)

@sit_event_router.get("/sit-event/weekly")
async def get_weekly_sit_events(
    user_id: int = Query(..., description="使用者 ID"),
    date: str = Query(..., description="日期 (格式: YYYY-MM-DD)"),
    sunday_first: bool = Query(True, description="是否以星期日為一周的開始")
):
    """
        抓取指定日期所在週的久坐事件
    """
    try:
        result = await get_weekly_sit_event_records(user_id, date, sunday_first)
        return await make_json_response(data=result, message="成功取得一周久坐事件記錄", code=200)
    except NotFoundError as e:
        return await make_json_response(success=False, message=str(e), code=404)
    except Exception as e:
        return await make_json_response(success=False, message=str(e), code=500)