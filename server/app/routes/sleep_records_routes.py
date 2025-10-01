from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from ..service.sleep_records_service import (
    get_daily_sleep_record,
    get_weekly_sleep_records
)
from ..utils.response_util import make_json_response
from ..exceptions import NotFoundError, DatabaseError

sleep_records_router = APIRouter(tags=["睡眠記錄"])

class WeeklySleepRequest(BaseModel):
    user_id: int
    date: str
    sunday_first: bool = True

@sleep_records_router.get("/sleep/daily")
async def get_daily_sleep(user_id: int = Query(..., description="使用者 ID"), date: str = Query(..., description="日期 (格式: YYYY-MM-DD)")):
    """
        抓取指定日期的睡眠記錄
    """
    try:
        record = await get_daily_sleep_record(user_id, date)
        return await make_json_response(data=record, message="成功取得當天睡眠記錄")
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@sleep_records_router.get("/sleep/weekly")
async def get_weekly_sleep(data: WeeklySleepRequest):
    """
        抓取指定日期所在週的睡眠記錄
    """
    try:
        records = await get_weekly_sleep_records(data.user_id, data.date, data.sunday_first)
        return await make_json_response(data=records, message="成功取得一周睡眠記錄")
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))