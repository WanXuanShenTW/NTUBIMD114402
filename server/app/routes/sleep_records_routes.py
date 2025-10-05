from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from ..service.sleep_records_service import (
    add_sleep_record,
    get_daily_sleep_record,
    get_weekly_sleep_records
)
from ..utils.response_util import make_json_response
from ..exceptions import NotFoundError, DatabaseError

sleep_records_router = APIRouter(tags=["睡眠記錄"])

class AddSleepRecordRequest(BaseModel):
    user_id: int
    sleep_time: str
    wake_time: str

@sleep_records_router.post("/sleep")
async def create_sleep_record(data: AddSleepRecordRequest):
    """
        新增睡眠紀錄
    """
    try:
        record_id = await add_sleep_record(data.user_id, data.sleep_time, data.wake_time)
        return await make_json_response(data={"record_id": record_id}, message="睡眠紀錄新增成功")
    except DatabaseError as e:
        return await make_json_response(success=False, message=str(e), status_code=500)
    except Exception as e:
        return await make_json_response(success=False, message=str(e), status_code=500)

@sleep_records_router.get("/sleep/daily")
async def get_daily_sleep(user_id: int = Query(..., description="使用者 ID"), date: str = Query(..., description="日期 (格式: YYYY-MM-DD)")):
    """
        抓取指定日期的睡眠記錄
    """
    try:
        record = await get_daily_sleep_record(user_id, date)
        if not record:
            return await make_json_response(data={}, message="尚為空值", success=True, code=200)
        return await make_json_response(data=record, message="成功取得當天睡眠記錄", code=200)
    except NotFoundError:
        return await make_json_response(data={}, message="尚為空值", success=True, code=200)
    except Exception as e:
        return await make_json_response(success=False, message=str(e), code=500)

@sleep_records_router.get("/sleep/weekly")
async def get_weekly_sleep(
    user_id: int = Query(..., description="使用者 ID"),
    date: str = Query(..., description="日期 (格式: YYYY-MM-DD)"),
    sunday_first: bool = Query(True, description="是否以星期日為一周的開始")
):
    """
        抓取指定日期所在週的睡眠記錄
    """
    try:
        result = await get_weekly_sleep_records(user_id, date, sunday_first)
        if not result:
            return await make_json_response(data=[], message="尚為空值", success=True, code=200)
        return await make_json_response(data=result, message="成功取得一周睡眠記錄", code=200)
    except NotFoundError:
        return await make_json_response(data=[], message="尚為空值", success=True, code=200)
    except Exception as e:
        return await make_json_response(success=False, message=str(e), code=500)