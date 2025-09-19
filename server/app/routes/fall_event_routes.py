from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from app.exceptions import NotFoundError

from ..service.fall_event_service import (
    get_fall_event_records
)
from ..utils.response_util import make_json_response

fall_event_router = APIRouter(tags=["跌倒事件"])

@fall_event_router.get("/fall_event/records")
async def get_fall_records(request: Request, user_id: str = Query(..., description="使用者ID")):
    """
        查詢跌倒事件紀錄
    """
    # 檢查是否只有 user_id 參數
    if len(request.query_params) != 1 or "user_id" not in request.query_params:
        return await make_json_response(code=400, message="參數錯誤", success=False)
    try:
        records = await get_fall_event_records(user_id)
        return await make_json_response(data={"records": records}, message="查詢成功")
    except NotFoundError as ne:
        return await make_json_response(code=404, message=str(ne), success=False)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)