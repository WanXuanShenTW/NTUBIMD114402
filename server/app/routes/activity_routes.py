from fastapi import APIRouter, Query
from ..utils.response_util import make_json_response
from ..service.activity_service import get_daily_activity, get_weekly_activity

activity_router = APIRouter(prefix="", tags=["活動量"])

@activity_router.get("/activity/daily")
async def activity_daily(
    user_id: int = Query(..., description="User ID"),
    date: str = Query(..., description="YYYY-MM-DD")
):
    try:
        data = await get_daily_activity(user_id, date)
        return await make_json_response(code=200, message="成功讀取當天活動量", data=data)
    except Exception as e:
        return await make_json_response(code=500, success=False, message=str(e))

@activity_router.get("/activity/weekly")
async def activity_weekly(
    user_id: int = Query(..., description="User ID"),
    date: str = Query(..., description="任一天：YYYY-MM-DD（用來定位該週）"),
    week_start: str = Query("mon", description="週起始：'sun' 或 'mon'")
):
    """
    以輸入日期定位該週；week_start='sun' 代表週日為第一天，'mon' 代表週一為第一天。
    """
    try:
        ws = (week_start or "mon").lower()
        if ws not in ("sun", "mon"):
            ws = "mon"
        data = await get_weekly_activity(user_id, date, ws)
        return await make_json_response(code=200, message="成功讀取當週活動量", data=data)
    except Exception as e:
        return await make_json_response(code=500, success=False, message=str(e))
