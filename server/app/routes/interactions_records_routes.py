from fastapi import APIRouter, Query
from ..utils.response_util import make_json_response
from ..exceptions import DatabaseError
from ..service.interactions_records_service import get_interactions_by_day

interactions_router = APIRouter(tags=["互動紀錄"])

@interactions_router.get("/interactions/day")
async def get_interactions_day(
    elder_id: int = Query(..., description="長者ID"),
    date: str = Query(..., description="查詢日期(YYYY-MM-DD)"),
):
    """
    依 elder_id + date(天) 取得當天互動紀錄（Q/A 已清理）
    回傳：
    {
      elder_id, date, total_sessions, total_pairs,
      sessions: [
        { session_id, start_at, pairs: [ {Q, A}, ... ] },
        ...
      ]
    }
    """
    try:
        data = await get_interactions_by_day(elder_id, date)
        return await make_json_response(data=data, message="查詢成功", code=200)
    except DatabaseError as e:
        return await make_json_response(code=500, message=str(e), success=False)
    except Exception as e:
        return await make_json_response(code=500, message=f"未知錯誤: {e}", success=False)
