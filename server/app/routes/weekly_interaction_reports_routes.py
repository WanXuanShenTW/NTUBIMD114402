from fastapi import APIRouter, Query
from ..service.weekly_interaction_reports_service import get_latest_report
from ..utils.response_util import make_json_response
from ..exceptions import NotFoundError, DatabaseError

weekly_reports_router = APIRouter(tags=["每週互動報告"])

@weekly_reports_router.get("/weekly-report/latest")
async def get_latest_weekly_report(elder_id: int = Query(..., description="長者 ID")):
    """
        取得指定 elder_id 的最新每週互動報告
    """
    try:
        report = await get_latest_report(elder_id)
        return await make_json_response(data=report, message="成功取得最新報告")
    except NotFoundError as e:
        return await make_json_response(code=404, message=str(e), success=False)
    except DatabaseError as e:
        return await make_json_response(code=500, message=str(e), success=False)
    except Exception as e:
        return await make_json_response(code=500, message=f"未知錯誤: {e}", success=False)