from datetime import datetime
from fastapi import APIRouter, Query
from ..service.weekly_interaction_reports_service import get_latest_report, get_reports_by_week
from ..service.weekly_report_push_service import push_weekly_reports_batch, push_weekly_report_to_elder, push_weekly_reports_by_time
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

@weekly_reports_router.get("/weekly-report/by-week")
async def get_weekly_report_by_date(
    elder_id: int = Query(..., description="長者 ID"),
    date: str = Query(..., description="查詢日期 (格式: YYYY-MM-DD)"),
    sunday_as_first_day: bool = Query(False, description="是否以星期日為一週的第一天")
):
    """
        取得指定 elder_id 和日期所在週的每週互動報告
    """
    try:
        reports = await get_reports_by_week(elder_id, date, sunday_as_first_day)
        return await make_json_response(data=reports, message="成功取得該週報告")
    except ValueError as e:
        return await make_json_response(code=400, message=str(e), success=False)
    except NotFoundError as e:
        return await make_json_response(code=404, message=str(e), success=False)
    except DatabaseError as e:
        return await make_json_response(code=500, message=str(e), success=False)
    except Exception as e:
        return await make_json_response(code=500, message=f"未知錯誤: {e}", success=False)

@weekly_reports_router.post("/weekly-report/push-by-time")
async def push_weekly_reports_by_schedule(
    hour: int = Query(None, description="推播小時 (0-23)，不指定則使用當前時間"),
    minute: int = Query(None, description="推播分鐘 (0-59)，不指定則使用當前時間"), 
    weekday: int = Query(None, description="星期 (0=週日, 1=週一...6=週六)，不指定則使用當前星期")
):
    """
        根據用戶推播時間設定，推播週報給符合時間的用戶
        適合用於n8n定時觸發 (建議每小時或每30分鐘執行一次)
    """
    try:
        result = await push_weekly_reports_by_time(hour, minute, weekday)
        
        if result["status"] == "completed":
            return await make_json_response(
                data=result,
                message=f"定時推播完成：{result['success_count']}/{result['total_elders']} 成功"
            )
        elif result["status"] == "no_users":
            return await make_json_response(
                data=result,
                message=result["message"],
                code=200
            )
        else:
            return await make_json_response(
                data=result,
                message=result.get("message", "推播失敗"),
                success=False,
                code=500
            )
    except Exception as e:
        return await make_json_response(
            code=500,
            message=f"定時推播週報時發生錯誤: {e}",
            success=False
        )

@weekly_reports_router.post("/weekly-report/push-all")
async def push_all_weekly_reports():
    """
        自動推播所有長者的週報給照護者
        適合用於n8n定時觸發
    """
    try:
        result = await push_weekly_reports_batch()
        
        if result["status"] == "completed":
            return await make_json_response(
                data=result, 
                message=f"批量推播完成：{result['success_count']}/{result['total_elders']} 成功"
            )
        elif result["status"] == "no_data":
            return await make_json_response(
                data=result,
                message="沒有需要推播的週報",
                code=200
            )
        else:
            return await make_json_response(
                data=result,
                message=result.get("message", "推播失敗"),
                success=False,
                code=500
            )
            
    except Exception as e:
        return await make_json_response(
            code=500, 
            message=f"推播週報時發生錯誤: {e}", 
            success=False
        )

@weekly_reports_router.post("/weekly-report/push/{elder_id}")
async def push_weekly_report_by_elder(elder_id: int):
    """
        推播指定長者的週報給其照護者
    """
    try:
        result = await push_weekly_report_to_elder(elder_id)
        
        if result["status"] == "success":
            return await make_json_response(
                data=result,
                message=f"週報推播成功，發送給 {result['recipients_count']} 位照護者"
            )
        elif result["status"] == "no_recipients":
            return await make_json_response(
                data=result,
                message="該長者沒有綁定的照護者",
                code=404
            )
        elif result["status"] == "no_report":
            return await make_json_response(
                data=result,
                message="找不到該長者的最新週報",
                code=404
            )
        else:
            return await make_json_response(
                data=result,
                message=result.get("message", "推播失敗"),
                success=False,
                code=500
            )
            
    except Exception as e:
        return await make_json_response(
            code=500,
            message=f"推播週報時發生錯誤: {e}",
            success=False
        )