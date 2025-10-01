from ..dao.weekly_interaction_reports_dao import select_latest_report_by_elder_id
from ..db import Database
from ..exceptions import NotFoundError, DatabaseError

async def get_latest_report(elder_id: int) -> dict:
    """
    取得指定 elder_id 的最新 weekly_interaction_report。

    :param elder_id: 長者 ID
    :return: 最新的報告資料
    """
    async with Database.connection() as conn:
        try:
            report = await select_latest_report_by_elder_id(conn, elder_id)
            if not report:
                raise NotFoundError(f"找不到 elder_id={elder_id} 的最新報告")
            if "analysis_result" in report and report["analysis_result"].startswith("報告：\n\n"):
                report["analysis_result"] = report["analysis_result"][len("報告：\n\n"):]
            return report
        except NotFoundError:
            raise NotFoundError(f"找不到 elder_id={elder_id} 的最新報告")
        except Exception as e:
            raise DatabaseError(f"查詢最新報告時發生錯誤: {e}")