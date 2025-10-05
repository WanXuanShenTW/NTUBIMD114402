from datetime import datetime
from ..db import Database
from ..exceptions import NotFoundError, DatabaseError

from ..dao.weekly_interaction_reports_dao import select_reports_by_week
from ..dao.weekly_interaction_reports_dao import select_latest_report_by_elder_id

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
        
async def get_reports_by_week(elder_id: int, date: str, sunday_as_first_day: bool) -> list[dict]:
    """
    取得指定 elder_id 和日期所在週的 weekly_interaction_report。

    :param elder_id: 長者 ID
    :param date: 日期 (格式: YYYY-MM-DD)
    :param sunday_as_first_day: 是否以星期日為一週的第一天
    :return: 該週的報告資料
    """
    try:
        query_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("日期格式錯誤，請使用 YYYY-MM-DD 格式")

    async with Database.connection() as conn:
        try:
            reports = await select_reports_by_week(conn, elder_id, query_date, sunday_as_first_day)
            return reports
        except NotFoundError:
            raise NotFoundError(f"找不到 elder_id={elder_id} 在 {date} 所在週的報告")
        except Exception as e:
            raise DatabaseError(f"查詢該週報告時發生錯誤: {e}")