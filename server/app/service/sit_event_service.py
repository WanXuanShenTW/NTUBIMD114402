from datetime import datetime, timedelta
from ..dao.sit_event_dao import (
    insert_sit_event,
    get_sit_events_by_date,
    get_sit_events_by_week
)
from ..db import Database
from ..exceptions import NotFoundError, DatabaseError

async def add_sit_event(user_id: int, start_at: str, end_at: str) -> int:
    """
    新增一筆久坐事件資料到資料庫
    """
    async with Database.connection() as conn:
        try:
            record_id = await insert_sit_event(conn, user_id, start_at, end_at)
            if record_id is None:
                raise DatabaseError("資料庫新增久坐事件失敗")
            return record_id
        except Exception as e:
            raise DatabaseError(f"新增久坐事件時發生錯誤: {e}")

async def get_daily_sit_event_records(user_id: int, date: str) -> dict:
    """
    抓取指定日期的久坐事件記錄
    """
    async with Database.connection() as conn:
        try:
            records = await get_sit_events_by_date(conn, user_id, date)
            return {
                "date": date,
                "records": records
            }
        except NotFoundError:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 的久坐事件記錄")
        except Exception as e:
            raise DatabaseError(f"查詢某天久坐事件記錄時發生錯誤: {e}")
        
async def get_weekly_sit_event_records(user_id: int, date: str, sunday_first: bool = True) -> dict:
    """
    抓取指定日期所在週的久坐事件記錄
    """
    try:
        # 計算一周的起始和結束日期
        input_date = datetime.strptime(date, "%Y-%m-%d")
        weekday = input_date.weekday()  # 0: 周一, 6: 周日
        if sunday_first:
            start_date = input_date - timedelta(days=(weekday + 1) % 7)  # 周日為第一天
        else:
            start_date = input_date - timedelta(days=weekday)  # 周一為第一天

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (start_date + timedelta(days=7)).strftime("%Y-%m-%d")

        async with Database.connection() as conn:
            records = await get_sit_events_by_week(conn, user_id, start_date_str, end_date_str)
            return {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "records": records
            }
    except NotFoundError:
        raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 所在一周的久坐事件記錄")
    except Exception as e:
        raise DatabaseError(f"查詢一周久坐事件記錄時發生錯誤: {e}")