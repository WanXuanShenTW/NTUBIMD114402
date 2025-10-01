from ..dao.sleep_records_dao import (
    get_sleep_time_by_date,
    get_sleep_time_by_week
)
from ..db import Database
from ..exceptions import NotFoundError, DatabaseError

async def get_daily_sleep_record(user_id: int, date: str) -> dict:
    """
    抓取指定日期的睡眠記錄。

    :param user_id: 使用者 ID
    :param date: 日期 (格式: YYYY-MM-DD)
    :return: 當天的睡眠記錄
    """
    async with Database.connection() as conn:
        try:
            record = await get_sleep_time_by_date(conn, user_id, date)
            if not record:
                raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 的睡眠記錄")
            return record
        except NotFoundError:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 的睡眠記錄")
        except Exception as e:
            raise DatabaseError(f"查詢睡眠記錄時發生錯誤: {e}")

async def get_weekly_sleep_records(user_id: int, date: str, sunday_first: bool = True) -> list:
    """
    抓取指定日期所在週的睡眠記錄。

    :param user_id: 使用者 ID
    :param date: 日期 (格式: YYYY-MM-DD)
    :param sunday_first: 是否以周日作為一周的第一天 (True: 周日, False: 周一)
    :return: 一周的睡眠記錄
    """
    async with Database.connection() as conn:
        try:
            records = await get_sleep_time_by_week(conn, user_id, date, sunday_first)
            if not records:
                raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 起始的一周的睡眠記錄")
            return records
        except NotFoundError:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 起始的一周的睡眠記錄")
        except Exception as e:
            raise DatabaseError(f"查詢一周睡眠記錄時發生錯誤: {e}")