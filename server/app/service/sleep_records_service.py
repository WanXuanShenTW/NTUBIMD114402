from datetime import datetime, timedelta
from ..dao.sleep_records_dao import (
    get_sleep_time_by_date,
    get_sleep_time_by_week,
    insert_sleep_record
)
from ..db import Database
from ..exceptions import NotFoundError, DatabaseError

async def add_sleep_record(user_id: int, sleep_time: str, wake_time: str) -> int:
    """
    新增一筆睡眠紀錄到資料庫
    :param user_id: 使用者 ID
    :param sleep_time: 睡眠開始時間 (格式: YYYY-MM-DD HH:MM:SS)
    :param wake_time: 睡眠結束時間 (格式: YYYY-MM-DD HH:MM:SS)
    :return: 新增的紀錄 record_id
    """
    async with Database.connection() as conn:
        try:
            record_id = await insert_sleep_record(conn, user_id, sleep_time, wake_time)
            if record_id is None:
                raise DatabaseError("資料庫新增睡眠紀錄失敗")
            return record_id
        except Exception as e:
            raise DatabaseError(f"新增睡眠紀錄時發生錯誤: {e}")

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

async def get_weekly_sleep_records(user_id: int, date: str, sunday_first: bool = True) -> dict:
    """
    抓取指定日期所在週的睡眠記錄。

    :param user_id: 使用者 ID
    :param date: 日期 (格式: YYYY-MM-DD)
    :param sunday_first: 是否以周日作為一周的第一天 (True: 周日, False: 周一)
    :return: 一周的睡眠記錄
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
            records = await get_sleep_time_by_week(conn, user_id, start_date_str, end_date_str)
            return {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "records": records
            }
    except NotFoundError:
        raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 所在一周的睡眠記錄")
    except Exception as e:
        raise DatabaseError(f"查詢一周睡眠記錄時發生錯誤: {e}")