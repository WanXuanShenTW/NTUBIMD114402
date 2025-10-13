from typing import Any, Dict, List, Optional
from aiomysql.cursors import DictCursor
from datetime import datetime, timedelta
from pymysql.err import IntegrityError
from ..exceptions import DatabaseError, NotFoundError

async def insert_sleep_record(
    conn,
    user_id: int,
    sleep_time: str,
    wake_time: str
) -> Optional[int]:
    """
    新增一筆睡眠紀錄到資料庫
    :param conn: 資料庫連線
    :param user_id: 使用者 ID
    :param sleep_time: 睡眠開始時間 (格式: YYYY-MM-DD HH:MM:SS)
    :param wake_time: 睡眠結束時間 (格式: YYYY-MM-DD HH:MM:SS)
    :return: 新增的紀錄 record_id
    """
    async with conn.cursor() as cursor:
        try:
            query = """
                INSERT INTO `114-402`.sleep_records (user_id, sleep_time, wake_time)
                VALUES (%s, %s, %s)
            """
            values = (user_id, sleep_time, wake_time)
            await cursor.execute(query, values)
            return cursor.lastrowid
        except IntegrityError as e:
            raise DatabaseError(f"資料庫新增睡眠紀錄失敗: {e}")
        except Exception as e:
            raise DatabaseError(f"新增睡眠紀錄時發生錯誤: {e}")

async def get_sleep_time_by_date(conn, user_id: int, date: str) -> Dict[str, Any]:
    """
    抓取指定日期的睡眠時間
    :param conn: 資料庫連線
    :param user_id: 使用者 ID
    :param date: 日期 (格式: YYYY-MM-DD)
    :return: 當天的睡眠記錄
    """
    async with conn.cursor(DictCursor) as cursor:
        query = """
            SELECT sleep_time, wake_time
            FROM sleep_records
            WHERE user_id = %s AND DATE(sleep_time) = %s
        """
        await cursor.execute(query, (user_id, date))
        row = await cursor.fetchone()
        if not row:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 的睡眠記錄")
        return row

async def get_sleep_time_by_week(conn, user_id: int, date: str, sunday_first: bool = True) -> List[Dict[str, Any]]:
    """
    抓取指定日期所在週的睡眠時間
    :param conn: 資料庫連線
    :param user_id: 使用者 ID
    :param date: 日期 (格式: YYYY-MM-DD)
    :param sunday_first: 是否以周日作為一周的第一天 (True: 周日, False: 周一)
    :return: 一周的睡眠記錄
    """
    # 計算一周的起始日期
    input_date = datetime.strptime(date, "%Y-%m-%d")
    weekday = input_date.weekday()  # 0: 周一, 6: 周日
    if sunday_first:
        start_date = input_date - timedelta(days=(weekday + 1) % 7)  # 周日為第一天
    else:
        start_date = input_date - timedelta(days=weekday)  # 周一為第一天

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = (start_date + timedelta(days=7)).strftime("%Y-%m-%d")

    async with conn.cursor(DictCursor) as cursor:
        query = """
            SELECT sleep_time, wake_time
            FROM sleep_records
            WHERE user_id = %s AND sleep_time >= %s AND sleep_time < %s
        """
        await cursor.execute(query, (user_id, start_date_str, end_date_str))
        rows = await cursor.fetchall()
        if not rows:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {start_date_str} 起始的一周的睡眠記錄")
        return rows
    
# === 新增：抓與 [day_start, day_end) 有交集的睡眠區段（未裁切）===
from aiomysql.cursors import DictCursor
from typing import Any, Dict, List

async def get_sleep_records_overlapping(
    conn,
    user_id: int,
    day_start,   # datetime.datetime
    day_end      # datetime.datetime
) -> List[Dict[str, Any]]:
    """
    取出與 [day_start, day_end) 有交集的 sleep_records。
    條件：sleep_time < day_end AND COALESCE(wake_time, day_end) > day_start
    """
    sql = """
        SELECT record_id, user_id, sleep_time, wake_time
        FROM `114-402`.sleep_records
        WHERE user_id = %s
          AND sleep_time < %s
          AND COALESCE(wake_time, %s) > %s
        ORDER BY sleep_time ASC
    """
    async with conn.cursor(DictCursor) as cur:
        await cur.execute(sql, (user_id, day_end, day_end, day_start))
        return await cur.fetchall()
