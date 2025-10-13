from typing import Any, Dict, List, Optional
from aiomysql.cursors import DictCursor
from pymysql.err import IntegrityError
from ..exceptions import DatabaseError, NotFoundError

async def insert_sit_event(
    conn,
    user_id: int,
    start_at: str,
    end_at: str
) -> Optional[int]:
    """
    新增一筆久坐事件資料到資料庫
    """
    async with conn.cursor() as cursor:
        try:
            query = """
                INSERT INTO `114-402`.sit_events (user_id, start_at, end_at)
                VALUES (%s, %s, %s)
            """
            values = (user_id, start_at, end_at)
            await cursor.execute(query, values)
            return cursor.lastrowid
        except IntegrityError as e:
            raise DatabaseError(f"資料庫新增久坐事件失敗: {e}")
        except Exception as e:
            raise DatabaseError(f"新增久坐事件時發生錯誤: {e}")

async def get_sit_events_by_date(conn, user_id: int, date: str) -> List[Dict[str, Any]]:
    """
    抓取指定日期的久坐事件
    """
    async with conn.cursor(DictCursor) as cursor:
        query = """
            SELECT *
            FROM `114-402`.sit_events
            WHERE user_id = %s AND DATE(start_at) = %s
        """
        await cursor.execute(query, (user_id, date))
        rows = await cursor.fetchall()
        if not rows:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {date} 的久坐事件記錄")
        return rows

async def get_sit_events_by_week(conn, user_id: int, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    抓取指定日期範圍內的久坐事件
    """
    async with conn.cursor(DictCursor) as cursor:
        query = """
            SELECT *
            FROM `114-402`.sit_events
            WHERE user_id = %s AND start_at >= %s AND start_at < %s
        """
        await cursor.execute(query, (user_id, start_date, end_date))
        rows = await cursor.fetchall()
        if not rows:
            raise NotFoundError(f"找不到 user_id={user_id} 在 {start_date} 至 {end_date} 的久坐事件記錄")
        return rows
    
async def get_sit_events_overlapping(
    conn,
    user_id: int,
    day_start,   # datetime.datetime
    day_end      # datetime.datetime
) -> List[Dict[str, Any]]:
    """
    取出與 [day_start, day_end) 有交集的 sit_events。
    條件：start_at < day_end AND (end_at IS NULL OR end_at > day_start)
    """
    sql = """
        SELECT record_id, user_id, start_at, end_at
        FROM `114-402`.sit_events
        WHERE user_id = %s
          AND start_at < %s
          AND (end_at IS NULL OR end_at > %s)
        ORDER BY start_at ASC
    """
    async with conn.cursor(DictCursor) as cur:
        await cur.execute(sql, (user_id, day_end, day_start))
        return await cur.fetchall()