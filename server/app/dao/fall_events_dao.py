import datetime
from typing import Any, Dict, List, Optional
from aiomysql.cursors import DictCursor
from ..exceptions import DatabaseError, NotFoundError

async def insert_fall_event(
    conn,
    user_id: int,
    detected_time: datetime.datetime,
    location: str,
    pose_before_fall: str
) -> Optional[int]:
    """將跌倒事件紀錄插入資料庫"""
    try:
        async with conn.cursor() as cursor:
            query = """
                INSERT INTO fall_events (user_id, detected_time, location, pose_before_fall)
                VALUES (%s, %s, %s, %s)
            """
            values = (user_id, detected_time, location, pose_before_fall)
            await cursor.execute(query, values)
            record_id = cursor.lastrowid
            print(f"[INFO] 新增跌倒事件成功: record_id={record_id}")
            return record_id
    except Exception as e:
        raise DatabaseError(f"新增跌倒事件失敗: {e}")

async def select_fall_event_by_user_id_and_time_range(
    conn,
    user_id: int,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None
) -> List[Dict[str, Any]]:
    """根據 user_id 和時間區段查詢跌倒事件紀錄"""
    try:
        async with conn.cursor(DictCursor) as cursor:
            query = """
                SELECT record_id, user_id, detected_time, location, pose_before_fall
                FROM fall_events
                WHERE user_id = %s
            """
            values = [user_id]

            # 添加時間篩選條件
            if start_time:
                query += " AND detected_time >= %s"
                values.append(start_time)
            if end_time:
                query += " AND detected_time <= %s"
                values.append(end_time)

            query += " ORDER BY detected_time DESC"
            await cursor.execute(query, values)
            records = await cursor.fetchall()
            return records
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢跌倒事件失敗: {e}")