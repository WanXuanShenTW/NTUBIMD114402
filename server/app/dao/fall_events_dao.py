import datetime
from typing import Any, Dict, List, Optional
from ..exceptions import DatabaseError, NotFoundError
import aiomysql

async def insert_fall_event(
    conn,
    user_id: int,
    detected_time: datetime.datetime,
    location: str,
    pose_before_fall: str
) -> Optional[int]:
    """
    將跌倒事件紀錄插入資料庫。
    """
    try:
        async with conn.cursor() as cursor:
            query = """
                INSERT INTO fall_events (user_id, detected_time, location, pose_before_fall)
                VALUES (%s, %s, %s, %s)
            """
            values = (
                user_id,
                detected_time,
                location,
                pose_before_fall
            )
            await cursor.execute(query, values)
            await conn.commit()
            record_id = cursor.lastrowid
            print(f"[INFO] 新增跌倒事件成功: record_id={record_id}")
            return record_id
    except Exception as e:
        raise DatabaseError(f"新增跌倒事件失敗: {e}")

async def select_fall_event_by_user_id(
    conn,
    user_id: int,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    根據 user_id 查詢跌倒事件紀錄。
    """
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = """
                SELECT record_id, user_id, detected_time, location, pose_before_fall, video_filename
                FROM fall_events
                WHERE user_id = %s
                ORDER BY detected_time DESC
                LIMIT %s
            """
            values = [user_id, limit]
            await cursor.execute(query, values)
            records = await cursor.fetchall()
            return records
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢跌倒事件失敗: {e}")
