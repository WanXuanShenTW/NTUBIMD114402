import datetime
from typing import Any, Dict, List, Optional
from ..exceptions import DatabaseError, NotFoundError
import aiomysql

async def insert_fall_event(
    conn,
    user_id: int,
    location: str,
    pose_before_fall: str,
    video_filename: str
) -> Optional[int]:
    """
    將跌倒事件紀錄插入資料庫。
    """
    try:
        async with conn.cursor() as cursor:
            query = """
                INSERT INTO fall_events (user_id, detected_time, location, pose_before_fall, video_filename)
                VALUES (%s, %s, %s, %s, %s)
            """
            values = (
                user_id,
                datetime.datetime.now(),
                location,
                pose_before_fall,
                video_filename
            )
            await cursor.execute(query, values)
            await conn.commit()
            record_id = cursor.lastrowid
            print(f"[INFO] 新增跌倒事件成功: record_id={record_id}")
            return record_id
    except Exception as e:
        raise DatabaseError(f"新增跌倒事件失敗: {e}")

async def select_fall_event_records_by_user_and_time_range(
    conn,
    user_id: int,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = """
                SELECT record_id, user_id, detected_time, location, pose_before_fall, video_filename
                FROM fall_events
                WHERE user_id = %s AND video_filename IS NOT NULL
            """
            values = [user_id]
            if start and end:
                query += " AND detected_time BETWEEN %s AND %s"
                values.extend([start, end])
            elif start:
                query += " AND detected_time >= %s"
                values.append(start)
            elif end:
                query += " AND detected_time <= %s"
                values.append(end)
            query += " ORDER BY detected_time DESC LIMIT %s"
            values.append(limit)
            await cursor.execute(query, tuple(values))
            rows = await cursor.fetchall()
            if not rows:
                raise NotFoundError(f"找不到 user_id={user_id} 的跌倒事件紀錄")
            return [
                {
                    "record_id": row["record_id"],
                    "user_id": row["user_id"],
                    "detected_time": row["detected_time"],
                    "location": row["location"],
                    "pose_before_fall": row["pose_before_fall"],
                    "video_filename": row["video_filename"]
                }
                for row in rows
            ]
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢跌倒事件失敗: {e}")

async def select_fall_event_video_filename_by_id(conn, record_id: int) -> Optional[str]:
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = """
                SELECT video_filename 
                FROM fall_events 
                WHERE record_id = %s AND video_filename IS NOT NULL
            """
            await cursor.execute(query, (record_id,))
            row = await cursor.fetchone()
            if not row or not row["video_filename"]:
                raise NotFoundError(f"找不到 record_id={record_id} 的影片檔名")
            return row["video_filename"]
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢影片檔名失敗: {e}")

async def select_fall_event_by_id(conn, record_id: int) -> Optional[Dict[str, Any]]:
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = """
                SELECT record_id, user_id, detected_time, location, pose_before_fall, video_filename
                FROM fall_events
                WHERE record_id = %s
            """
            await cursor.execute(query, (record_id,))
            row = await cursor.fetchone()
            if not row:
                raise NotFoundError(f"找不到 record_id={record_id} 的跌倒事件")
            return {
                "record_id": row["record_id"],
                "user_id": row["user_id"],
                "detected_time": row["detected_time"],
                "location": row["location"],
                "pose_before_fall": row["pose_before_fall"],
                "video_filename": row["video_filename"]
            }
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢跌倒事件失敗: {e}")