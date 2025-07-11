import datetime
from typing import Optional
from mysql.connector import IntegrityError
from ..exceptions import DatabaseError, NotFoundError, AlreadyExistsError

def insert_video_watchlist(
    conn,
    record_id: int,
    user_id: int,
    video_type: str
) -> Optional[int]:
    """
    將影片加入觀看清單。
    """
    cursor = None
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO video_watchlist (record_id, user_id, video_type, added_at)
            VALUES (%s, %s, %s, %s)
        """
        values = (
            record_id,
            user_id,
            video_type,
            datetime.datetime.now()
        )
        cursor.execute(query, values)
        conn.commit()
        print(f"[INFO] 新增影片到觀看清單成功: record_id={record_id}")
        return record_id
    except IntegrityError as e:
        if "Duplicate entry" in str(e):
            print(f"[ERROR] 觀看清單已存在: user_id={user_id}, record_id={record_id}, video_type={video_type}")
            raise AlreadyExistsError("該影片已在觀看清單中")
        raise DatabaseError(f"資料庫完整性錯誤: {e}")
    except Exception as e:
        print(f"[ERROR] 新增影片到觀看清單失敗: {e}")
        raise DatabaseError(f"新增影片到觀看清單失敗: {e}")
    finally:    
        if cursor:
            cursor.close()

def select_watchlist_entry_by_user_and_record(conn, user_id: int, record_id: int):
    """
    查詢指定 user_id 和 record_id 是否存在於觀看清單，回傳資料列或拋出 NotFoundError。
    """
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT * FROM video_watchlist
            WHERE user_id = %s AND record_id = %s
            LIMIT 1
        """
        cursor.execute(query, (user_id, record_id))
        result = cursor.fetchone()
        if not result:
            raise NotFoundError(f"找不到 user_id={user_id} 和 record_id={record_id} 的觀看清單資料")
        return result
    except NotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] 查詢觀看清單失敗: {e}")
        raise DatabaseError(f"查詢觀看清單失敗: {e}")
    finally:
        if cursor:
            cursor.close()

def select_watchlist_record_ids_by_user(conn, user_id: int, video_type: str) -> list:
    """
    根據 user_id 查詢觀看清單，回傳所有符合條件的 record_id 清單。
    """
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT record_id FROM video_watchlist
            WHERE user_id = %s AND video_type = %s
        """
        cursor.execute(query, (user_id, video_type))
        rows = cursor.fetchall()
        return [row["record_id"] for row in rows] if rows else []
    except Exception as e:
        print(f"[ERROR] 查詢觀看清單失敗: {e}")
        raise DatabaseError(f"查詢觀看清單失敗: {e}")
    finally:
        if cursor:
            cursor.close()
    
def delete_watchlist_by_id(
    conn,
    user_id: int,
    record_id: int,
    video_type: str
) -> bool:
    """
    以 user_id、record_id、video_type 為條件刪除唯一一筆收藏。
    """
    cursor = None
    try:
        cursor = conn.cursor()
        query = """
            DELETE FROM video_watchlist
            WHERE user_id = %s AND record_id = %s AND video_type = %s
            LIMIT 1
        """
        cursor.execute(query, (user_id, record_id, video_type))
        conn.commit()
        if cursor.rowcount == 0:
            raise NotFoundError(f"找不到 user_id={user_id}、record_id={record_id}、video_type={video_type} 的收藏可刪除")
        return True
    except NotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] 刪除收藏失敗: {e}")
        raise DatabaseError(f"刪除收藏失敗: {e}")
    finally:
        if cursor:
            cursor.close()