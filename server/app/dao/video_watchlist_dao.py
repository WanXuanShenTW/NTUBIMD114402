import datetime
from typing import Optional

def insert_video_watchlist(
    conn,
    record_id: int,
    user_id: int,
    video_type: str
) -> Optional[int]:
    """
    將影片加入觀看清單。

    :param conn: 資料庫連線物件，由 service 層傳入
    :param record_id: 影片記錄 ID
    :param user_id: 使用者 ID
    :param video_type: 影片類型（如 "fall", "gait"）
    :return: 成功回傳 record_id，失敗回傳 None
    """
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
    except Exception as e:
        print(f"[ERROR] 新增影片到觀看清單失敗: {e}")
        return None

def select_watchlist_entry_by_user_and_record(conn, user_id: int, record_id: int):
    """
    查詢指定 user_id 和 record_id 是否存在於觀看清單，回傳資料列或 None。
    """
    try:
        cursor = conn.cursor()
        query = """
            SELECT * FROM video_watchlist
            WHERE user_id = %s AND record_id = %s
            LIMIT 1
        """
        cursor.execute(query, (user_id, record_id))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] 查詢觀看清單失敗: {e}")
        return None

def select_watchlist_record_ids_by_user(conn, user_id: int, viedo_type:str) -> list:
    """
    根據 user_id 查詢觀看清單，回傳所有符合條件的 record_id 清單。
    """
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT record_id FROM video_watchlist
            WHERE user_id = %s AND video_type = %s
        """
        cursor.execute(query, (user_id, viedo_type))
        rows = cursor.fetchall()
        return [row["record_id"] for row in rows] if rows else []
    except Exception as e:
        print(f"[ERROR] 查詢觀看清單失敗: {e}")
        return []
    
def delete_watchlist_by_id(
    conn,
    user_id: int,
    record_id: int,
    video_type: str
) -> bool:
    """
    以 user_id、record_id、video_type 為條件刪除唯一一筆收藏。

    :return: 成功回傳 True，失敗回傳 False
    """
    try:
        cursor = conn.cursor()
        query = """
            DELETE FROM video_watchlist
            WHERE user_id = %s AND record_id = %s AND video_type = %s
            LIMIT 1
        """
        cursor.execute(query, (user_id, record_id, video_type))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] 刪除收藏失敗: {e}")
        return False