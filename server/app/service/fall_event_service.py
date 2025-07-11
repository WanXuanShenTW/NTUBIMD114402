from ..dao.fall_events_dao import (
    insert_fall_event,
    select_fall_event_records_by_user_and_time_range,
    select_fall_event_video_filename_by_id
)
from ..dao.video_watchlist_dao import select_watchlist_entry_by_user_and_record
from ..db import get_connection
from ..exceptions import DatabaseError, NotFoundError

def add_fall_event_with_video(
    user_id: int,
    location: str,
    pose_before_fall: str,
    video_filename: str
) -> int:
    """
    將跌倒事件影片資訊存入資料庫，包含位置與跌倒前姿勢。

    :param user_id: 使用者 ID
    :param location: 跌倒地點
    :param pose_before_fall: 跌倒前的動作敘述
    :param video_filename: 儲存的影片檔案名稱
    :return: 新增的 record_id，若失敗則拋出例外
    """
    with get_connection() as conn:
        try:
            record_id = insert_fall_event(conn, user_id, location, pose_before_fall, video_filename)
            if record_id is None:
                raise DatabaseError("資料庫紀錄跌倒影像資料失敗")
            return record_id
        except Exception as e:
            raise DatabaseError(f"新增跌倒事件時發生錯誤: {e}")

def get_fall_event_video_filename_by_record_id(record_id: int) -> str:
    with get_connection() as conn:
        try:
            return select_fall_event_video_filename_by_id(conn, record_id)
        except NotFoundError as e:
            raise NotFoundError(str(e))
        except Exception as e:
            raise DatabaseError(f"查詢影片檔名時發生錯誤: {e}")
    
def get_video_filename_for_fall_event(elder_id: int, caregiver_id: int, start=None, end=None, limit=5):
    """
    取得跌倒影片資料，並標記每筆是否在該使用者的觀看清單。
    """
    with get_connection() as conn:
        try:
            results = select_fall_event_records_by_user_and_time_range(conn, elder_id, start, end, limit)
            for row in results:
                record_id = row.get("record_id")
                try:
                    row["in_watchlist"] = select_watchlist_entry_by_user_and_record(conn, caregiver_id, record_id) is not None
                except NotFoundError:
                    row["in_watchlist"] = False
            return results
        except NotFoundError as e:
            # 若查無資料，回傳空清單或拋出錯誤皆可
            return []
        except Exception as e:
            raise DatabaseError(f"查詢跌倒影片資料時發生錯誤: {e}")