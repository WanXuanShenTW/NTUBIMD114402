from ..dao.fall_event_dao import insert_fall_event, fetch_fall_events_by_user_and_time_range, get_fall_event_video_filename_by_id
from ..db import get_connection

def save_fall_video_path_with_video(
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
        record_id = insert_fall_event(conn, user_id, location, pose_before_fall, video_filename)
        if record_id is None:
            raise RuntimeError("資料庫紀錄跌倒影像資料失敗")
        return record_id
    
def list_fall_video_data_from_reange(user_id: int, start=None, end=None, limit=5):
    with get_connection() as conn:
        return fetch_fall_events_by_user_and_time_range(conn, user_id, start, end, limit)

def get_video_filename_with_id(record_id: int) -> str:
    with get_connection() as conn:
        return get_fall_event_video_filename_by_id(conn, record_id)
