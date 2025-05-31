from ..dao.video_watchlist_dao import insert_video_watchlist, delete_watchlist_by_id
from ..db import get_connection

def add_video_to_watchlist(record_id: int, user_id: int, video_type: str) -> int:
    """
    將影片加入觀察清單。

    :param record_id: 影片記錄 ID
    :param user_id: 使用者 ID
    :param video_type: 影片類型
    :return: 新增的 record_id，失敗則拋出例外
    """
    with get_connection() as conn:
        result = insert_video_watchlist(conn, record_id, user_id, video_type)
        if result is None:
            raise RuntimeError("新增影片到觀察清單失敗")
        return result

def remove_video_from_watchlist(user_id: int, record_id: int, video_type: str) -> bool:
    """
    以 user_id、record_id、video_type 刪除唯一一筆資料。
    """
    with get_connection() as conn:
        return delete_watchlist_by_id(conn, user_id, record_id, video_type)