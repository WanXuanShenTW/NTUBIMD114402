from ..dao.video_watchlist_dao import (
    insert_video_watchlist,
    delete_watchlist_by_id,
    select_watchlist_entry_by_user_and_record,
    select_watchlist_record_ids_by_user
)
from ..dao.fall_events_dao import select_fall_event_by_id
from ..db import get_connection

def get_watchlist_by_user_id(user_id: int) -> list:
    """
    根據 user_id 取得觀看清單。
    :param user_id: 使用者 ID
    :return: 影片清單（list），失敗則回傳空 list
    """
    with get_connection() as conn:
        results = select_watchlist_entry_by_user_and_record(conn, user_id)
        return results if results is not None else []  

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
    
def get_watchlist_video_data_by_id(user_id: int, video_type:str) -> list:
    """
    根據 user_id 取得觀看清單中的跌倒影片資料。

    :param user_id: 使用者 ID
    :return: 包含影片資料的清單，若無資料則回傳空清單
    """
    with get_connection() as conn:
        record_ids = select_watchlist_record_ids_by_user(conn, user_id, video_type)
        if not record_ids:
            return []
        
        video_data = []
        for record_id in record_ids:
            fall_event = select_fall_event_by_id(conn, record_id)
            if fall_event:
                video_data.append(fall_event)
        
        return video_data