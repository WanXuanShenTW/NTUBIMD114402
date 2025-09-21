import datetime
from ..dao.fall_events_dao import (
    insert_fall_event,
    select_fall_event_by_user_id
)
from ..db import Database
from ..exceptions import DatabaseError, NotFoundError

async def add_fall_event(
    user_id: int,
    detected_time: datetime.datetime,
    location: str,
    pose_before_fall: str
) -> int:
    """
    將跌倒事件影片資訊存入資料庫，包含位置與跌倒前姿勢。

    :param user_id: 使用者 ID
    :param location: 跌倒地點
    :param pose_before_fall: 跌倒前的動作敘述
    :param video_filename: 儲存的影片檔案名稱
    :return: 新增的 record_id，若失敗則拋出例外
    """
    async with Database.connection() as conn:
        try:
            record_id = await insert_fall_event(conn, user_id, detected_time, location, pose_before_fall)
            if record_id is None:
                raise DatabaseError("資料庫紀錄跌倒影像資料失敗")
            return record_id
        except Exception as e:
            raise DatabaseError(f"新增跌倒事件時發生錯誤: {e}")
        
async def get_fall_event_records(user_id: int, limit: int = 10):
    """
    根據 user_id 查詢跌倒事件紀錄，移除時間範圍限制。

    :param user_id: 使用者 ID
    :param limit: 查詢筆數上限，預設為 10
    :return: 跌倒事件紀錄列表
    """
    async with Database.connection() as conn:
        try:
            records = await select_fall_event_by_user_id(conn, user_id, limit)
            return records
        except NotFoundError:
            raise NotFoundError("查無此使用者")
        except Exception as e:
            raise DatabaseError(f"查詢跌倒事件時發生錯誤: {e}")