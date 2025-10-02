from typing import List, Dict, Any
from aiomysql.cursors import DictCursor
from ..exceptions import DatabaseError

async def get_users_for_weekly_push_by_time(conn, hour: int, minute: int, weekday: int) -> List[Dict[str, Any]]:
    """獲取指定時間應該推播週報的用戶列表"""
    async with conn.cursor(DictCursor) as cursor:
        try:
            # 查找符合推播時間的用戶
            # weekday: 0=週日, 1=週一, ..., 6=週六
            query = """
                SELECT DISTINCT 
                    np.user_id,
                    np.weekdays,
                    np.hour,
                    np.minute,
                    lb.line_user_id,
                    ec.elder_user_id as elder_id
                FROM notification_preferences np
                JOIN linebot_notification_binding lb ON np.user_id = lb.user_id
                JOIN emergency_contacts ec ON np.user_id = ec.caregiver_user_id
                WHERE np.hour = %s 
                  AND np.minute = %s
                  AND FIND_IN_SET(%s, np.weekdays) > 0
                  AND lb.line_user_id IS NOT NULL
            """
            await cursor.execute(query, (hour, minute, weekday))
            rows = await cursor.fetchall()
            return rows or []
        except Exception as e:
            raise DatabaseError(f"查詢推播用戶失敗: {e}")

async def get_all_elders_with_recent_reports(conn) -> List[Dict[str, Any]]:
    """獲取所有有最近週報的長者列表"""
    async with conn.cursor(DictCursor) as cursor:
        try:
            query = """
                SELECT DISTINCT elder_id
                FROM weekly_interaction_reports 
                WHERE DATE(end_date) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
                ORDER BY elder_id
            """
            await cursor.execute(query)
            rows = await cursor.fetchall()
            return rows or []
        except Exception as e:
            raise DatabaseError(f"獲取長者列表失敗: {e}")