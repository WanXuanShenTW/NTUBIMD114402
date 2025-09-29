from typing import List, Tuple

async def list_line_uids_by_role(conn, role_id: int = 2) -> List[str]:
    async with conn.cursor() as cursor:
        await cursor.execute("""
            SELECT b.line_user_id
            FROM users u
            JOIN linebot_notification_binding b ON u.user_id = b.user_id
            WHERE u.role_id = %s
        """, (role_id,))
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

async def list_caregiver_line_uids_by_elder(conn, elder_user_id: int) -> Tuple[List[str], str]:
    async with conn.cursor() as cursor:
        await cursor.execute("""
            SELECT b.line_user_id
            FROM emergency_contacts ec
            JOIN users u ON ec.caregiver_user_id = u.user_id
            JOIN linebot_notification_binding b ON u.user_id = b.user_id
            WHERE ec.elder_user_id = %s
        """, (elder_user_id,))
        caregiver_rows = await cursor.fetchall()
        
        await cursor.execute("""
            SELECT u.name
            FROM users u
            WHERE u.user_id = %s
        """, (elder_user_id,))
        elder_row = await cursor.fetchone()
        
        caregiver_line_uids = [row[0] for row in caregiver_rows]
        elder_name = elder_row[0] if elder_row else "未知長者"
        
        return caregiver_line_uids, elder_name