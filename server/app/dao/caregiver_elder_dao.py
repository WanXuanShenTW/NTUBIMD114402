from typing import Optional, Dict, Any
from aiomysql.cursors import DictCursor

async def get_elder_by_caregiver_user_id(conn, caregiver_user_id: int) -> Optional[Dict[str, Any]]:
    """根據照護者 user_id 查找對應的長者資訊"""
    async with conn.cursor(DictCursor) as cursor:
        await cursor.execute("""
            SELECT 
                u.user_id as elder_id,
                u.name as elder_name,
                u.phone as elder_phone,
                u.role_id as elder_role_id,
                u.gender as elder_gender,
                u.address as elder_address,
                u.created_at as elder_created_at,
                ec.elder_user_id as elder_user_id,
                ec.relationship
            FROM users u
            JOIN emergency_contacts ec ON ec.elder_user_id = u.user_id
            WHERE ec.caregiver_user_id = %s
            LIMIT 1
        """, (caregiver_user_id,))
        return await cursor.fetchone()