from typing import Optional, Dict, Any
from aiomysql.cursors import DictCursor

async def get_elder_by_caregiver_user_id(conn, caregiver_user_id: int) -> Optional[Dict[str, Any]]:
    async with conn.cursor(DictCursor) as cursor:
        await cursor.execute("""
            SELECT u.*
            FROM users u
            JOIN emergency_contacts ec ON ec.elder_user_id = u.user_id
            WHERE ec.caregiver_user_id = %s
            LIMIT 1
        """, (caregiver_user_id,))
        return await cursor.fetchone()

async def get_elder_by_reversed_mapping(conn, user_id: int) -> Optional[Dict[str, Any]]:
    async with conn.cursor(DictCursor) as cursor:
        await cursor.execute("""
            SELECT u.*
            FROM users u
            JOIN emergency_contacts ec ON ec.caregiver_user_id = u.user_id
            WHERE ec.elder_user_id = %s
            LIMIT 1
        """, (user_id,))
        return await cursor.fetchone()