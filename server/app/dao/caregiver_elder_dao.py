# app/dao/caregiver_elder_dao.py
from typing import Optional, Dict, Any
from app.db import Database

async def get_elder_by_caregiver_user_id(caregiver_user_id: int) -> Optional[Dict[str, Any]]:
    """
    正向關係：
      emergency_contacts.caregiver_user_id = 照護者
      emergency_contacts.elder_user_id     = 長者
    回傳：{ elder_id, elder_name }
    """
    sql = """
        SELECT e.user_id AS elder_id, e.name AS elder_name
        FROM emergency_contacts ec
        JOIN users e ON e.user_id = ec.elder_user_id
        WHERE ec.caregiver_user_id = %s
        ORDER BY e.user_id ASC
        LIMIT 1
    """
    async with Database.connection() as conn:
        async with conn.cursor(dictionary=True) as cur:
            await cur.execute(sql, (caregiver_user_id,))
            return await cur.fetchone()

async def get_elder_by_reversed_mapping(user_id: int) -> Optional[Dict[str, Any]]:
    """
    反向容錯（表裡不小心把欄位放反）：
      emergency_contacts.caregiver_user_id = 長者
      emergency_contacts.elder_user_id     = 照護者
    由『照護者 user_id』往回找長者。
    """
    sql = """
        SELECT e.user_id AS elder_id, e.name AS elder_name
        FROM emergency_contacts ec
        JOIN users e ON e.user_id = ec.caregiver_user_id   -- 這欄其實放了長者
        WHERE ec.elder_user_id = %s
        ORDER BY e.user_id ASC
        LIMIT 1
    """
    async with Database.connection() as conn:
        async with conn.cursor(dictionary=True) as cur:
            await cur.execute(sql, (user_id,))
            return await cur.fetchone()
