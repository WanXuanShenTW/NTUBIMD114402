from typing import List, Tuple
from app.db import Database

async def list_line_uids_by_role(role_id: int = 2) -> List[str]:
    """
    取某角色（預設：照護者 role_id=2）的所有已綁定 LINE UID。
    來源：users + system_push_notification_account_binding
    """
    sql = """
        SELECT b.line_user_id
        FROM users u
        JOIN system_push_notification_account_binding b
          ON b.user_id = u.user_id
        WHERE u.role_id = %s
        GROUP BY b.line_user_id
    """
    async with Database.connection() as conn:
        async with conn.cursor(dictionary=True) as cur:
            await cur.execute(sql, (role_id,))
            rows = await cur.fetchall()
    return [r["line_user_id"] for r in (rows or [])]

async def list_caregiver_line_uids_by_elder(elder_user_id: int) -> Tuple[List[str], str]:
    """
    由 elder_user_id 找出該長者之照護者的 LINE UID 清單。
    回傳 (uids, source)
      - "caregivers"           ：正向（ec.caregiver_user_id=照護者, ec.elder_user_id=長者）
      - "caregivers_reversed"  ：反向容錯（表曾誤填）
      - "elder_fallback"       ：找不到照護者綁定時，回長者本人之 LINE UID
      - "none"                 ：完全沒找到
    """
    # 正向：照護者→LINE 綁定
    sql_cg = """
        SELECT b.line_user_id
        FROM emergency_contacts ec
        JOIN system_push_notification_account_binding b
          ON b.user_id = ec.caregiver_user_id
        WHERE ec.elder_user_id = %s
        GROUP BY b.line_user_id
    """
    # 反向容錯：欄位填反
    sql_cg_rev = """
        SELECT b.line_user_id
        FROM emergency_contacts ec
        JOIN system_push_notification_account_binding b
          ON b.user_id = ec.elder_user_id
        WHERE ec.caregiver_user_id = %s
        GROUP BY b.line_user_id
    """
    # Fallback：長者本人
    sql_elder_self = """
        SELECT b.line_user_id
        FROM system_push_notification_account_binding b
        WHERE b.user_id = %s
        LIMIT 1
    """

    async with Database.connection() as conn:
        async with conn.cursor(dictionary=True) as cur:
            # 正向
            await cur.execute(sql_cg, (elder_user_id,))
            rows = await cur.fetchall()
            if rows:
                return [r["line_user_id"] for r in rows], "caregivers"

            # 反向容錯
            await cur.execute(sql_cg_rev, (elder_user_id,))
            rows = await cur.fetchall()
            if rows:
                return [r["line_user_id"] for r in rows], "caregivers_reversed"

            # Fallback：長者本人
            await cur.execute(sql_elder_self, (elder_user_id,))
            row = await cur.fetchone()
            if row:
                return [row["line_user_id"]], "elder_fallback"

    return [], "none"
