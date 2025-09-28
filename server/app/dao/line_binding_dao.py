from typing import Optional, Dict, Any
from aiomysql.cursors import DictCursor

async def upsert_binding(conn, user_id: int, line_user_id: str) -> None:
    """
    一對一綁定（不改 schema）：
    先刪掉同 user_id 或同 line_user_id 的舊紀錄 → 再插入新紀錄
    使用表：system_push_notification_account_binding
    """
    async with conn.cursor() as cur:
        await cur.execute(
            "DELETE FROM system_push_notification_account_binding "
            "WHERE user_id=%s OR line_user_id=%s",
            (user_id, line_user_id)
        )
        await cur.execute(
            "INSERT INTO system_push_notification_account_binding "
            "(user_id, line_user_id, last_bind_time) "
            "VALUES (%s, %s, NOW())",
            (user_id, line_user_id)
        )
    # 移除 await conn.commit() - 因為連線池設定了 autocommit=True

async def get_user_by_line_user_id(conn, line_user_id: str) -> Optional[Dict[str, Any]]:
    """
    由 LINE 的 Uxxxx… 找到對應使用者
    回傳：{user_id, name, role_id, phone} 或 None
    """
    async with conn.cursor(DictCursor) as cur:
        await cur.execute("""
            SELECT u.user_id, u.name, u.role_id, u.phone
            FROM users u
            JOIN system_push_notification_account_binding b
              ON b.user_id = u.user_id
            WHERE b.line_user_id = %s
            ORDER BY b.last_bind_time DESC
            LIMIT 1
        """, (line_user_id,))
        return await cur.fetchone()

async def get_line_user_id_by_user(conn, user_id: int) -> Optional[str]:
    """
    由 user_id 反查最近一次綁定的 LINE userId（Uxxxx…）
    """
    async with conn.cursor() as cur:
        await cur.execute("""
            SELECT line_user_id
            FROM system_push_notification_account_binding
            WHERE user_id=%s
            ORDER BY last_bind_time DESC
            LIMIT 1
        """, (user_id,))
        row = await cur.fetchone()
        return row[0] if row else None

async def unbind_by_line_user_id(conn, line_user_id: str) -> int:
    async with conn.cursor() as cur:
        await cur.execute(
            "DELETE FROM system_push_notification_account_binding WHERE line_user_id=%s",
            (line_user_id,)
        )
        affected = cur.rowcount
    # 移除 await conn.commit()
    return affected