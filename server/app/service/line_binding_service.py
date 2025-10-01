from typing import Optional, Dict, Any
from app.db import Database
from app.dao.line_binding_dao import (
    unbind_by_line_user_id,
    get_line_user_id_by_user,
    get_user_by_line_user_id,
)

async def unbind_line_user(line_user_id: str) -> int:
    """解除綁定（由 Uxxxx 刪除 mapping）"""
    async with Database.connection() as conn:
        return await unbind_by_line_user_id(conn, line_user_id)

async def get_line_uid_by_user(user_id: int):
    """由 user_id 反查 Uxxxx"""
    async with Database.connection() as conn:
        return await get_line_user_id_by_user(conn, user_id)

async def get_user_by_line_uid(line_user_id: str) -> Optional[Dict[str, Any]]:
    """由 Uxxxx 查到對應 user"""
    async with Database.connection() as conn:
        return await get_user_by_line_user_id(conn, line_user_id)