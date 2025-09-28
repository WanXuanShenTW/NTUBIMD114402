from typing import Optional
from aiomysql.cursors import DictCursor
from ..exceptions import DatabaseError, NotFoundError

async def select_role_name_by_id(conn, role_id: int) -> str:
    """根據 role_id 查詢角色名稱"""
    try:
        async with conn.cursor(DictCursor) as cursor:
            query = "SELECT role_name FROM role WHERE role_id = %s"
            await cursor.execute(query, (role_id,))
            result = await cursor.fetchone()
            if not result:
                raise NotFoundError(f"找不到角色 ID {role_id} 的名稱")
            return result['role_name']
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢角色名稱失敗: {e}")