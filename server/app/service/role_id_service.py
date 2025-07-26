from ..dao.role_dao import select_role_name_by_id
from ..db import get_connection
from ..exceptions import NotFoundError, DatabaseError

async def get_role_name(role_id: int) -> str:
    """
    根據角色 ID 查詢角色名稱。
    """
    async with get_connection() as conn:
        try:
            role_name = await select_role_name_by_id(conn, role_id)
            if role_name is None:
                raise NotFoundError(f"找不到角色 ID {role_id} 的名稱")
            return role_name
        except NotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"查詢角色名稱時發生錯誤: {e}")