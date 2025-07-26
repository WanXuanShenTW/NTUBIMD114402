from typing import Optional
from ..exceptions import DatabaseError, NotFoundError
import aiomysql

async def select_role_name_by_id(conn, role_id: int) -> str:
    """
    根據 role_id 查詢角色名稱。
    :param conn: 資料庫連線物件
    :param role_id: 角色ID
    :return: 角色名稱（找不到則拋出 NotFoundError）
    :raises DatabaseError: 資料庫操作發生錯誤時
    """
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = "SELECT role_name FROM role WHERE role_id = %s"
            await cursor.execute(query, (role_id,))
            result = await cursor.fetchone()
            if not result:
                raise NotFoundError(f"找不到角色 ID {role_id} 的名稱")
            return result['role_name']
    except NotFoundError:
        raise
    except Exception as e:
        # 發生資料庫錯誤時，拋出自訂 DatabaseError
        raise DatabaseError(f"查詢角色名稱失敗: {e}")