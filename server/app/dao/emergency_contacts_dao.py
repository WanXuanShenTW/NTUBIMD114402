import aiomysql
from ..exceptions import DatabaseError, NotFoundError, AlreadyExistsError

async def insert_emergency_contacts(conn, user_id: int, contact_id: int, priority: int, relationship: str) -> bool:
    try:
        async with conn.cursor() as cursor:
            query = """
                INSERT INTO emergency_contacts (user_id, contact_id, priority, relationship)
                VALUES (%s, %s, %s, %s)
            """
            await cursor.execute(query, (user_id, contact_id, priority, relationship))
            await conn.commit()
            return True
    except Exception as e:
        raise DatabaseError(f"新增照護關係失敗: {e}")

async def select_contacts_by_user_id(conn, user_id: int):
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = "SELECT * FROM emergency_contacts WHERE user_id = %s"
            await cursor.execute(query, (user_id,))
            results = await cursor.fetchall()
            if not results:
                raise NotFoundError(f"找不到 user_id={user_id} 的照護關係")
            return results
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢照護關係失敗: {e}")

async def select_contact_by_pair(conn, user_id: int, contact_id: int):
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = "SELECT * FROM emergency_contacts WHERE user_id = %s AND contact_id = %s"
            await cursor.execute(query, (user_id, contact_id))
            result = await cursor.fetchone()
            if not result:
                raise NotFoundError(f"找不到 user_id={user_id} 與 contact_id={contact_id} 的照護組合")
            return result
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢照護組合失敗: {e}")

async def delete_contact(conn, user_id: int, contact_id: int) -> bool:
    try:
        async with conn.cursor() as cursor:
            query = "DELETE FROM emergency_contacts WHERE user_id = %s AND contact_id = %s"
            await cursor.execute(query, (user_id, contact_id))
            await conn.commit()
            if cursor.rowcount == 0:
                raise NotFoundError(f"找不到 user_id={user_id} 與 contact_id={contact_id} 的照護關係可刪除")
            return True
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"刪除照護關係失敗: {e}")