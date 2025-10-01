import aiomysql
from ..exceptions import DatabaseError, NotFoundError

async def insert_emergency_contacts(conn, elder_user_id: int, caregiver_user_id: int, relationship: str) -> bool:
    """新增緊急聯絡人"""
    async with conn.cursor() as cursor:
        try:
            query = """
                INSERT INTO emergency_contacts (elder_user_id, caregiver_user_id, relationship)
                VALUES (%s, %s, %s)
            """
            await cursor.execute(query, (elder_user_id, caregiver_user_id, relationship))
            return True
        except Exception as e:
            raise DatabaseError(f"新增照護關係失敗: {e}")

async def select_contacts_by_elder_user_id(conn, elder_user_id: int) -> list:
    """根據長者 user_id 查詢所有照護關係"""
    async with conn.cursor(aiomysql.DictCursor) as cursor:
        query = "SELECT * FROM emergency_contacts WHERE elder_user_id = %s"
        await cursor.execute(query, (elder_user_id,))
        results = await cursor.fetchall()
        if not results:
            raise NotFoundError(f"找不到 elder_user_id={elder_user_id} 的照護關係")
        return results

async def select_contacts_by_caregiver_user_id(conn, caregiver_user_id: int) -> list:
    """根據照護者 user_id 查詢所有被照護關係"""
    async with conn.cursor(aiomysql.DictCursor) as cursor:
        query = "SELECT * FROM emergency_contacts WHERE caregiver_user_id = %s"
        await cursor.execute(query, (caregiver_user_id,))
        results = await cursor.fetchall()
        if not results:
            raise NotFoundError(f"找不到 caregiver_user_id={caregiver_user_id} 的照護關係")
        return results

async def select_contact_by_pair(conn, elder_user_id: int, caregiver_user_id: int) -> dict:
    """根據長者與照護者的 user_id 查詢單一關係"""
    async with conn.cursor(aiomysql.DictCursor) as cursor:
        query = "SELECT * FROM emergency_contacts WHERE elder_user_id = %s AND caregiver_user_id = %s"
        await cursor.execute(query, (elder_user_id, caregiver_user_id))
        result = await cursor.fetchone()
        if not result:
            raise NotFoundError(f"找不到 elder_user_id={elder_user_id} 與 caregiver_user_id={caregiver_user_id} 的照護組合")
        return result

async def delete_contact(conn, elder_user_id: int, caregiver_user_id: int) -> bool:
    """根據長者與照護者的 user_id 刪除關係 (邏輯修正)"""
    async with conn.cursor() as cursor:
        query = "DELETE FROM emergency_contacts WHERE elder_user_id = %s AND caregiver_user_id = %s"
        await cursor.execute(query, (elder_user_id, caregiver_user_id))
        if cursor.rowcount == 0:
            raise NotFoundError(f"找不到 elder_user_id={elder_user_id} 與 caregiver_user_id={caregiver_user_id} 的照護關係可刪除")
        return True