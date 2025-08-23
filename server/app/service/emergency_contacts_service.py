import aiomysql
from ..db import Database
from ..dao.emergency_contacts_dao import (
    insert_emergency_contacts,
    select_contacts_by_user_id,
    select_contact_by_pair,
    delete_contact
)
from ..dao.users_dao import select_user_by_id, select_user_by_phone
from ..dao.role_dao import select_role_name_by_id
from ..exceptions import NotFoundError, AlreadyExistsError, DatabaseError


async def add_contact_by_phone(user_phone, contact_phone, priority, relationship):
    async with Database.connection() as conn:
        user = await select_user_by_phone(conn, user_phone)
        if not user:
            raise NotFoundError(f"phone {user_phone} 不存在")
        contact = await select_user_by_phone(conn, contact_phone)
        if not contact:
            raise NotFoundError(f"phone {contact_phone} 不存在")

        user_id = user["user_id"]
        contact_id = contact["user_id"]

        # 檢查是否已存在雙向關係
        try:
            await select_contact_by_pair(conn, user_id, contact_id)
            raise AlreadyExistsError("該關係已存在")
        except NotFoundError:
            pass
        try:
            await select_contact_by_pair(conn, contact_id, user_id)
            raise AlreadyExistsError("該關係已存在")
        except NotFoundError:
            pass

        try:
            success = await insert_emergency_contacts(conn, user_id, contact_id, priority, relationship)
            if not success:
                raise DatabaseError("新增關係失敗")
            return "新增成功"
        except Exception as e:
            raise DatabaseError(f"新增照護關係時發生錯誤: {e}")

async def get_contact_relations(user_phone: int, role: int = None) -> list:
    """
    根據使用者電話取得照護關係列表。
    :param user_phone: 使用者電話
    :param role: 可選，若提供則只返回特定角色的聯絡人
    :return: 照護關係列表，每個元素包含聯絡人資訊
    """
    async with Database.connection() as conn:
        user = await select_user_by_phone(conn, user_phone)
        if not user:
            raise NotFoundError(f"phone {user_phone} 不存在")

        user_id = user["user_id"]

        try:
            contacts_as_user = await select_contacts_by_user_id(conn, user_id)
        except NotFoundError:
            contacts_as_user = []

        contacts_as_contact = []
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                query = "SELECT * FROM emergency_contacts WHERE contact_id = %s"
                await cursor.execute(query, (user_id,))
                contacts_as_contact = list(await cursor.fetchall())
        except Exception as e:
            print(f"[ERROR] 查詢照護關係失敗: {e}")

        all_contacts = contacts_as_user + contacts_as_contact
        unique = {}
        for c in all_contacts:
            key = (c["user_id"], c["contact_id"])
            unique[key] = c

        result = []
        for c in unique.values():
            other_id = c["contact_id"] if c["user_id"] == user_id else c["user_id"]

            try:
                other_user = await select_user_by_id(conn, other_id)
            except NotFoundError:
                continue

            try:
                role_name = await select_role_name_by_id(conn, other_user["role_id"])
            except NotFoundError:
                role_name = "未知角色"

            contact_info = {
                "user_id": other_user["user_id"],
                "name": other_user.get("name"),
                "phone": other_user.get("phone"),
                "role_id": other_user.get("role_id"),
                "role_name": role_name,
                "line_id": other_user.get("line_id"),
                "priority": c["priority"],
                "relationship": c["relationship"]
            }
            result.append(contact_info)

        # ⬇️ 這裡做條件過濾
        if role is not None:
            # 支援 role 為 int 或 str（習慣上資料庫 role_id 多為 int）
            result = [c for c in result if c["role_id"] == role]

        return result

async def remove_contact(user_phone: int, contact_phone: int) -> str:
    async with Database.connection() as conn:
        user = await select_user_by_phone(conn, user_phone)
        if not user:
            raise NotFoundError(f"phone {user_phone} 不存在")
        contact = await select_user_by_phone(conn, contact_phone)
        if not contact:
            raise NotFoundError(f"phone {contact_phone} 不存在")

        user_id = user["user_id"]
        contact_id = contact["user_id"]

        found = False
        try:
            await select_contact_by_pair(conn, user_id, contact_id)
            found = True
        except NotFoundError:
            pass
        try:
            await select_contact_by_pair(conn, contact_id, user_id)
            found = True
        except NotFoundError:
            pass

        if not found:
            raise NotFoundError("該關係不存在")

        try:
            success1 = await delete_contact(conn, user_id, contact_id)
            success2 = await delete_contact(conn, contact_id, user_id)
            if not (success1 or success2):
                raise DatabaseError("刪除失敗")
            return "刪除成功"
        except Exception as e:
            raise DatabaseError(f"刪除照護關係時發生錯誤: {e}")

