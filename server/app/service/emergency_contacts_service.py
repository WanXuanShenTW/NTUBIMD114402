import aiomysql
from ..db import Database
from ..dao.emergency_contacts_dao import (
    insert_emergency_contacts,
    select_contacts_by_elder_user_id,
    select_contacts_by_caregiver_user_id,
    select_contact_by_pair,
    delete_contact
)
from ..dao.users_dao import select_user_by_id, select_user_by_phone
from ..dao.role_dao import select_role_name_by_id
from ..exceptions import NotFoundError, AlreadyExistsError, DatabaseError

async def add_contact_by_phone(elder_phone, caregiver_phone, relationship):
    async with Database.connection() as conn:
        elder = await select_user_by_phone(conn, elder_phone)
        if not elder:
            raise NotFoundError(f"phone {elder_phone} 不存在")
        caregiver = await select_user_by_phone(conn, caregiver_phone)
        if not caregiver:
            raise NotFoundError(f"phone {caregiver_phone} 不存在")

        elder_user_id = elder["user_id"]
        caregiver_user_id = caregiver["user_id"]

        # 檢查是否已存在單向關係
        try:
            await select_contact_by_pair(conn, elder_user_id, caregiver_user_id)
            raise AlreadyExistsError("該關係已存在")
        except NotFoundError:
            pass

        try:
            success = await insert_emergency_contacts(conn, elder_user_id, caregiver_user_id, relationship)
            if not success:
                raise DatabaseError("新增關係失敗")
            return "新增成功"
        except Exception as e:
            raise DatabaseError(f"新增照護關係時發生錯誤: {e}")

async def get_contacts_by_elder(elder_phone: str) -> list:
    """
    根據長者電話取得其所有的照護者關係。
    :param elder_phone: 長者電話
    :return: 照護者列表
    ✅ 修正：使用單一連線 + 移除 line_id
    """
    async with Database.connection() as conn:  # ✅ 使用單一連線處理整個操作
        caregiver = await select_user_by_phone(conn, elder_phone)
        if not caregiver:
            raise NotFoundError(f"phone {elder_phone} 不存在")

        elder_user_id = caregiver["user_id"]

        try:
            contacts = await select_contacts_by_elder_user_id(conn, elder_user_id)
        except NotFoundError:
            return []

        result = []
        for contact in contacts:
            caregiver_user_id = contact["caregiver_user_id"]

            try:
                caregiver = await select_user_by_id(conn, caregiver_user_id)
            except NotFoundError:
                continue

            try:
                role_name = await select_role_name_by_id(conn, caregiver["role_id"])
            except NotFoundError:
                role_name = "未知角色"

            contact_info = {
                "user_id": caregiver["user_id"],
                "name": caregiver.get("name"),
                "phone": caregiver.get("phone"),
                "role_id": caregiver.get("role_id"),
                "role_name": role_name,
                # ❌ 移除這行：line_id 不存在了
                # "line_id": caregiver.get("line_id"),
                "relationship": contact["relationship"]
            }
            result.append(contact_info)

        return result

async def get_contacts_by_caregiver(caregiver_phone: str) -> list:
    """
    根據照護者電話取得其所有的長者關係。
    :param caregiver_phone: 照護者電話
    :return: 長者列表
    """
    async with Database.connection() as conn:
        caregiver = await select_user_by_phone(conn, caregiver_phone)
        if not caregiver:
            raise NotFoundError(f"phone {caregiver_phone} 不存在")

        caregiver_user_id = caregiver["user_id"]

        try:
            contacts = await select_contacts_by_caregiver_user_id(conn, caregiver_user_id)
        except NotFoundError:
            return []

        result = []
        for contact in contacts:
            elder_user_id = contact["elder_user_id"]

            try:
                elder = await select_user_by_id(conn, elder_user_id)
            except NotFoundError:
                continue

            try:
                role_name = await select_role_name_by_id(conn, elder["role_id"])
            except NotFoundError:
                role_name = "未知角色"

            contact_info = {
                "user_id": elder["user_id"],
                "name": elder.get("name"),
                "phone": elder.get("phone"),
                "role_id": elder.get("role_id"),
                "role_name": role_name,
                "relationship": contact["relationship"]
            }
            result.append(contact_info)

        return result

async def remove_contact(elder_phone: str, caregiver_phone: str) -> str:
    async with Database.connection() as conn:
        elder = await select_user_by_phone(conn, elder_phone)
        if not elder:
            raise NotFoundError(f"phone {elder_phone} 不存在")
        caregiver = await select_user_by_phone(conn, caregiver_phone)
        if not caregiver:
            raise NotFoundError(f"phone {caregiver_phone} 不存在")

        elder_user_id = elder["user_id"]
        caregiver_user_id = caregiver["user_id"]

        try:
            success = await delete_contact(conn, elder_user_id, caregiver_user_id)
            if not success:
                raise DatabaseError("刪除失敗")
            return "刪除成功"
        except NotFoundError:
            raise NotFoundError("該關係不存在")
        except Exception as e:
            raise DatabaseError(f"刪除照護關係時發生錯誤: {e}")