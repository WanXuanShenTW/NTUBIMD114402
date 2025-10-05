import aiomysql
from ..db import Database
from ..dao.emergency_contacts_dao import (
    delete_all_contacts_by_caregiver_user_id,
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
        elder_user = await select_user_by_phone(conn, elder_phone)  # 修正：先存入變數
        caregiver_user = await select_user_by_phone(conn, caregiver_phone)  # 修正：先存入變數

        elder_user_id = elder_user["user_id"]
        caregiver_user_id = caregiver_user["user_id"]

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
    async with Database.connection() as conn:
        elder_user = await select_user_by_phone(conn, elder_phone)  # 修正：先存入變數
        elder_user_id = elder_user["user_id"]

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
                "relationship": contact["relationship"]
            }
            result.append(contact_info)

        return result

async def get_contacts_by_caregiver(caregiver_phone: str) -> list:
    async with Database.connection() as conn:
        caregiver_user = await select_user_by_phone(conn, caregiver_phone)  # 修正：先存入變數
        caregiver_user_id = caregiver_user["user_id"]

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
        elder_user = await select_user_by_phone(conn, elder_phone)  # 修正：先存入變數
        caregiver_user = await select_user_by_phone(conn, caregiver_phone)  # 修正：先存入變數

        elder_user_id = elder_user["user_id"]
        caregiver_user_id = caregiver_user["user_id"]

        try:
            success = await delete_contact(conn, elder_user_id, caregiver_user_id)
            if not success:
                raise DatabaseError("刪除失敗")
            return "刪除成功"
        except NotFoundError:
            raise NotFoundError("該關係不存在")
        except Exception as e:
            raise DatabaseError(f"刪除照護關係時發生錯誤: {e}")

async def remove_all_contacts_by_caregiver(caregiver_phone: str) -> str:
    async with Database.connection() as conn:
        caregiver_user = await select_user_by_phone(conn, caregiver_phone)  # 修正：先存入變數
        caregiver_user_id = caregiver_user["user_id"]

        try:
            deleted_count = await delete_all_contacts_by_caregiver_user_id(conn, caregiver_user_id)
            return f"成功刪除 {deleted_count} 筆照護關係"
        except NotFoundError:
            raise NotFoundError(f"找不到 phone {caregiver_phone} 的任何照護關係可刪除")
        except Exception as e:
            raise DatabaseError(f"刪除照護關係時發生錯誤: {e}")