from ..db import get_connection
from ..dao.emergency_contacts_dao import (
    insert_emergency_contacts,
    select_contacts_by_user_id,
    select_contact_by_pair,
    delete_contact
)
from ..dao.users_dao import select_user_by_id, select_user_by_phone
from ..dao.role_dao import select_role_name_by_id
from ..exceptions import NotFoundError, AlreadyExistsError, DatabaseError

def add_contact_by_phone(user_phone, contact_phone, priority, relationship):
    with get_connection() as conn:
        user = select_user_by_phone(conn, user_phone)
        if not user:
            raise NotFoundError(f"phone {user_phone} 不存在")
        contact = select_user_by_phone(conn, contact_phone)
        if not contact:
            raise NotFoundError(f"phone {contact_phone} 不存在")

        user_id = user["user_id"]
        contact_id = contact["user_id"]

        # 檢查關係是否已存在（雙向）
        exists = False
        try:
            select_contact_by_pair(conn, user_id, contact_id)
            exists = True
        except NotFoundError:
            pass
        try:
            select_contact_by_pair(conn, contact_id, user_id)
            exists = True
        except NotFoundError:
            pass
        if exists:
            raise AlreadyExistsError("該關係已存在")

        try:
            success = insert_emergency_contacts(conn, user_id, contact_id, priority, relationship)
            if not success:
                raise DatabaseError("新增關係失敗")
            return "新增成功"
        except Exception as e:
            raise DatabaseError(f"新增照護關係時發生錯誤: {e}")

def get_contact_relations(user_phone: int):
    with get_connection() as conn:
        user = select_user_by_phone(conn, user_phone)
        if not user:
            raise NotFoundError(f"phone {user_phone} 不存在")
        
        user_id = user["user_id"]
        try:
            contacts_as_user = select_contacts_by_user_id(conn, user_id)
        except NotFoundError:
            contacts_as_user = []
        contacts_as_contact = []
        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM emergency_contacts WHERE contact_id = %s"
            cursor.execute(query, (user_id,))
            contacts_as_contact = cursor.fetchall()
        except Exception as e:
            print(f"[ERROR] 查詢照護關係失敗: {e}")

        all_contacts = contacts_as_user + contacts_as_contact
        unique = {}
        for c in all_contacts:
            key = (c["user_id"], c["contact_id"])
            unique[key] = c

        result = []
        for c in unique.values():
            if c["user_id"] == user_id:
                other_id = c["contact_id"]
            elif c["contact_id"] == user_id:
                other_id = c["user_id"]
            else:
                continue

            try:
                other_user = select_user_by_id(conn, other_id)
            except NotFoundError:
                continue
            role_name = None
            try:    
                role_name = select_role_name_by_id(conn, other_user["role_id"])
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
        return result

def remove_contact(user_phone: int, contact_phone: int) -> str:
    with get_connection() as conn:
        user = select_user_by_phone(conn, user_phone)
        if not user:
            raise NotFoundError(f"phone {user_phone} 不存在")
        contact = select_user_by_phone(conn, contact_phone)
        if not contact:
            raise NotFoundError(f"phone {contact_phone} 不存在")

        user_id = user["user_id"]
        contact_id = contact["user_id"]
        
        if not (select_contact_by_pair(conn, user_id, contact_id) or select_contact_by_pair(conn, contact_id, user_id)):
            raise NotFoundError("該關係不存在")

        try:
            success1 = delete_contact(conn, user_id, contact_id)
            success2 = delete_contact(conn, contact_id, user_id)
            success = success1 or success2
            if not success:
                raise DatabaseError("刪除失敗")
            return "刪除成功"
        except Exception as e:
            raise DatabaseError(f"刪除照護關係時發生錯誤: {e}")