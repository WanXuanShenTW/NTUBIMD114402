from ..db import get_connection
from ..dao.emergency_contacts_dao import (
    insert_emergency_contacts,
    select_contacts_by_user_id,
    select_contact_by_pair,
    delete_contact
)
from ..dao.users_dao import select_user_by_id, select_user_by_phone

def add_contact_by_phone(user_phone, contact_phone, priority, relationship):
    with get_connection() as conn:
        user = select_user_by_phone(conn, user_phone)
        if not user:
            raise ValueError(f"phone {user_phone} 不存在")
        contact = select_user_by_phone(conn, contact_phone)
        if not contact:
            raise ValueError(f"phone {contact_phone} 不存在")

        user_id = user["user_id"]
        contact_id = contact["user_id"]

        # 再呼叫 insert_care_relation
        if select_contact_by_pair(conn, user_id, contact_id) or select_contact_by_pair(conn, contact_id, user_id):
            raise ValueError("該關係已存在")

        success = insert_emergency_contacts(conn, user_id, contact_id, priority, relationship)
        if not success:
            raise RuntimeError("新增關係失敗")
        return "新增成功"

def get_contact_relations(user_phone: int):
    with get_connection() as conn:
        user = select_user_by_phone(conn, user_phone)
        if not user:
            raise ValueError(f"phone {user_phone} 不存在")
        
        user_id = user["user_id"]
        contacts_as_user = select_contacts_by_user_id(conn, user_id)
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

            other_user = select_user_by_id(conn, other_id)
            if not other_user:
                continue

            # 你可以根據需求選擇要回傳哪些欄位
            contact_info = {
                "user_id": other_user["user_id"],
                "name": other_user.get("name"),
                "phone": other_user.get("phone"),
                "role_id": other_user.get("role_id"),
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
            raise ValueError(f"phone {user_phone} 不存在")
        contact = select_user_by_phone(conn, contact_phone)
        if not contact:
            raise ValueError(f"phone {contact_phone} 不存在")

        user_id = user["user_id"]
        contact_id = contact["user_id"]
        
        if not (select_contact_by_pair(conn, user_id, contact_id) or select_contact_by_pair(conn, contact_id, user_id)):
            raise ValueError("該關係不存在")

        success1 = delete_contact(conn, user_id, contact_id)
        success2 = delete_contact(conn, contact_id, user_id)
        success = success1 or success2
        if not success:
            raise RuntimeError("刪除失敗")
        return "刪除成功"
