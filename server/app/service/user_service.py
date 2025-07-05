from ..dao.users_dao import (
    insert_user,
    update_user,
    select_user_by_phone,
    delete_user
)
from ..db import get_connection

def add_user(
    name: str,
    phone: str,
    role_id: int,
    password: str,
    line_id: str = None
) -> int:
    """
    新增一筆使用者資料到資料庫。

    :param name: 使用者名稱
    :param phone: 電話
    :param role_id: 角色ID
    :param password: 密碼
    :param line_id: LINE ID（可選）
    :return: 新增的 user_id，若失敗則拋出例外
    """
    with get_connection() as conn:
        user_id = insert_user(conn, name, phone, role_id, password, line_id)
        if user_id is None:
            raise RuntimeError("資料庫新增使用者失敗")
        return user_id

def update_user_info(phone: str, **data) -> bool:
    with get_connection() as conn:
        user = select_user_by_phone(conn, phone)
        if not user:
            raise ValueError("找不到該電話對應的使用者")
        user_id = user["user_id"]
        return update_user(conn, user_id, **data)

def get_user_info(phone: str) -> dict:
    with get_connection() as conn:
        user = select_user_by_phone(conn, phone)
        if not user:
            raise ValueError("找不到該使用者")
        return user

def delete_user_account(phone: str) -> bool:
    with get_connection() as conn:
        return delete_user(conn, phone)