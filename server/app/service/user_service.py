from ..dao.users_dao import (
    insert_user,
    update_user,
    select_user_by_phone,
    delete_user
)
from ..db import get_connection
from ..exceptions import NotFoundError, DatabaseError, AlreadyExistsError

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
        try:
            user_id = insert_user(conn, name, phone, role_id, password, line_id)
            if user_id is None:
                raise DatabaseError("資料庫新增使用者失敗")
            return user_id
        except AlreadyExistsError:
            raise
        except Exception as e:
            raise DatabaseError(f"新增使用者時發生錯誤: {e}")

def update_user_info(phone: str, **data) -> bool:
    with get_connection() as conn:
        user = select_user_by_phone(conn, phone)
        if not user:
            raise NotFoundError("找不到該電話對應的使用者")
        user_id = user["user_id"]
        try:
            return update_user(conn, user_id, **data)
        except NotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"更新使用者資料時發生錯誤: {e}")

def change_user_password(phone: str, old_password: str, new_password: str) -> bool:
    """
    更改使用者密碼。

    :param phone: 使用者電話
    :param old_password: 舊密碼
    :param new_password: 新密碼
    :return: 是否成功更改密碼
    """
    with get_connection() as conn:
        user = select_user_by_phone(conn, phone)
        if not user:
            raise NotFoundError("找不到該電話對應的使用者")
        if str(user["password"]) != str(old_password):
            raise DatabaseError("舊密碼不正確")
        try:
            return update_user(conn, user["user_id"], password=new_password)
        except NotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"更改密碼時發生錯誤: {e}")

def get_user_info(phone: str) -> dict:
    with get_connection() as conn:
        try:
            user = select_user_by_phone(conn, phone)
            if not user:
                raise NotFoundError("找不到該使用者")
            return user
        except NotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"查詢使用者資料時發生錯誤: {e}")

def delete_user_account(phone: str) -> bool:
    with get_connection() as conn:
        try:
            return delete_user(conn, phone)
        except NotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"刪除使用者時發生錯誤: {e}")