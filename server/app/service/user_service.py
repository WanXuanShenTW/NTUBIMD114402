from ..db import get_connection
from ..dao.users_dao import insert_user
from mysql.connector import IntegrityError

def register_user(name: str, phone: str, password: str, role_id: int, line_id: str = None):
    with get_connection() as conn:
        try:
            user_id = insert_user(conn, name, phone, password, role_id, line_id)
            return {"user_id": user_id}
        except IntegrityError as e:
            if "Duplicate entry" in str(e):
                return {"error": "此電話號碼已註冊"}, 409
            return {"error": "資料庫錯誤"}, 500
