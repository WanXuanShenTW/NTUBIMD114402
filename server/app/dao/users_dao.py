import datetime
from typing import Any, Dict, Optional
from mysql.connector import IntegrityError

def insert_user(
    conn,
    name: str,
    phone: str,
    role_id: int,
    password: str,
    line_id: Optional[str] = None
) -> Optional[int]:
    """
    新增一筆使用者資料到資料庫。

    Parameters:
        conn (Connection): 資料庫連線物件
        name (str): 使用者名稱（最長 5 字）
        phone (str): 使用者電話號碼（10 碼，為唯一帳號）
        role_id (int): 使用者角色編號（例如 1=長輩、2=照護者）
        password (str): 使用者密碼（建議經 bcrypt 雜湊）
        line_id (Optional[str]): LINE ID，預設為 None

    Returns:
        int | None: 若成功回傳 user_id，失敗則為 None
    """

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO users (name, phone, role_id, password, created_at, line_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (
            name,
            phone,
            role_id,
            password,
            datetime.datetime.now(),
            line_id
        )
        cursor.execute(query, values)
        conn.commit()
        user_id = cursor.lastrowid
        print(f"[INFO] 新增使用者成功: user_id={user_id}")
        return user_id
    except IntegrityError as e:
        if "Duplicate entry" in str(e) and "phone" in str(e):
            print(f"[ERROR] 電話已被註冊: {phone}")
            raise ValueError(f"電話已被註冊")
        raise e
    except Exception as e:
        print(f"[ERROR] 新增使用者失敗: {e}")
        return None

def update_user(conn, user_id: int, **kwargs) -> bool:
    """
    更新使用者資料。允許更新 name、phone、password、role_id、line_id。
    
    :return: 成功回傳 True，失敗回傳 False
    """
    allowed = ["name", "phone", "password", "role_id", "line_id"]
    fields = []
    values = []

    for k in allowed:
        if k in kwargs:
            fields.append(f"{k} = %s")
            values.append(kwargs[k])
    if not fields:
        return False

    values.append(user_id)
    query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s"
    try:
        cursor = conn.cursor()
        cursor.execute(query, tuple(values))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] 更新使用者失敗: {e}")
        return False

def select_user_by_phone(conn, phone: str):
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE phone = %s"
        cursor.execute(query, (phone,))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] 查詢使用者資料失敗: {e}")
        return None
    
def select_user_by_id(conn, id: str):
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE user_id = %s"
        cursor.execute(query, (id,))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] 查詢使用者資料失敗: {e}")
        return None
    
def delete_user(conn, phone: str) -> bool:
    """
    根據 phone 刪除使用者帳號。
    :return: 成功回傳 True，失敗回傳 False
    """
    try:
        cursor = conn.cursor()
        query = "DELETE FROM users WHERE phone = %s"
        cursor.execute(query, (phone,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] 刪除使用者失敗: {e}")
        return False