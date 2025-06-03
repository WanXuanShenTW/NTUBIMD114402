import datetime
from typing import Optional
from mysql.connector import IntegrityError

def insert_user(conn, name: str, phone: str, password: str, role_id: int, line_id: Optional[str] = None) -> Optional[int]:
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO users (name, phone, password, role_id, line_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (
            name,
            phone,
            password,
            role_id,
            line_id if line_id else None,
            datetime.datetime.now()
        )
        cursor.execute(query, values)
        conn.commit()
        return cursor.lastrowid
    except IntegrityError as e:
        raise e
    except Exception as e:
        print(f"[ERROR] 註冊使用者失敗: {e}")
        return None
