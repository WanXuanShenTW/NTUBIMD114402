import datetime
import aiomysql
from typing import Any, Dict, Optional
from mysql.connector import IntegrityError
from ..exceptions import DatabaseError, NotFoundError, AlreadyExistsError

async def insert_user(
    conn,
    name: str,
    phone: str,
    role_id: int,
    password: str,
    line_id: Optional[str] = None
) -> Optional[int]:
    """
    新增一筆使用者資料到資料庫。
    """
    try:
        async with conn.cursor() as cursor:
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
            await cursor.execute(query, values)
            await conn.commit()
            user_id = await cursor.lastrowid
            print(f"[INFO] 新增使用者成功: user_id={user_id}")
            return user_id
    except IntegrityError as e:
        if "Duplicate entry" in str(e) and "phone" in str(e):
            print(f"[ERROR] 帳號已被註冊: {phone}")
            raise AlreadyExistsError("帳號已被註冊")
        raise DatabaseError(f"資料庫完整性錯誤: {e}")
    except Exception as e:
        print(f"[ERROR] 新增使用者失敗: {e}")
        raise DatabaseError(f"新增使用者失敗: {e}")

async def update_user(conn, user_id: int, **kwargs) -> bool:
    """
    更新使用者資料。允許更新 name、phone、password、role_id、line_id。
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
        async with conn.cursor() as cursor:
            await cursor.execute(query, tuple(values))
            await conn.commit()
            if cursor.rowcount == 0:
                raise NotFoundError(f"找不到 user_id={user_id} 的使用者可更新")
            return True
    except NotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] 更新使用者失敗: {e}")
        raise DatabaseError(f"更新使用者失敗: {e}")

async def select_user_by_phone(conn, phone: str) -> Dict[str, Any]:
    """
    根據 phone 查詢使用者資料。
    """
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = "SELECT * FROM users WHERE phone = %s"
            await cursor.execute(query, (phone,))
            result = await cursor.fetchone()
            if not result:
                raise NotFoundError(f"找不到 phone={phone} 的使用者")
            return result
    except NotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] 查詢使用者資料失敗: {e}")
        raise DatabaseError(f"查詢使用者資料失敗: {e}")
    
async def select_user_by_id(conn, user_id: int) -> Dict[str, Any]:
    """
    根據 user_id 查詢使用者資料。
    """
    try:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            query = "SELECT * FROM users WHERE user_id = %s"
            await cursor.execute(query, (user_id,))
            result = await cursor.fetchone()
            if not result:
                raise NotFoundError(f"找不到 user_id={user_id} 的使用者")
            return result
    except NotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] 查詢使用者資料失敗: {e}")
        raise DatabaseError(f"查詢使用者資料失敗: {e}")

async def delete_user(conn, phone: str) -> bool:
    """
    根據 phone 刪除使用者帳號。
    """
    try:
        async with conn.cursor() as cursor:
            query = "DELETE FROM users WHERE phone = %s"
            await cursor.execute(query, (phone,))
            await conn.commit()
            if cursor.rowcount == 0:
                raise NotFoundError(f"找不到 phone={phone} 的使用者可刪除")
            return True
    except NotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] 刪除使用者失敗: {e}")
        raise DatabaseError(f"刪除使用者失敗: {e}")