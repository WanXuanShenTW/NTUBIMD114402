import datetime
from typing import Any, Dict, Optional
from aiomysql.cursors import DictCursor
from pymysql.err import IntegrityError
from ..exceptions import DatabaseError, NotFoundError, AlreadyExistsError

async def insert_user(
    conn,
    name: str,
    phone: str,
    role_id: int,
    password: str,
    gender: str,
    address: str
) -> Optional[int]:
    """新增一筆使用者資料到資料庫"""
    async with conn.cursor() as cursor:
        try:
            query = """
                INSERT INTO users (name, phone, role_id, password, gender, created_at, address)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (name, phone, role_id, password, gender, datetime.datetime.now(), address)
            await cursor.execute(query, values)
            return cursor.lastrowid
        except IntegrityError as e:
            if "Duplicate entry" in str(e) and "phone" in str(e):
                raise AlreadyExistsError("此電話已被註冊")
            raise DatabaseError(f"資料庫完整性錯誤: {e}")
        except Exception as e:
            raise DatabaseError(f"新增使用者失敗: {e}")

async def update_user(conn, user_id: int, **kwargs) -> bool:
    """更新使用者資料"""
    allowed = {"name", "phone", "password", "role_id", "gender", "address"}
    fields, values = [], []
    for k, v in kwargs.items():
        if k in allowed:
            fields.append(f"{k} = %s")
            values.append(v)
    if not fields:
        return False

    values.append(user_id)
    query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s"

    async with conn.cursor() as cursor:
        await cursor.execute(query, tuple(values))
        if cursor.rowcount == 0:
            raise NotFoundError(f"找不到 user_id={user_id} 的使用者可更新")
        return True

async def select_user_by_phone(conn, phone: str) -> Dict[str, Any]:
    """依 phone 查詢使用者資料"""
    async with conn.cursor(DictCursor) as cursor:
        await cursor.execute("SELECT * FROM users WHERE phone = %s", (phone,))
        row = await cursor.fetchone()
        if not row:
            raise NotFoundError(f"找不到 phone={phone} 的使用者")
        return row

async def select_user_by_id(conn, user_id: int) -> Dict[str, Any]:
    """依 user_id 查詢使用者資料"""
    async with conn.cursor(DictCursor) as cursor:
        await cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
        row = await cursor.fetchone()
        if not row:
            raise NotFoundError(f"找不到 user_id={user_id} 的使用者")
        return row

async def delete_user(conn, phone: str) -> bool:
    """依 phone 刪除使用者"""
    async with conn.cursor() as cursor:
        await cursor.execute("DELETE FROM users WHERE phone = %s", (phone,))
        if cursor.rowcount == 0:
            raise NotFoundError(f"找不到 phone={phone} 的使用者可刪除")
        return True

async def get_user_auth_by_id(conn, user_id: int) -> Optional[Dict[str, Any]]:
    """取使用者的密碼"""
    async with conn.cursor(DictCursor) as cursor:
        await cursor.execute("SELECT password FROM users WHERE user_id=%s LIMIT 1", (user_id,))
        return await cursor.fetchone()