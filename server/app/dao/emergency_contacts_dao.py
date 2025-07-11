from ..exceptions import DatabaseError, NotFoundError, AlreadyExistsError

def insert_emergency_contacts(conn, user_id: int, contact_id: int, priority: int, relationship: str) -> bool:
    cursor = None
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO emergency_contacts (user_id, contact_id, priority, relationship)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (user_id, contact_id, priority, relationship))
        conn.commit()
        return True
    except Exception as e:
        # 若有唯一鍵衝突，可根據實際錯誤型別改用 AlreadyExistsError
        raise DatabaseError(f"新增照護關係失敗: {e}")
    finally:
        if cursor:
            cursor.close()

def select_contacts_by_user_id(conn, user_id: int):
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM emergency_contacts WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        results = cursor.fetchall()
        if not results:
            raise NotFoundError(f"找不到 user_id={user_id} 的照護關係")
        return results
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢照護關係失敗: {e}")
    finally:
        if cursor:
            cursor.close()

def select_contact_by_pair(conn, user_id: int, contact_id: int):
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM emergency_contacts WHERE user_id = %s AND contact_id = %s"
        cursor.execute(query, (user_id, contact_id))
        result = cursor.fetchone()
        if not result:
            raise NotFoundError(f"找不到 user_id={user_id} 與 contact_id={contact_id} 的照護組合")
        return result
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"查詢照護組合失敗: {e}")
    finally:
        if cursor:
            cursor.close()

def delete_contact(conn, user_id: int, contact_id: int) -> bool:
    cursor = None
    try:
        cursor = conn.cursor()
        query = "DELETE FROM emergency_contacts WHERE user_id = %s AND contact_id = %s"
        cursor.execute(query, (user_id, contact_id))
        conn.commit()
        if cursor.rowcount == 0:
            raise NotFoundError(f"找不到 user_id={user_id} 與 contact_id={contact_id} 的照護關係可刪除")
        return True
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(f"刪除照護關係失敗: {e}")
    finally:
        if cursor:
            cursor.close()