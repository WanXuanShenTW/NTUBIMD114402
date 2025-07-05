def insert_emergency_contacts(conn, user_id: int, contact_id: int, priority: int, relationship: str) -> bool:
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
        print(f"[ERROR] 新增照護關係失敗: {e}")
        return False

def select_contacts_by_user_id(conn, user_id: int):
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM emergency_contacts WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] 查詢照護關係失敗: {e}")
        return []

def select_contact_by_pair(conn, user_id: int, contact_id: int):
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM emergency_contacts WHERE user_id = %s AND contact_id = %s"
        cursor.execute(query, (user_id, contact_id))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] 查詢照護組合失敗: {e}")
        return None

def delete_contact(conn, user_id: int, contact_id: int) -> bool:
    try:
        cursor = conn.cursor()
        query = "DELETE FROM emergency_contacts WHERE user_id = %s AND contact_id = %s"
        cursor.execute(query, (user_id, contact_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[ERROR] 刪除照護關係失敗: {e}")
        return False
