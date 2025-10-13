import datetime
from typing import List, Dict, Any
from aiomysql.cursors import DictCursor
from ..exceptions import DatabaseError

async def select_interactions_by_elder_and_date(
    conn,
    elder_id: int,
    date_str: str,
) -> List[Dict[str, Any]]:
    """
    依 elder_id 與日期(YYYY-MM-DD) 取得當天的互動紀錄
    回傳欄位：session_id, elder_id, conversation_history, start_at
    """
    query = """
        SELECT session_id, elder_id, conversation_history, start_at
        FROM interactions_records
        WHERE elder_id = %s
          AND DATE(start_at) = %s
        ORDER BY start_at ASC, session_id ASC
    """
    try:
        async with conn.cursor(DictCursor) as cursor:
            await cursor.execute(query, (elder_id, date_str))
            rows = await cursor.fetchall()
            return rows or []
    except Exception as e:
        raise DatabaseError(f"查詢互動紀錄失敗: {e}")
