import datetime
from typing import Any, Dict, Optional
from aiomysql.cursors import DictCursor
from pymysql.err import IntegrityError
from ..exceptions import DatabaseError, NotFoundError, AlreadyExistsError

async def select_latest_report_by_elder_id(conn, elder_id: int) -> Dict[str, Any]:
    """依 elder_id 查找最新的 weekly_interaction_report 資料"""
    async with conn.cursor(DictCursor) as cursor:
        try:
            query = """
                SELECT * 
                FROM weekly_interaction_reports 
                WHERE elder_id = %s 
                ORDER BY end_date DESC 
                LIMIT 1
            """
            await cursor.execute(query, (elder_id,))
            row = await cursor.fetchone()
            if not row:
                raise NotFoundError(f"找不到 elder_id={elder_id} 的最新報告")
            return row
        except Exception as e:
            raise DatabaseError(f"查詢最新報告失敗: {e}")