import datetime
from datetime import datetime, timedelta
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

async def select_reports_by_week(conn, elder_id: int, date: datetime, sunday_as_first_day: bool) -> list[Dict[str, Any]]:
    """依 elder_id 和日期查找該週的 weekly_interaction_report 資料"""
    async with conn.cursor(DictCursor) as cursor:
        try:
            # 計算該週的開始日期和結束日期
            weekday = date.weekday()
            if sunday_as_first_day:
                start_date = date - timedelta(days=weekday + 1)
            else:
                start_date = date - timedelta(days=weekday)
            end_date = start_date + timedelta(days=6)

            query = """
                SELECT * 
                FROM weekly_interaction_reports 
                WHERE elder_id = %s 
                AND end_date BETWEEN %s AND %s
                ORDER BY end_date DESC
            """
            await cursor.execute(query, (elder_id, start_date, end_date))
            rows = await cursor.fetchall()
            if not rows:
                raise NotFoundError(f"找不到 elder_id={elder_id} 在 {start_date} 至 {end_date} 的報告")
            return rows
        except Exception as e:
            raise DatabaseError(f"查詢該週報告失敗: {e}")