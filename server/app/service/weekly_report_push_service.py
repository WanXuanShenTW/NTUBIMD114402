from typing import List, Dict, Any, Optional
from datetime import datetime, time
from ..db import Database
from ..dao.weekly_interaction_reports_dao import select_latest_report_by_elder_id
from ..dao.notify_line_dao import list_caregiver_line_uids_by_elder
from ..dao.weekly_report_push_dao import get_users_for_weekly_push_by_time, get_all_elders_with_recent_reports
from ..service.notify_line_service import push_text_bulk
from ..exceptions import NotFoundError, DatabaseError
import asyncio

async def push_weekly_reports_by_time(hour: int = None, minute: int = None, weekday: int = None) -> Dict[str, Any]:
    """根據指定時間推播週報"""
    now = datetime.now()
    target_hour = hour if hour is not None else now.hour
    target_minute = minute if minute is not None else now.minute
    target_weekday = weekday if weekday is not None else now.weekday() + 1  # 轉換為1-7格式
    
    # 將週日從7轉為0 (配合資料庫格式)
    if target_weekday == 7:
        target_weekday = 0
    
    try:
        async with Database.connection() as conn:
            # 獲取該時間應該推播的用戶
            users = await get_users_for_weekly_push_by_time(conn, target_hour, target_minute, target_weekday)
            
            if not users:
                return {
                    "status": "no_users",
                    "message": f"沒有用戶設定在 週{target_weekday} {target_hour:02d}:{target_minute:02d} 推播",
                    "target_time": f"{target_hour:02d}:{target_minute:02d}",
                    "target_weekday": target_weekday,
                    "total_users": 0,
                    "results": []
                }
            
            # 按elder_id分組，避免重複推播
            elder_to_users = {}
            for user in users:
                elder_id = user["elder_id"]
                if elder_id not in elder_to_users:
                    elder_to_users[elder_id] = []
                elder_to_users[elder_id].append(user)
            
            # 為每個長者推播週報
            results = []
            for elder_id, user_list in elder_to_users.items():
                try:
                    # 獲取週報
                    report = await select_latest_report_by_elder_id(conn, elder_id)
                    
                    # 格式化訊息
                    message = await format_weekly_report_message(report)
                    
                    # 推播給該長者的所有照護者 (在此時間設定推播的)
                    line_uids = [user["line_user_id"] for user in user_list]
                    push_result = await push_text_bulk(line_uids, message)
                    
                    results.append({
                        "elder_id": elder_id,
                        "status": "success",
                        "report_period": f"{report.get('start_date')} ~ {report.get('end_date')}",
                        "target_users": len(user_list),
                        **push_result
                    })
                    
                except NotFoundError:
                    results.append({
                        "elder_id": elder_id,
                        "status": "no_report",
                        "message": "找不到最新週報",
                        "target_users": len(user_list),
                        "sent": [],
                        "failed": [user["line_user_id"] for user in user_list]
                    })
                except Exception as e:
                    results.append({
                        "elder_id": elder_id,
                        "status": "error", 
                        "message": f"推播失敗: {str(e)}",
                        "target_users": len(user_list),
                        "sent": [],
                        "failed": [user["line_user_id"] for user in user_list]
                    })
            
            # 統計結果
            success_count = sum(1 for r in results if r.get("status") == "success")
            total_sent = sum(len(r.get("sent", [])) for r in results)
            total_failed = sum(len(r.get("failed", [])) for r in results)
            
            return {
                "status": "completed",
                "message": f"定時推播完成，成功推播 {success_count}/{len(elder_to_users)} 個長者的週報",
                "target_time": f"{target_hour:02d}:{target_minute:02d}",
                "target_weekday": target_weekday,
                "total_users": len(users),
                "total_elders": len(elder_to_users),
                "success_count": success_count,
                "total_sent": total_sent,
                "total_failed": total_failed,
                "results": results
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"定時推播失敗: {str(e)}",
            "target_time": f"{target_hour:02d}:{target_minute:02d}",
            "target_weekday": target_weekday,
            "total_users": 0,
            "results": []
        }

async def push_weekly_reports_batch() -> Dict[str, Any]:
    """批量推播週報給所有長者的照護者"""
    try:
        async with Database.connection() as conn:
            # 獲取所有有週報的長者
            elders = await get_all_elders_with_recent_reports(conn)
            
            if not elders:
                return {
                    "status": "no_data",
                    "message": "沒有找到需要推播的週報",
                    "total_elders": 0,
                    "results": []
                }
            
            # 並行推播所有長者的週報
            tasks = [push_weekly_report_to_elder(elder["elder_id"]) for elder in elders]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 統計結果
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
            total_sent = sum(len(r.get("sent", [])) for r in results if isinstance(r, dict))
            total_failed = sum(len(r.get("failed", [])) for r in results if isinstance(r, dict))
            
            return {
                "status": "completed",
                "message": f"批量推播完成，成功推播 {success_count}/{len(elders)} 個長者的週報",
                "total_elders": len(elders),
                "success_count": success_count,
                "total_sent": total_sent,
                "total_failed": total_failed,
                "results": [r for r in results if isinstance(r, dict)]
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"批量推播失敗: {str(e)}",
            "total_elders": 0,
            "results": []
        }
    """獲取所有有週報的長者列表"""
    async with Database.connection() as conn:
        async with conn.cursor() as cursor:
            try:
                # 獲取所有有最新週報的長者
                query = """
                    SELECT DISTINCT elder_id
                    FROM weekly_interaction_reports 
                    WHERE DATE(end_date) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
                    ORDER BY elder_id
                """
                await cursor.execute(query)
                rows = await cursor.fetchall()
                return [{"elder_id": row[0]} for row in rows]
            except Exception as e:
                raise DatabaseError(f"獲取長者列表失敗: {e}")

async def format_weekly_report_message(report: Dict[str, Any], elder_name: str = None) -> str:
    """格式化週報推播訊息"""
    start_date = report.get("start_date", "未知")
    end_date = report.get("end_date", "未知")
    analysis_result = report.get("analysis_result", "無分析結果")
    
    # 處理長者姓名
    elder_display = elder_name if elder_name else f"長者 (ID: {report.get('elder_id', '未知')})"
    
    # 限制文字長度，LINE訊息有限制
    if len(analysis_result) > 1300:
        analysis_result = analysis_result[:1300] + "...\n\n(完整報告請至App查看)"
    
    message = f"""📊 【{elder_display}的週報】

📅 報告期間：{start_date} ~ {end_date}

{analysis_result}
"""
    
    return message

async def push_weekly_report_to_elder(elder_id: int) -> Dict[str, Any]:
    """推播指定長者的週報給其照護者"""
    try:
        async with Database.connection() as conn:
            # 獲取最新週報
            report = await select_latest_report_by_elder_id(conn, elder_id)
            
            # 獲取照護者的LINE UIDs
            uids, source = await list_caregiver_line_uids_by_elder(conn, elder_id)
            
            if not uids:
                return {
                    "elder_id": elder_id,
                    "status": "no_recipients",
                    "message": "無照護者LINE綁定",
                    "sent": [],
                    "failed": []
                }
            
            # 格式化週報訊息
            message = await format_weekly_report_message(report)
            
            # 推播給所有照護者
            result = await push_text_bulk(uids, message)
            
            return {
                "elder_id": elder_id,
                "status": "success",
                "message": "週報推播完成",
                "report_period": f"{report.get('start_date')} ~ {report.get('end_date')}",
                "recipients_count": len(uids),
                **result
            }
            
    except NotFoundError:
        return {
            "elder_id": elder_id,
            "status": "no_report",
            "message": "找不到最新週報",
            "sent": [],
            "failed": []
        }
    except Exception as e:
        return {
            "elder_id": elder_id,
            "status": "error",
            "message": f"推播失敗: {str(e)}",
            "sent": [],
            "failed": []
        }

async def push_weekly_reports_batch() -> Dict[str, Any]:
    """批量推播週報給所有長者的照護者"""
    try:
        async with Database.connection() as conn:
            # 獲取所有有週報的長者
            elders = await get_all_elders_with_recent_reports(conn)
            
            if not elders:
                return {
                    "status": "no_data",
                    "message": "沒有找到需要推播的週報",
                    "total_elders": 0,
                    "results": []
                }
            
            # 並行推播所有長者的週報
            tasks = [push_weekly_report_to_elder(elder["elder_id"]) for elder in elders]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 統計結果
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
            total_sent = sum(len(r.get("sent", [])) for r in results if isinstance(r, dict))
            total_failed = sum(len(r.get("failed", [])) for r in results if isinstance(r, dict))
            
            return {
                "status": "completed",
                "message": f"批量推播完成，成功推播 {success_count}/{len(elders)} 個長者的週報",
                "total_elders": len(elders),
                "success_count": success_count,
                "total_sent": total_sent,
                "total_failed": total_failed,
                "results": [r for r in results if isinstance(r, dict)]
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"批量推播失敗: {str(e)}",
            "total_elders": 0,
            "results": []
        }