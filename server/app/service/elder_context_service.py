from typing import Dict, Any, Optional
from ..db import Database
from ..dao.line_binding_dao import get_user_by_line_user_id
from ..dao.caregiver_elder_dao import get_elder_by_caregiver_user_id
from ..dao.users_dao import select_user_by_id

async def resolve_current_elder(line_user_id: str) -> Optional[Dict[str, Any]]:
    """
    解析當前照護者對應的長者資訊
    ✅ 修正：Service 層不寫 SQL，只調用 DAO
    """
    try:
        async with Database.connection() as conn:
            # 1. 通過 DAO 獲取用戶資訊
            user = await get_user_by_line_user_id(conn, line_user_id)
            if not user:
                return None
            
            caregiver_user_id = user["user_id"]
            
            # 2. 通過 DAO 獲取長者關係
            elder_relation = await get_elder_by_caregiver_user_id(conn, caregiver_user_id)
            if not elder_relation:
                return None
            
            elder_user_id = elder_relation["elder_user_id"]
            
            # 3. 通過 DAO 獲取長者詳細資訊
            elder_user = await select_user_by_id(conn, elder_user_id)
            if not elder_user:
                return None
            
            return {
                "elder_id": elder_user["user_id"],
                "elder_name": elder_user["name"],
                "elder_phone": elder_user["phone"],
                "elder_gender": elder_user["gender"],
                "elder_address": elder_user["address"],
                "relationship": elder_relation.get("relationship", "")
            }
            
    except Exception as e:
        print(f"解析長者資訊時發生錯誤: {e}")
        return None

def format_elder_info(elder_info: Dict[str, Any]) -> str:
    """格式化長者資訊顯示 - 純業務邏輯，不是 async"""
    if not elder_info:
        return "❌ 目前沒有照護的長者"
    
    name = elder_info.get("elder_name", "未知")
    phone = elder_info.get("elder_phone", "未提供")
    gender = elder_info.get("elder_gender", "未提供")
    address = elder_info.get("elder_address", "未提供")
    
    # 格式化性別顯示
    gender_display = {
        'M': '男', 'F': '女',
        'male': '男', 'female': '女'
    }.get(gender, gender if gender else "未提供")
    
    return f"""{name}的個人資訊：
姓名：{name}
電話：{phone}
性別：{gender_display}
住址：{address}"""