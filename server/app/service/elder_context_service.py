# app/service/elder_context_service.py
from typing import Optional, Dict, Any
from app.dao.line_binding_dao import get_user_by_line_user_id
from app.dao.caregiver_elder_dao import (
    get_elder_by_caregiver_user_id,
    get_elder_by_reversed_mapping,
)

async def resolve_current_elder(line_user_id: str) -> Optional[Dict[str, Any]]:
    """
    由 LINE 使用者（照護者）找出「目前服務對象（長者）」。
    流程：
      1) 先找這個 LINE 綁定的是哪個 users.user_id
      2) 先用『正向』查 emergency_contacts（照護者→長者）
      3) 找不到再用『反向』容錯（把表裡放反了的情況）
    回傳：
      { "elder_id": int, "elder_name": str } 或 None
    """
    bound = await get_user_by_line_user_id(line_user_id)
    if not bound:
        return None

    caregiver_id = int(bound["user_id"])

    # 先正向
    elder = await get_elder_by_caregiver_user_id(caregiver_id)
    if elder:
        return elder

    # 再反向（容錯）
    elder = await get_elder_by_reversed_mapping(caregiver_id)
    if elder:
        return elder

    return None
