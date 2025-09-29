import os
import re
import time
from typing import Tuple, Optional, Dict, Any
from app.db import Database
from app.dao.users_dao import select_user_by_phone, get_user_auth_by_id
from app.dao.line_binding_dao import upsert_binding, get_user_by_line_user_id

PHONE_RE = re.compile(r"^09\d{8}$")

# 兩步驗證暫存（記憶體）
PENDING: Dict[str, Dict[str, Any]] = {}  # key: line_uid → {user_id, name, expires, attempts}
PENDING_TTL = int(os.getenv("LINE_BIND_TTL", 300))
MAX_ATTEMPTS = int(os.getenv("LINE_BIND_MAX_ATTEMPTS", 5))

try:
    import bcrypt  # type: ignore
    _HAS_BCRYPT = True
except Exception:
    _HAS_BCRYPT = False

def _now() -> int:
    return int(time.time())

async def need_binding(line_user_id: str) -> bool:
    async with Database.connection() as conn:
        return (await get_user_by_line_user_id(conn, line_user_id)) is None

def is_waiting_password(line_user_id: str) -> bool:
    st = PENDING.get(line_user_id)
    return bool(st and st.get("expires", 0) > _now())

async def start_phone_step(line_user_id: str, text: str) -> Tuple[bool, str]:
    """
    第一步：手機驗證（僅允許照護者綁定；照護者 role_id = 2）
    成功 → 進入輸入密碼步驟
    """
    if not PHONE_RE.match(text):
        return False, "請先輸入手機號碼（格式：09XXXXXXXX）。"

    async with Database.connection() as conn:
        user = await select_user_by_phone(conn, text)  # {user_id, name, role_id}
        if not user:
            return False, "查無此手機號碼。"

        # ✅ 只有 role_id = 2 才能綁定
        if int(user["role_id"]) != 2:
            return False, "此 Line Bot 僅提供『照護者』使用。請以照護者的手機號碼綁定。"

        PENDING[line_user_id] = {
            "user_id": user["user_id"],
            "name": user["name"],
            "expires": _now() + PENDING_TTL,
            "attempts": 0,
        }
        return True, "請輸入密碼以完成驗證。"

def _check_password(plain: str, stored: Optional[str]) -> bool:
    if not stored:
        return False
    if stored.startswith("$2") and _HAS_BCRYPT:
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), stored.encode("utf-8"))
        except Exception:
            return False
    return plain == stored

async def confirm_password_step(line_user_id: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    st = PENDING.get(line_user_id)
    if not st or st["expires"] <= _now():
        PENDING.pop(line_user_id, None)
        return False, "驗證逾時，請重新輸入手機號碼。", None

    user_id = st["user_id"]
    user_name = st["name"]
    attempts = st["attempts"]

    async with Database.connection() as conn:
        auth = await get_user_auth_by_id(conn, user_id)  # {password: "..."} 或 None
        if not auth:
            PENDING.pop(line_user_id, None)
            return False, "使用者尚未設定密碼，請洽系統管理員。", None

        if not _check_password(password, auth.get("password")):
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                PENDING.pop(line_user_id, None)
                return False, "密碼錯誤次數過多，請重新輸入手機號碼開始綁定。", None
            st["attempts"] = attempts
            return False, f"密碼錯誤，還可再嘗試 {MAX_ATTEMPTS - attempts} 次。", None

        # ✅ 密碼正確 → 綁定並回傳 user（role_id = 2）
        await upsert_binding(conn, user_id, line_user_id)
        PENDING.pop(line_user_id, None)
        user = {"user_id": user_id, "name": user_name, "role_id": 2}
        return True, f"✅ 認證成功！歡迎，{user_name}（照護者）。", user