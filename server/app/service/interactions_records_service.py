import json
import re
from typing import Any, Dict, List
from ..db import Database
from ..exceptions import DatabaseError
from ..dao.interactions_records_dao import select_interactions_by_elder_and_date

_SANITIZE_RE = re.compile(r"[\r\n\t]+|\u200b|\u200c|\u200d|\ufeff")
_SPACE_RE = re.compile(r"\s+")

def _sanitize_text(text: Any) -> str:
    """
    清理文字：移除換行/Tab/零寬字元、壓縮連續空白、去除首尾空白。
    """
    if text is None:
        return ""
    s = str(text)
    s = _SANITIZE_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s)
    return s.strip()

def _parse_pairs(conversation_history: Any) -> List[Dict[str, str]]:
    """
    將 conversation_history(可能是JSON字串或物件) 解析成 [{Q, A}, ...]
    """
    if conversation_history is None:
        return []

    # MySQL JSON 欄位在某些情況會以 str 形式回來
    if isinstance(conversation_history, str):
        try:
            conversation = json.loads(conversation_history)
        except Exception:
            # 若不是合法JSON就當作一整段A文本處理
            return [{"Q": "", "A": _sanitize_text(conversation_history)}]
    else:
        conversation = conversation_history

    pairs: List[Dict[str, str]] = []
    # 期待格式：[{ "Q": "...", "A": "..."}, ...]
    if isinstance(conversation, list):
        for item in conversation:
            if isinstance(item, dict):
                q = _sanitize_text(item.get("Q", ""))
                a = _sanitize_text(item.get("A", ""))
                if q or a:
                    pairs.append({"Q": q, "A": a})
    elif isinstance(conversation, dict) and ("Q" in conversation or "A" in conversation):
        pairs.append({
            "Q": _sanitize_text(conversation.get("Q", "")),
            "A": _sanitize_text(conversation.get("A", "")),
        })
    else:
        # 其他不符合格式的情況，整段當作A
        pairs.append({"Q": "", "A": _sanitize_text(conversation)})

    return pairs

async def get_interactions_by_day(elder_id: int, date_str: str) -> Dict[str, Any]:
    """
    取得某位長者在特定「日」的所有聊天紀錄（分會話、含Q/A配對與文字清理）
    """
    async with Database.connection() as conn:
        rows = await select_interactions_by_elder_and_date(conn, elder_id, date_str)

    sessions = []
    total_pairs = 0
    for r in rows:
        pairs = _parse_pairs(r.get("conversation_history"))
        total_pairs += len(pairs)
        sessions.append({
            "session_id": r.get("session_id"),
            "start_at": r.get("start_at"),
            "pairs": pairs,
        })

    return {
        "elder_id": elder_id,
        "date": date_str,
        "total_sessions": len(sessions),
        "total_pairs": total_pairs,
        "sessions": sessions,
    }
