import os
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

# ---- LINE SDK v3 ----
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    FollowEvent,
    UnfollowEvent,
    PostbackEvent
)
from linebot.v3.messaging import (
    AsyncApiClient,
    AsyncMessagingApi,
    Configuration,
    ReplyMessageRequest,
    TextMessage,
    MessageAction,
)
from linebot.v3.messaging.models import (
    PostbackAction,
    QuickReply,
    QuickReplyItem
)

# ---- services ----
from app.service.line_auth_service import (
    need_binding,
    is_waiting_password,
    start_phone_step,
    confirm_password_step,
)
from app.service.elder_context_service import resolve_current_elder
from app.service.line_binding_service import (
    unbind_line_user,
    get_user_by_line_uid,
)
from app.service.notify_line_service import notify_from_payload
from app.service.line_richmenu_service import (
    ensure_default_richmenu,
    ensure_user_linked_richmenu,
    reset_all_and_setup,
)

from app.service.notify_prefs_service import (
    parse_sel, toggle_day, human_days, save_prefs_by_line_uid
)

router = APIRouter(prefix="/line", tags=["LINE"])

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# LINE API client
config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = AsyncApiClient(config)
line_api = AsyncMessagingApi(api_client)
parser = WebhookParser(CHANNEL_SECRET)

ALLOWED_CMDS = {"個人資訊", "推播設定", "解除綁定"}

# 追蹤使用者推播設定狀態
USER_PREFS_STATE = {}  # key: line_user_id -> {"step": "waiting_time", "selected_days": [1,2,3]}

def validate_time_format(time_str: str) -> tuple[bool, str, int, int]:
    """
    驗證時間格式 HHMM (24小時制)
    返回: (是否有效, 錯誤訊息, 小時, 分鐘)
    """
    try:
        time_str = time_str.strip()
        
        # 檢查是否為4位數字
        if len(time_str) != 4 or not time_str.isdigit():
            return False, "請輸入正確格式", 0, 0
        
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        
        # 接受0-23小時制
        if hour < 0 or hour > 23:
            return False, "請輸入正確格式", 0, 0
        
        if minute < 0 or minute > 59:
            return False, "請輸入正確格式", 0, 0
            
        return True, "", hour, minute
        
    except (ValueError, IndexError):
        return False, "請輸入正確格式", 0, 0

@router.post("/richmenu/reset")
async def reset_richmenu():
    import httpx
    try:
        rid = await reset_all_and_setup(line_api)
        return {"status": "ok", "rich_menu_id": rid, "action": "reset_and_setup"}
    except FileNotFoundError as e:
        msg = str(e)
        if msg.startswith("IMAGE_NOT_FOUND::"):
            path = msg.split("::", 1)[1]
            raise HTTPException(status_code=400, detail=f"找不到底圖：{path}。請放 2500x1686 PNG/JPG，或設 RICHMENU_IMG/LINE_RICHMENU_IMG")
        raise HTTPException(status_code=400, detail=msg)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        raise HTTPException(status_code=status if 400 <= status < 500 else 502,
                            detail=f"LINE API 錯誤（{status}）：{e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RM-RESET-UNEXPECTED：{e}")
    
@router.get("/webhook")
async def health():
    return "OK"

class NotifyLinePayload(BaseModel):
    elder_id: Optional[int] = None
    status: Optional[str] = None
    message: Optional[str] = None
    detected_at: Optional[str] = None

@router.post("/notify_line")
async def notify_line(payload: NotifyLinePayload):
    try:
        return await notify_from_payload(
            elder_id=payload.elder_id,
            status=payload.status,
            message=payload.message,
            detected_at=payload.detected_at,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {e}")

@router.post("/richmenu/setup")
async def setup_richmenu():
    import httpx, os
    if not os.getenv("LINE_CHANNEL_ACCESS_TOKEN"):
        raise HTTPException(status_code=400, detail="缺少 LINE_CHANNEL_ACCESS_TOKEN（Messaging API 長期 Token）")
    if not os.getenv("LINE_CHANNEL_SECRET"):
        raise HTTPException(status_code=400, detail="缺少 LINE_CHANNEL_SECRET")

    try:
        rid = await ensure_default_richmenu(line_api)
        return {"status": "ok", "rich_menu_id": rid}
    except FileNotFoundError as e:
        msg = str(e)
        if msg.startswith("IMAGE_NOT_FOUND::"):
            path = msg.split("::", 1)[1]
            raise HTTPException(status_code=400, detail=f"找不到底圖：{path}。請放 2500x1686 PNG/JPG，或設 LINE_RICHMENU_IMG")
        raise HTTPException(status_code=400, detail=msg)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        raise HTTPException(status_code=status if 400 <= status < 500 else 502,
                            detail=f"LINE API 錯誤（{status}）：{e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RM-SETUP-UNEXPECTED：{e}")

# ---- Webhook ----
@router.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
    except InvalidSignatureError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, PostbackEvent):
            await handle_postback(event)
            continue

    for event in events:
        # 1) 新增好友
        if isinstance(event, FollowEvent):
            uid = event.source.user_id
            try:
                await ensure_default_richmenu(line_api)
                await ensure_user_linked_richmenu(line_api, uid)

                hello = "照護者"
                user = await get_user_by_line_uid(uid)
                if user:
                    hello = f"{user.get('name', '照護者')}"
                    elder = await resolve_current_elder(uid)
                    if elder:
                        text = f"目前服務對象：{elder['elder_name']}（ID: {elder['elder_id']}）"
                    else:
                        text = "認證完成，但尚未指派被照護者，請先在 App 建立關係。"
                    await line_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=text)],
                        )
                    )
                else:
                    await line_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(
                                text=f"Hi {hello}！未綁定者請輸入手機號碼；已綁定可輸入「個人資訊」「我的長者」「解除綁定」。",
                            )]
                        )
                    )
            except Exception:
                # 避免 webhook 500
                await line_api.reply_message(
                    ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text="系統忙碌中，請稍後再試。")])
                )
            continue

        # 2) 取消好友(避免殭屍帳號)
        if isinstance(event, UnfollowEvent):
            try:
                await unbind_line_user(event.source.user_id)
            except Exception as e:
                print(f"[unfollow] unbind error: {e}")
            continue

        # 3) 訊息事件
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
            uid = event.source.user_id
            text = (event.message.text or "").strip()

            # 3-1) 正在等待密碼 → 繼續驗證流程
            if is_waiting_password(uid):
                ok, msg, _user = await confirm_password_step(uid, text)
                if not ok:
                    await line_api.reply_message(
                        ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=msg)])
                    )
                    continue

                elder = await resolve_current_elder(uid)
                if elder:
                    msgs = [TextMessage(
                        text=f"認證成功！目前服務對象：{elder['elder_name']}（ID: {elder['elder_id']}）"
                    )]
                else:
                    msgs = [TextMessage(text="認證完成，但尚未指派被照護者，請先在 App 建立關係。")]
                await line_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=msgs))
                continue

            # 3-2) 尚未綁定 → 先走手機號碼步驟
            if await need_binding(uid):
                ok, msg = await start_phone_step(uid, text)
                await line_api.reply_message(
                    ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=msg)])
                )
                continue

            # 3-3) 已綁定 → 僅接受白名單指令
            if text in ALLOWED_CMDS:
                if text == "個人資訊":
                    try:
                        # 獲取當前照護的長者資訊
                        elder_info = await resolve_current_elder(uid)
                        if elder_info:
                            # 格式化個人資訊
                            name = elder_info.get("elder_name", "未提供")
                            phone = elder_info.get("elder_phone", "未提供") 
                            gender = elder_info.get("elder_gender", "未提供")
                            address = elder_info.get("elder_address", "未提供")
                            
                            # 格式化性別顯示
                            if gender in ['M', 'male', '男']:
                                gender_display = '男'
                            elif gender in ['F', 'female', '女']:
                                gender_display = '女'
                            else:
                                gender_display = gender if gender else "未提供"
                            
                            # 避免電話號碼被識別為連結，加入空格或其他符號
                            if phone and phone != "未提供":
                                # 將電話號碼格式化，避免自動連結
                                formatted_phone = phone.replace("-", "").strip()
                                if len(formatted_phone) == 10 and formatted_phone.startswith("09"):
                                    # 格式：09XX-XXX-XXX
                                    formatted_phone = f"{formatted_phone[:4]}-{formatted_phone[4:7]}-{formatted_phone[7:]}"
                                else:
                                    # 在數字間加入空格避免自動識別
                                    formatted_phone = " ".join(formatted_phone)
                            else:
                                formatted_phone = phone
                            
                            # 格式化輸出文字
                            formatted_info = f"""{name}的個人資訊：
姓名：{name}
電話：{formatted_phone}
性別：{gender_display}
住址：{address}"""
                            
                            await line_api.reply_message(ReplyMessageRequest(
                                reply_token=event.reply_token,
                                messages=[TextMessage(text=formatted_info)]
                            ))
                        else:
                            await line_api.reply_message(ReplyMessageRequest(
                                reply_token=event.reply_token,
                                messages=[TextMessage(text="❌ 目前沒有照護的長者資訊，請確認是否已正確綁定帳號")]
                            ))
                    except Exception as e:
                        print(f"查詢個人資訊時發生錯誤: {e}")
                        await line_api.reply_message(ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text="❌ 查詢個人資訊時發生錯誤，請稍後再試或聯繫系統管理員")]
                        ))
                    continue

                if text == "推播設定":
                    await line_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(
                            text="請選擇每週要通知的『星期』：",
                            quick_reply=build_weekday_qr([])
                        )]
                    ))
                    continue

                if text == "解除綁定":
                    n = await unbind_line_user(uid)
                    msg = "已解除綁定。" if n > 0 else "目前沒有綁定資訊。"
                    await line_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=msg)]
                    ))
                    continue

            # 3-4) 檢查是否正在等待時間輸入
            user_state = USER_PREFS_STATE.get(uid)
            if user_state and user_state.get("step") == "waiting_time":
                # 檢查是否要取消
                if text == "取消":
                    # 清除使用者狀態
                    USER_PREFS_STATE.pop(uid, None)
                    await line_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text="已取消設定。", quick_reply=quick_reply_main())]
                    ))
                    continue
                
                # 使用者正在輸入時間
                is_valid, error_msg, hour, minute = validate_time_format(text)
                
                if not is_valid:
                    await line_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=error_msg)]
                    ))
                    continue
                
                # 時間格式正確，保存設定
                selected_days = user_state["selected_days"]
                ok, msg = await save_prefs_by_line_uid(uid, selected_days, hour, minute)
                
                # 清除使用者狀態
                USER_PREFS_STATE.pop(uid, None)
                
                await line_api.reply_message(ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=msg, quick_reply=quick_reply_main())]
                ))
                continue

            # 3-5) 其它任何輸入 → 無效的操作
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(
                    text="無效的操作。請從底部選單或快速回覆中選擇「個人資訊」「推播設定」「解除綁定」。"
                )]
            ))
            continue

    return "OK"

@router.get("/richmenu/debug")
async def richmenu_debug():
    import httpx
    from app.service.line_richmenu_service import _get_token_from_api

    token = _get_token_from_api(line_api)
    fp = f"{token[:6]}...{token[-6:]}" if len(token) > 12 else token

    r = await line_api.get_rich_menu_list()
    menus = getattr(r, "richmenus", None) or []

    out = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for m in menus:
            rid = m.rich_menu_id
            url = f"https://api-data.line.me/v2/bot/richmenu/{rid}/content"
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            out.append({
                "name": m.name,
                "rich_menu_id": rid,
                "content_get_status": resp.status_code,
            })

    return {"token_fingerprint": fp, "count": len(menus), "menus": out}

#-------------------------------
def quick_reply_main():
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="個人資訊", text="個人資訊")),
        QuickReplyItem(action=MessageAction(label="推播設定", text="推播設定")),
        QuickReplyItem(action=MessageAction(label="解除綁定", text="解除綁定")),
    ])

# 選星期（單選）
def build_weekday_qr(sel_days: list[int]):
    names = ["日","一","二","三","四","五","六"]
    items = []
    for d, name in enumerate(names):
        items.append(QuickReplyItem(
            action=PostbackAction(
                label=f"週{name}",
                data=f"prefs:weekday_single:{d}",
                display_text=f"週{name}",
            )
        ))
    items.append(QuickReplyItem(action=PostbackAction(label="取消", data="prefs:cancel", display_text="取消")))
    return QuickReply(items=items)

async def handle_postback(event: PostbackEvent):
    uid = event.source.user_id
    raw = event.postback.data or ""

    # data 範例：
    #   prefs:weekday:3|sel=1,3
    #   prefs:done|sel=1,3
    #   prefs:time:08:00|sel=1,3
    parts = raw.split("|")
    key = parts[0]
    sel_csv = ""
    for p in parts[1:]:
        if p.startswith("sel="):
            sel_csv = p[4:]
    sel_days = parse_sel(sel_csv)

    try:
        # 處理單選星期
        if key.startswith("prefs:weekday_single:"):
            d = int(key.split(":")[2])
            selected_days = [d]  # 單選，只有一個星期
            
            # 儲存使用者狀態，等待時間輸入
            USER_PREFS_STATE[uid] = {
                "step": "waiting_time",
                "selected_days": selected_days
            }
            
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(
                    text=f"已選：{human_days(selected_days)}\n請輸入通知時間（24小時制，格式：HHMM）\n例如：0830、1400、0015，若要退出請輸入取消"
                )]
            ))
            return

        # 舊的多選邏輯（保留以防萬一）
        if key.startswith("prefs:weekday:"):
            d = int(key.split(":")[2])
            new_days = toggle_day(sel_days, d)
            msg = f"已選：{human_days(new_days)}\n繼續勾選或按「完成」。"
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=msg, quick_reply=build_weekday_qr(new_days))]
            ))
            return

        if key == "prefs:cancel":
            # 清除使用者狀態
            USER_PREFS_STATE.pop(uid, None)
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="已取消設定。", quick_reply=quick_reply_main())]
            ))
            return

    except Exception as ex:
        print(f"[prefs postback] {ex}")
        await line_api.reply_message(ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text="系統忙碌中，請稍後再試。")]
        ))