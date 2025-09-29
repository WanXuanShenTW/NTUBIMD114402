# app/routes/line_routes.py
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
)
from linebot.v3.messaging import (
    AsyncApiClient,
    AsyncMessagingApi,
    Configuration,
    ReplyMessageRequest,
    TextMessage,
    MessageAction,
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

router = APIRouter(prefix="/line", tags=["LINE"])

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# LINE API client
config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = AsyncApiClient(config)
line_api = AsyncMessagingApi(api_client)
parser = WebhookParser(CHANNEL_SECRET)

ALLOWED_CMDS = {"個人資訊", "我的長者", "解除綁定"}

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
                            
                            # 格式化輸出文字
                            formatted_info = f"""{name}的個人資訊：
姓名：{name}
電話：{phone}
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

                if text == "我的長者":
                    elder = await resolve_current_elder(uid)
                    if elder:
                        msg = f"目前服務對象：{elder['elder_name']}（ID: {elder['elder_id']}）"
                    else:
                        msg = "目前沒有指派服務的長者。"
                    await line_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=msg)]
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

            # 3-4) 其它任何輸入 → 無效的操作
            await line_api.reply_message(ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(
                    text="無效的操作。請從底部選單或快速回覆中選擇「個人資訊」「我的長者」「解除綁定」。"
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
