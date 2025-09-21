# app/routes/line_routes.py
# -*- coding: utf-8 -*-

import os
from fastapi import APIRouter, Request, HTTPException

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
)
from linebot.v3.messaging.models import (
    ReplyMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    MessageAction,
)

# ---- services / dao ----
from app.service.line_auth_service import (
    need_binding,
    is_waiting_password,
    start_phone_step,
    confirm_password_step,
)
from app.service.elder_context_service import resolve_current_elder
from app.dao.line_binding_dao import get_user_by_line_user_id, unbind_by_line_user_id

router = APIRouter(prefix="/line", tags=["LINE"])

# ---- 初始化 v3 元件 ----
configuration = Configuration(access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
async_api_client = AsyncApiClient(configuration)
line_api = AsyncMessagingApi(async_api_client)
parser = WebhookParser(os.environ["LINE_CHANNEL_SECRET"])

# ---- UI ----
def quick_reply_main():
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="個人資訊", text="個人資訊")),
        QuickReplyItem(action=MessageAction(label="我的長者", text="我的長者")),
        QuickReplyItem(action=MessageAction(label="解除綁定", text="解除綁定")),
    ])

@router.get("/webhook")
async def health():
    return "OK"

@router.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = (await request.body()).decode("utf-8")
    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for e in events:
        if isinstance(e, FollowEvent):
            await handle_follow(e)
        elif isinstance(e, UnfollowEvent):
            await handle_unfollow(e)
        elif isinstance(e, MessageEvent) and isinstance(e.message, TextMessageContent):
            await handle_text(e)
    return "OK"

# ---- 事件處理 ----
async def handle_follow(event: FollowEvent):
    try:
        await line_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(
                    text="嗨～請先輸入你的手機號碼（照護者）。格式：09XXXXXXXX",
                )]
            )
        )
    except Exception as e:
        print(f"[follow] reply error: {e}")

async def handle_unfollow(event: UnfollowEvent):
    try:
        await unbind_by_line_user_id(event.source.user_id)
    except Exception as e:
        print(f"[unfollow] unbind error: {e}")

async def handle_text(event: MessageEvent):
    uid = event.source.user_id
    text = event.message.text.strip()

    try:
        # 1) 若正在等待密碼 → 當作密碼驗證
        if is_waiting_password(uid):
            ok, msg, user_info = await confirm_password_step(uid, text)
            if not ok:
                await line_api.reply_message(
                    ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=msg)])
                )
                return

            # 認證成功 → 歡迎 + 目前服務對象
            elder = await resolve_current_elder(uid)
            msgs = [
                TextMessage(text=msg, quick_reply=quick_reply_main())  # 來自 confirm_password_step 的歡迎詞
            ]
            if elder:
                msgs.append(TextMessage(
                    text=f"目前服務對象：{elder['elder_name']}（ID: {elder['elder_id']}）",
                    quick_reply=quick_reply_main(),
                ))
            else:
                msgs.append(TextMessage(text="認證完成，但尚未指派被照護者，請先在 App 建立關係。"))
            await line_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=msgs))
            return

        # 2) 尚未綁定 → 把輸入當「手機號碼步驟」
        if await need_binding(uid):
            ok, msg = await start_phone_step(uid, text)
            await line_api.reply_message(
                ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=msg)])
            )
            return

        # 3) 已綁定者指令
        if text in ("解除綁定", "unbind"):
            await unbind_by_line_user_id(uid)
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="已解除綁定。若要再次使用，請先輸入手機號碼。")]
                )
            )
            return

        if text in ("我的長者",):
            elder = await resolve_current_elder(uid)
            msg = f"目前服務對象：{elder['elder_name']}（ID: {elder['elder_id']}）" if elder else "尚未指派被照護者。"
            await line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=msg, quick_reply=quick_reply_main())]
                )
            )
            return

        if text in ("個人資訊", "個資"):
            elder = await resolve_current_elder(uid)
            msg = f"{elder['elder_name']} 的個人資訊（示範）" if elder else "尚未指派被照護者。"
            await line_api.reply_message(
                ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=msg)])
            )
            return

        # 4) 預設說明
        user = await get_user_by_line_user_id(uid)
        hello = user["name"] if user else "使用者"
        await line_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(
                    text=f"Hi {hello}！未綁定者請輸入手機號碼；已綁定可輸入「個人資訊」「我的長者」「解除綁定」。",
                    quick_reply=quick_reply_main(),
                )]
            )
        )
    except Exception as e:
        # fallback：避免 webhook 500
        try:
            await line_api.reply_message(
                ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text="系統忙碌中，請稍後再試。")])
            )
        except Exception as e2:
            print(f"[webhook] reply fallback failed: {e2}")
        print(f"[webhook] error: {e}")
