import os
from pathlib import Path
from typing import Optional, List
import asyncio
import httpx

API_DATA_BASE = "https://api-data.line.me"

from linebot.v3.messaging import (
    AsyncMessagingApi,
    RichMenuRequest,
    RichMenuArea,
    RichMenuBounds,
    RichMenuSize,
    MessageAction,
)

RICHMENU_NAME = "SmartCare 主選單"
RICHMENU_IMG="static/line_background/1.png"
RICHMENU_IMG_TYPE = "image/jpeg" if RICHMENU_IMG.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _build_main_richmenu_request() -> RichMenuRequest:
    return RichMenuRequest(
        name=RICHMENU_NAME,
        chatBarText="開啟服務選單",
        size=RichMenuSize(width=2500, height=1686),
        selected=True,
        areas=[
            RichMenuArea(
                bounds=RichMenuBounds(x=0, y=0, width=833, height=1686),
                action=MessageAction(label="個人資訊", text="個人資訊"),
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=834, y=0, width=833, height=1686),
                action=MessageAction(label="我的長者", text="我的長者"),
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=1667, y=0, width=833, height=1686),
                action=MessageAction(label="解除綁定", text="解除綁定"),
            ),
        ],
    )


def _get_token_from_api(api: AsyncMessagingApi) -> str:
    candidates = []
    try:
        candidates.append(getattr(getattr(api, "api_client", None), "rest_client", None).configuration.access_token)
    except Exception:
        pass
    try:
        candidates.append(getattr(getattr(api, "api_client", None), "configuration", None).access_token)
    except Exception:
        pass
    try:
        candidates.append(getattr(getattr(api, "api_client", None), "config", None).access_token)
    except Exception:
        pass

    for tok in candidates:
        if tok:
            return tok

    # 最後退回環境變數（避免卡死）
    env_tok = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if env_tok:
        return env_tok

    raise RuntimeError("NO_TOKEN::AsyncMessagingApi 與環境變數都沒有 Access Token。請確認初始化與 .env 設定。")


async def _list_menus(api: AsyncMessagingApi) -> List:
    r = await api.get_rich_menu_list()
    return getattr(r, "richmenus", None) or []


async def _richmenu_has_image(token: str, rid: str) -> bool:
    """
    GET content：200=已有底圖；404=尚未上傳。
    其它狀態碼一律 raise，讓上層看得到錯誤。
    """
    url = f"{API_DATA_BASE}/v2/bot/richmenu/{rid}/content"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return False


async def _upload_richmenu_image(token: str, rid: str, img: Path, ctype: str) -> bool:
    if not img.exists():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND::{img}")

    url = f"{API_DATA_BASE}/v2/bot/richmenu/{rid}/content"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": ctype}
    data = img.read_bytes()

    for attempt in range(6):  # 最多重試 6 次（約 3 秒）
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, content=data)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            if attempt < 5:
                await asyncio.sleep(0.5)
                continue
            return False
        resp.raise_for_status()

    return False


async def _create_and_set_default(api: AsyncMessagingApi, token: str, img: Path) -> str:
    req = _build_main_richmenu_request()
    created = await api.create_rich_menu(rich_menu_request=req)
    rid = created.rich_menu_id
    await _upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
    await api.set_default_rich_menu(rid)
    return rid


async def ensure_default_richmenu(api: AsyncMessagingApi) -> str:
    token = _get_token_from_api(api)
    img = Path(RICHMENU_IMG)
    if not img.exists():
        raise RuntimeError(
            f"底圖不存在：{img}（需 2500x1686 PNG/JPG）。請放檔或以 LINE_RICHMENU_IMG 指向正確路徑。"
        )

    r = await api.get_rich_menu_list()
    menus = getattr(r, "richmenus", None) or []

    # 先找同名
    for m in menus:
        if m.name in ("SmartCareMain", "SmartCare 主選單"):
            ok = await _upload_richmenu_image(token, m.rich_menu_id, img, RICHMENU_IMG_TYPE)
            if not ok:
                try:
                    await api.delete_rich_menu(m.rich_menu_id)
                except Exception:
                    pass
                req = _build_main_richmenu_request()
                created = await api.create_rich_menu(rich_menu_request=req)
                rid = created.rich_menu_id
                ok2 = await _upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
                if not ok2:
                    raise RuntimeError("上傳底圖仍失敗：請再次確認 Token/Channel 與圖片路徑/尺寸。")
                await api.set_default_rich_menu(rid)
                return rid

            await api.set_default_rich_menu(m.rich_menu_id)
            return m.rich_menu_id

    # 沒同名 → 新建
    req = _build_main_richmenu_request()
    created = await api.create_rich_menu(rich_menu_request=req)
    rid = created.rich_menu_id
    ok = await _upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
    if not ok:
        # 新建後仍上傳失敗 → 再等 1 秒重試一次（保底）
        await asyncio.sleep(1.0)
        ok = await _upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
        if not ok:
            raise RuntimeError("上傳底圖失敗：請檢查圖片尺寸(2500x1686)、檔案路徑、以及 Messaging API Token。")
    await api.set_default_rich_menu(rid)
    return rid


async def ensure_user_linked_richmenu(api: AsyncMessagingApi, user_id: str) -> Optional[str]:
    menus = await _list_menus(api)
    if not menus:
        return None
    rid = menus[0].rich_menu_id
    await api.link_user_id_to_rich_menu_id(user_id=user_id, rich_menu_id=rid)
    return rid

async def reset_all_and_setup(api: AsyncMessagingApi) -> str:
    """
    一鍵重置：刪掉目前 Channel 底下的所有 Rich Menu，
    再用同一把 token 建立新的、上傳底圖並設為預設。
    """
    # 用穩健方法從 SDK 取得 token
    token = _get_token_from_api(api)

    img = Path(RICHMENU_IMG)
    if not img.exists():
        raise RuntimeError(
            f"底圖不存在：{img}（需 2500x1686 PNG/JPG）。請放檔或以 RICHMENU_IMG/LINE_RICHMENU_IMG 指定正確路徑。"
        )

    # 刪光
    r = await api.get_rich_menu_list()
    menus = getattr(r, "richmenus", None) or []
    for m in menus:
        try:
            await api.delete_rich_menu(m.rich_menu_id)
        except Exception:
            pass

    # 重建 + 上圖（帶重試）+ 設預設
    req = _build_main_richmenu_request()
    created = await api.create_rich_menu(rich_menu_request=req)
    rid = created.rich_menu_id

    # 這裡用你現有的上傳函式（具重試）：_upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
    ok = await _upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
    if not ok:
        # 再保底等一下重試一次
        await asyncio.sleep(1.0)
        ok = await _upload_richmenu_image(token, rid, img, RICHMENU_IMG_TYPE)
        if not ok:
            raise RuntimeError("上傳底圖失敗：請檢查圖片尺寸(2500x1686)、檔案路徑，以及 Messaging API Token。")

    await api.set_default_rich_menu(rid)
    return rid