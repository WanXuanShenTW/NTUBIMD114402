from fastapi import APIRouter, WebSocket, Query
from ..utils.ws_connection_manager import ws_manager

ws_pose_router = APIRouter()

@ws_pose_router.websocket("/ws/pose")
async def websocket_pose(
    websocket: WebSocket,
    user_id: str = Query(..., description="用戶唯一識別ID")
):
    """
    WebSocket 路由：管理連線、斷線、訊息收發，並用 user_id 區分使用者。
    """
    await ws_manager.connect(user_id, websocket)
    try:
        while True:
            # 等待接收訊息
            data = await websocket.receive_text()
            print(f"[{user_id}] 收到訊息：{data}")

            # 回傳給同一個用戶（或你可以 broadcast，或再做動作分析等處理）
            await ws_manager.send_personal_message("資料已送達", user_id)
    except Exception as e:
        print(f"[{user_id}] 斷線/例外：{e}")
    finally:
        ws_manager.disconnect(user_id, websocket)
        print(f"[{user_id}] 已斷開連線")
