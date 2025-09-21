from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from ..utils.ws_connection_manager import ws_manager

ws_test_router = APIRouter(tags=["WS 測試"])

@ws_test_router.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    # 1) 確保 user_id 作為查詢參數傳遞
    user_id = websocket.query_params.get("user_id")
    if not user_id or not user_id.strip():
        await websocket.close(code=1008, reason="Missing required query param: user_id")
        return

    # 2) 建立 WebSocket 連接
    await ws_manager.connect(user_id, websocket)
    print(f"User {user_id} connected to /ws/test.")

    try:
        while True:
            # 3) 接收來自客戶端的訊息
            message = await websocket.receive_text()
            print(f"Received message from user {user_id}: {message}")

            # 4) 回傳確認消息給客戶端
            response = {"ok": True, "received": message}
            await websocket.send_json(response)
            print(f"Sent confirmation to user {user_id}: {response}")

    except WebSocketDisconnect:
        # 5) 處理 WebSocket 斷線
        print(f"User {user_id} disconnected from /ws/test.")
        ws_manager.disconnect(user_id, websocket)

    except Exception as e:
        # 6) 處理其他異常
        print(f"Error for user {user_id} in /ws/test: {e}")
        ws_manager.disconnect(user_id, websocket)
        await websocket.close(code=1011, reason="Internal server error")