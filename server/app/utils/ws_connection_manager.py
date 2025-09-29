from typing import Dict, List
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import traceback
from datetime import datetime

class WSConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    # ---- utils ----
    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _ws_id(self, ws: WebSocket) -> str:
        host = getattr(getattr(ws, "client", None), "host", "?")
        port = getattr(getattr(ws, "client", None), "port", "?")
        return f"id={id(ws)} {host}:{port}"

    def _count(self, user_id: str) -> int:
        return len(self.active_connections.get(user_id, []))

    # ---- lifecycle ----
    async def connect(self, user_id: str, websocket: WebSocket):
        print(f"[{self._ts()}][WS] Attempting to connect user={user_id} ({self._ws_id(websocket)})")
        await websocket.accept()
        self.active_connections.setdefault(user_id, []).append(websocket)
        print(f"[{self._ts()}][WS] User {user_id} connected ({self._ws_id(websocket)}). now={self._count(user_id)}")

    def disconnect(self, user_id: str, websocket: WebSocket):
        print(f"[{self._ts()}][WS] Disconnecting user={user_id} ({self._ws_id(websocket)})")
        if user_id in self.active_connections:
            conns = self.active_connections[user_id]
            if websocket in conns:
                conns.remove(websocket)
            if not conns:
                del self.active_connections[user_id]
        print(f"[{self._ts()}][WS] User {user_id} disconnected. remain={self._count(user_id)}")

    # ---- send helpers ----
    async def send_personal_message(self, message, user_id: str):
        conns = list(self.active_connections.get(user_id, []))  # copy 防 concurrent 修改
        for ws in conns:
            try:
                await ws.send_text(message)
            except WebSocketDisconnect as e:
                code = getattr(e, "code", None)
                print(f"[{self._ts()}][WS][SEND_TEXT] user={user_id} disconnect code={code} ({self._ws_id(ws)})")
                self.disconnect(user_id, ws)
            except Exception as e:
                print(f"[{self._ts()}][WS][SEND_TEXT][ERROR] user={user_id} ({self._ws_id(ws)}): {e}")
                traceback.print_exc()
                self.disconnect(user_id, ws)

    async def send_json(self, data, user_id: str):
        print(f"[{self._ts()}][WS] Sending data to user {user_id}: {data}")
        conns = list(self.active_connections.get(user_id, []))
        for ws in conns:
            try:
                await ws.send_json(data)
            except WebSocketDisconnect as e:
                code = getattr(e, "code", None)
                print(f"[{self._ts()}][WS][SEND_JSON] user={user_id} disconnect code={code} ({self._ws_id(ws)})")
                self.disconnect(user_id, ws)
                print(f"[{self._ts()}][WS][SEND_JSON] cleaned connection for user={user_id}")
            except Exception as e:
                print(f"[{self._ts()}][WS][SEND_JSON][ERROR] user={user_id} ({self._ws_id(ws)}): {e}")
                traceback.print_exc()
                self.disconnect(user_id, ws)

    async def broadcast(self, message):
        for user_id, user_conns in list(self.active_connections.items()):
            for ws in list(user_conns):
                try:
                    await ws.send_text(message)
                except WebSocketDisconnect as e:
                    code = getattr(e, "code", None)
                    print(f"[{self._ts()}][WS][BROADCAST_TEXT] user={user_id} disconnect code={code} ({self._ws_id(ws)})")
                    self.disconnect(user_id, ws)
                except Exception as e:
                    print(f"[{self._ts()}][WS][BROADCAST_TEXT][ERROR] user={user_id} ({self._ws_id(ws)}): {e}")
                    traceback.print_exc()
                    self.disconnect(user_id, ws)

    async def broadcast_json(self, data):
        for user_id, user_conns in list(self.active_connections.items()):
            for ws in list(user_conns):
                try:
                    await ws.send_json(data)
                except WebSocketDisconnect as e:
                    code = getattr(e, "code", None)
                    print(f"[{self._ts()}][WS][BROADCAST_JSON] user={user_id} disconnect code={code} ({self._ws_id(ws)})")
                    self.disconnect(user_id, ws)
                except Exception as e:
                    print(f"[{self._ts()}][WS][BROADCAST_JSON][ERROR] user={user_id} ({self._ws_id(ws)}): {e}")
                    traceback.print_exc()
                    self.disconnect(user_id, ws)

    # ---- debug helpers ----
    def get_user_connections(self, user_id: str) -> List[WebSocket]:
        return self.active_connections.get(user_id, [])

    def get_all_user_ids(self):
        return list(self.active_connections.keys())

    def debug_dump(self):
        print(f"[{self._ts()}][WS][DUMP] Active users: {len(self.active_connections)}")
        for uid, conns in self.active_connections.items():
            details = ", ".join(self._ws_id(ws) for ws in conns)
            print(f"  - user={uid} conns={len(conns)} [{details}]")

ws_manager = WSConnectionManager()
