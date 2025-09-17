from typing import Dict, List
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

class WSConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        print(f"Attempting to connect user {user_id}.")
        await websocket.accept()
        self.active_connections.setdefault(user_id, []).append(websocket)
        print(f"User {user_id} connected.")

    def disconnect(self, user_id: str, websocket: WebSocket):
        print(f"Disconnecting user {user_id}.")
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        print(f"User {user_id} disconnected.")

    async def send_personal_message(self, message, user_id: str):
        for ws in self.active_connections.get(user_id, []):
            try:
                await ws.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(user_id, ws)

    async def send_json(self, data, user_id: str):
        print(f"Sending data to user {user_id}: {data}")
        for ws in self.active_connections.get(user_id, []):
            try:
                await ws.send_json(data)
            except WebSocketDisconnect:
                self.disconnect(user_id, ws)
                print(f"User {user_id} disconnected due to WebSocket error.")

    async def broadcast(self, message):
        for user_id, user_conns in self.active_connections.items():
            for ws in user_conns:
                try:
                    await ws.send_text(message)
                except WebSocketDisconnect:
                    self.disconnect(user_id, ws)

    async def broadcast_json(self, data):
        for user_id, user_conns in self.active_connections.items():
            for ws in user_conns:
                try:
                    await ws.send_json(data)
                except WebSocketDisconnect:
                    self.disconnect(user_id, ws)

    def get_user_connections(self, user_id: str) -> List[WebSocket]:
        return self.active_connections.get(user_id, [])

    def get_all_user_ids(self):
        return list(self.active_connections.keys())

ws_manager = WSConnectionManager()