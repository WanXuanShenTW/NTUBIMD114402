from typing import Dict, List
from fastapi import WebSocket

class WSConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(user_id, []).append(websocket)

    def disconnect(self, user_id: str, websocket: WebSocket):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_personal_message(self, message, user_id: str):
        for ws in self.active_connections.get(user_id, []):
            await ws.send_text(message)

    async def send_json(self, data, user_id: str):
        for ws in self.active_connections.get(user_id, []):
            await ws.send_json(data)

    async def broadcast(self, message):
        for user_conns in self.active_connections.values():
            for ws in user_conns:
                await ws.send_text(message)

    async def broadcast_json(self, data):
        for user_conns in self.active_connections.values():
            for ws in user_conns:
                await ws.send_json(data)

    def get_user_connections(self, user_id: str) -> List[WebSocket]:
        return self.active_connections.get(user_id, [])

    def get_all_user_ids(self):
        return list(self.active_connections.keys())

ws_manager = WSConnectionManager()
