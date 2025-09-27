from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db import Database
from .routes.pose_routes import pose_router
from .routes.gait_routes import gait_router
from .routes.user_routes import user_router
from .routes.news_voice_routes import news_voice_router
from .routes.reels_routes import reels_router
from .routes.auth_routes import auth_router
from .routes.emergency_contacts_routes import contact_router 
from .routes.ws_test_routes import ws_test_router  # 測試用 WS
from .routes.fall_event_routes import fall_event_router
from .routes.line_routes import router as line_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時初始化
    try:
        await Database.init_pool()
        print("[✅] Database pool initialized successfully")
    except Exception as e:
        print(f"[❌] Failed to initialize database pool: {e}")
        raise
    
    yield
    
    # 關閉時清理
    try:
        await Database.close_pool()
        print("[✅] Database pool closed successfully")  
    except Exception as e:
        print(f"[⚠️] Error closing database pool: {e}")

def create_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(gait_router)
    app.include_router(user_router)
    app.include_router(news_voice_router)
    app.include_router(reels_router)
    app.include_router(auth_router)
    app.include_router(contact_router)
    app.include_router(pose_router)
    app.include_router(ws_test_router) 
    app.include_router(fall_event_router)
    app.include_router(line_router)
    return app