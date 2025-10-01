from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio

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
from .routes.debug_routes import debug_router
from .routes.debug_borrowers import borrowers_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    try:
        # 啟動時初始化資料庫池
        await Database.init_pool()
        print("[✅] Database pool initialized successfully")
        yield
    except Exception as e:
        print(f"[❌] Startup error: {e}")
        raise
    finally:
        # 關閉時強制清理連線池，設置超時避免卡住
        try:
            if Database._pool:
                print("[🔄] Closing database pool...")
                Database._pool.close()
                # 設置 5 秒超時，避免無限等待
                await asyncio.wait_for(Database._pool.wait_closed(), timeout=5.0)
                print("[✅] Database pool closed successfully")
        except asyncio.TimeoutError:
            print("[⚠️] Database pool close timeout - forcing shutdown")
            # 超時就直接設為 None，讓程式繼續關閉
            Database._pool = None
        except Exception as e:
            print(f"[❌] Database pool close error: {e}")
            Database._pool = None

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
    app.include_router(debug_router)
    app.include_router(borrowers_router)
    return app