from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db import Database
from .routes.video_routes import video_router
from .routes.fall_routes import fall_router
from .routes.gait_routes import gait_router
from .routes.watchlist_routes import watchlist_router
from .routes.user_routes import user_router
from .routes.notify_line_routes import notify_line_router
from .routes.news_voice_routes import news_voice_router
from .routes.reels_routes import reels_router
from .routes.auth_routes import auth_router
from .routes.emergency_contacts_routes import contact_router 

# ✅ 壽命管理：啟動與關閉時處理連線池
@asynccontextmanager
async def lifespan(app: FastAPI):
    await Database.init_pool()
    yield
    await Database.close_pool()

def create_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(video_router)
    app.include_router(fall_router)
    app.include_router(gait_router)
    app.include_router(watchlist_router)
    app.include_router(user_router)
    app.include_router(notify_line_router)
    app.include_router(news_voice_router)
    app.include_router(reels_router)
    app.include_router(auth_router)
    app.include_router(contact_router)
    return app