from fastapi import APIRouter
from ..db import Database

debug_router = APIRouter(tags=["_debug"])

@debug_router.get("/_db/status")
async def db_status():
    # 這裡直接呼叫同步的 debug_status() 印在 console，
    # 也回傳一份 JSON 方便你在手機上看
    resp = {}
    try:
        pool = Database._pool
        if pool is None:
            resp = {"initialized": False}
        else:
            resp = {
                "initialized": True,
                "size": pool.size,
                "free": pool.freesize,
                "used": pool.size - pool.freesize,
                "min": pool.minsize,
                "max": pool.maxsize,
            }
    except Exception as e:
        resp = {"error": str(e)}
    finally:
        Database.debug_status()
    return resp
