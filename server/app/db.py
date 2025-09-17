import os
import asyncio
from dotenv import load_dotenv
from aiomysql import create_pool
from contextlib import asynccontextmanager

load_dotenv()

USE_POOL_TIMEOUT = os.getenv("USE_POOL_TIMEOUT", "false").lower() == "true"
POOL_TIMEOUT = int(os.getenv("POOL_TIMEOUT", 2))  # 預設 timeout 2 秒

class Database:
    _pool = None

    @classmethod
    async def init_pool(cls):
        """初始化連線池"""
        cls._pool = await create_pool(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME"),
            minsize=1,
            maxsize=10,
            autocommit=True,
        )
        print("[✅] Connection pool initialized")

    @classmethod
    async def close_pool(cls):
        """關閉連線池"""
        if cls._pool:
            cls._pool.close()
            await cls._pool.wait_closed()
            print("[❎] Connection pool closed")

    @classmethod
    async def get_connection(cls):
        """取得連線，可選 timeout 控制"""
        if cls._pool is None:
            raise RuntimeError("❌ Connection pool not initialized")

        try:
            if USE_POOL_TIMEOUT:
                return await asyncio.wait_for(cls._pool.acquire(), timeout=POOL_TIMEOUT)
            else:
                return await cls._pool.acquire()
        except asyncio.TimeoutError:
            raise RuntimeError("❌ Connection pool acquire timed out")

    @classmethod
    async def release_connection(cls, conn):
        """釋放連線"""
        if cls._pool and conn:
            cls._pool.release(conn)

    @classmethod
    @asynccontextmanager
    async def connection(cls):
        """可使用 async with 的連線操作"""
        conn = await cls.get_connection()
        try:
            yield conn
        finally:
            await cls.release_connection(conn)

    @classmethod
    def debug_status(cls):
        """列出目前 pool 狀態（僅供除錯用）"""
        if cls._pool:
            print(f"[🌀] Pool size      : {cls._pool.size}")
            print(f"[🔒] Used          : {cls._pool._used}")
            print(f"[🆓] Free          : {cls._pool.freesize}")
        else:
            print("❌ Pool not initialized.")
