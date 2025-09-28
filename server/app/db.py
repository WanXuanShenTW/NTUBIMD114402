import os
import asyncio
from dotenv import load_dotenv
from aiomysql import create_pool, OperationalError
from contextlib import asynccontextmanager
from aiomysql.cursors import DictCursor

load_dotenv()

USE_POOL_TIMEOUT = os.getenv("USE_POOL_TIMEOUT", "false").lower() == "true"
POOL_TIMEOUT = int(os.getenv("POOL_TIMEOUT", 15))  # acquire 逾時（秒）

# 連線最長存活（秒）：應小於 MySQL wait_timeout；避免 NAT/防火牆閒置回收
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", 300))  # 改為5分鐘

# 借出時 pre-ping，避免用到壞連線
PRE_PING = os.getenv("DB_PRE_PING", "true").lower() == "true"
PING_TIMEOUT = float(os.getenv("DB_PING_TIMEOUT", 3.0))

# 會自動重試的 MySQL 錯誤碼
_RETRY_ERRCODES = {2006, 2013}  # MySQL server has gone away / Lost connection during query


class _RetryingCursor:
    """
    透明重試的 Cursor 包裝器：
    - 支援 async context manager
    - 在 execute / executemany 碰到 2006/2013 會自動重建連線並重試一次
    """
    def __init__(self, rconn, raw_cursor_factory_args, raw_cursor_factory_kwargs):
        self._rconn = rconn
        self._raw_args = raw_cursor_factory_args
        self._raw_kwargs = raw_cursor_factory_kwargs
        self._cur = None

    async def __aenter__(self):
        self._cur = await self._rconn._new_raw_cursor(*self._raw_args, **self._raw_kwargs)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._cur:
                await self._cur.close()
        finally:
            self._cur = None

    def __getattr__(self, name):
        return getattr(self._cur, name)

    async def execute(self, sql, params=None):
        try:
            return await self._cur.execute(sql, params)
        except OperationalError as e:
            code = e.args[0] if e.args else None
            if code in _RETRY_ERRCODES:
                await self._rconn._reacquire_new()
                self._cur = await self._rconn._new_raw_cursor(*self._raw_args, **self._raw_kwargs)
                return await self._cur.execute(sql, params)
            raise

    async def executemany(self, sql, seq_of_params):
        try:
            return await self._cur.executemany(sql, seq_of_params)
        except OperationalError as e:
            code = e.args[0] if e.args else None
            if code in _RETRY_ERRCODES:
                await self._rconn._reacquire_new()
                self._cur = await self._rconn._new_raw_cursor(*self._raw_args, **self._raw_kwargs)
                return await self._cur.executemany(sql, seq_of_params)
            raise


class _RetryingConnection:
    """
    透明重試的 Connection 包裝器：
    - 提供 cursor() 會回傳 _RetryingCursor（支援 async with）
    - 借出時 pre-ping；必要時自動換新線
    - 若執行中遭遇 2006/2013，會在 cursor 層自動重建連線並重試一次
    """
    def __init__(self, db):
        self._db = db
        self._raw = None
        self._closed = False
        self._released = False
        
    async def _acquire_initial(self):
        self._raw = await self._db._acquire_from_pool()
        if PRE_PING:
            try:
                await asyncio.wait_for(self._raw.ping(), timeout=PING_TIMEOUT)
            except Exception:
                try:
                    self._raw.close()
                finally:
                    try:
                        await self._raw.ensure_closed()
                    except Exception:
                        pass
                self._raw = await self._db._acquire_from_pool()

    async def _reacquire_new(self):
        old = self._raw
        try:
            if old:
                old.close()
                try:
                    await old.ensure_closed()
                except Exception:
                    pass
        finally:
            self._raw = await self._db._acquire_from_pool()
            if PRE_PING:
                try:
                    await asyncio.wait_for(self._raw.ping(), timeout=PING_TIMEOUT)
                except Exception:
                    self._raw.close()
                    try:
                        await self._raw.ensure_closed()
                    except Exception:
                        pass
                    self._raw = await self._db._acquire_from_pool()

    async def _new_raw_cursor(self, *args, **kwargs):
        if PRE_PING:
            try:
                await asyncio.wait_for(self._raw.ping(), timeout=PING_TIMEOUT)
            except Exception:
                await self._reacquire_new()
        return await self._raw.cursor(*args, **kwargs)

    def cursor(self, *args, **kwargs):
        return _RetryingCursor(self, args, kwargs)

    async def ensure_closed(self):
        if self._raw and not self._closed:
            try:
                await self._raw.ensure_closed()
                self._closed = True
                print(f"[✅] Connection properly closed")
            except Exception as e:
                print(f"[⚠️] Error closing connection: {e}")

    def close(self):
        if self._raw and not self._closed:
            try:
                self._raw.close()
                self._closed = True
                print(f"[✅] Connection closed")
            except Exception as e:
                print(f"[⚠️] Error in close(): {e}")

    def __getattr__(self, name):
        return getattr(self._raw, name)


class Database:
    _pool = None

    @classmethod
    async def init_pool(cls):
        """初始化連線池 - 遠端資料庫適用版"""
        if cls._pool is not None:
            return
        
        # 根據MySQL max_connections=151，使用保守設定
        min_size = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
        max_size = int(os.getenv("DB_POOL_MAX_SIZE", "8"))  # 保守設定，避免耗盡連線
        
        cls._pool = await create_pool(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME"),
            minsize=min_size,
            maxsize=max_size,
            autocommit=True,
            pool_recycle=POOL_RECYCLE,
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "20")),
            charset=os.getenv("DB_CHARSET", "utf8mb4"),
        )
        print(f"[✅] Connection pool initialized (min={min_size}, max={max_size}, recycle={POOL_RECYCLE}s)")

    @classmethod
    async def close_pool(cls):
        """關閉連線池"""
        if cls._pool:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None
            print("[❎] Connection pool closed")

    @classmethod
    async def _acquire_from_pool(cls):
        """從連線池取得原始連線"""
        if not cls._pool:
            raise RuntimeError("Connection pool not initialized")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if USE_POOL_TIMEOUT:
                    raw_conn = await asyncio.wait_for(cls._pool.acquire(), timeout=POOL_TIMEOUT)
                else:
                    raw_conn = await cls._pool.acquire()
                
                return raw_conn
                
            except asyncio.TimeoutError:
                print(f"[⚠️] Pool acquire timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Connection pool acquire timeout after retries")
            except Exception as e:
                print(f"[⚠️] Pool acquire failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
            
            # 短暫等待後重試
            await asyncio.sleep(0.1 * (attempt + 1))
        
        raise RuntimeError("Failed to acquire connection after retries")

    @classmethod
    async def get_connection(cls):
        """
        取得透明重試版連線（與 aiomysql.Connection 相容）
        可直接被既有 DAO / Service 使用。
        """
        rconn = _RetryingConnection(cls)
        await rconn._acquire_initial()
        return rconn

    @classmethod
    async def release_connection(cls, conn):
        """修正版釋放連線 - 確保連線被正確釋放"""
        if not cls._pool or not conn:
            return
            
        try:
            # 確保關閉連線相關資源
            if hasattr(conn, '_raw') and conn._raw:
                raw_conn = conn._raw
                # 重要：將原始連線還給池子
                cls._pool.release(raw_conn)
                print(f"[✅] Connection released to pool")
            elif hasattr(conn, 'close'):
                # 直接是原始連線的情況
                cls._pool.release(conn)
                print(f"[✅] Raw connection released to pool")
                
        except Exception as e:
            print(f"[⚠️] Error releasing connection: {e}")
            # 如果釋放失敗，強制關閉連線
            try:
                if hasattr(conn, '_raw') and conn._raw:
                    conn._raw.close()
                elif hasattr(conn, 'close'):
                    conn.close()
            except:
                pass

    @classmethod
    @asynccontextmanager
    async def connection(cls):
        """
        連線上下文管理器
        用法：
            async with Database.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
        """
        rconn = await cls.get_connection()
        try:
            yield rconn
        finally:
            try:
                await cls.release_connection(rconn)
            except Exception as e:
                print(f"[⚠️] Error in connection context cleanup: {e}")

    @classmethod
    def debug_status(cls):
        """列出目前 pool 狀態（僅供除錯用）"""
        if cls._pool:
            print(f"[🌀] Pool size      : {cls._pool.size}")
            print(f"[🔒] Used          : {cls._pool.size - cls._pool.freesize}")
            print(f"[🆓] Free          : {cls._pool.freesize}")
            print(f"[📊] Min size      : {cls._pool.minsize}")
            print(f"[📊] Max size      : {cls._pool.maxsize}")
            
            # 如果連線池滿了，這是問題所在
            if cls._pool.freesize == 0:
                print("🚨 WARNING: Connection pool is exhausted!")
        else:
            print("❌ Pool not initialized.")