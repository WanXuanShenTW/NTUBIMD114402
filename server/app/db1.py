# # app/db.py
# import os
# import asyncio
# from dotenv import load_dotenv
# from aiomysql import create_pool, OperationalError
# from contextlib import asynccontextmanager
# from aiomysql.cursors import DictCursor  # ✅ for dictionary-style rows

# load_dotenv()

# # ---- 可調參數（支援 .env）----
# USE_POOL_TIMEOUT = os.getenv("USE_POOL_TIMEOUT", "false").lower() == "true"
# POOL_TIMEOUT     = int(os.getenv("POOL_TIMEOUT", 15))          # 取得連線逾時（秒）← 放大預設
# POOL_RECYCLE     = int(os.getenv("DB_POOL_RECYCLE", 1800))     # 連線最長存活（秒）
# PRE_PING         = os.getenv("DB_PRE_PING", "true").lower() == "true"
# PING_TIMEOUT     = float(os.getenv("DB_PING_TIMEOUT", 1.0))
# DB_HOST          = os.getenv("DB_HOST", "127.0.0.1")
# DB_PORT          = int(os.getenv("DB_PORT", 3306))             # ✅ 支援自訂 port
# DB_USER          = os.getenv("DB_USER")
# DB_PASSWORD      = os.getenv("DB_PASSWORD")
# DB_NAME          = os.getenv("DB_NAME")
# DB_MINSIZE       = int(os.getenv("DB_POOL_MINSIZE", 1))
# DB_MAXSIZE       = int(os.getenv("DB_POOL_MAXSIZE", 10))
# CONNECT_TIMEOUT  = int(os.getenv("DB_CONNECT_TIMEOUT", 10))

# # 會自動重試的 MySQL 錯誤碼
# _RETRY_ERRCODES = {2006, 2013}  # MySQL server has gone away / Lost connection during query


# class _RetryingCursor:
#     """
#     透明重試的 Cursor 包裝器：
#     - 支援 async context manager
#     - execute / executemany 碰到 2006/2013 會自動重建連線並重試一次
#     """
#     def __init__(self, rconn, raw_cursor_factory_args, raw_cursor_factory_kwargs):
#         self._rconn = rconn
#         self._raw_args = raw_cursor_factory_args
#         self._raw_kwargs = raw_cursor_factory_kwargs
#         self._cur = None

#     async def __aenter__(self):
#         self._cur = await self._rconn._new_raw_cursor(*self._raw_args, **self._raw_kwargs)
#         return self

#     async def __aexit__(self, exc_type, exc, tb):
#         try:
#             if self._cur:
#                 await self._cur.close()
#         finally:
#             self._cur = None

#     def __getattr__(self, name):
#         return getattr(self._cur, name)

#     async def execute(self, sql, params=None):
#         try:
#             return await self._cur.execute(sql, params)
#         except OperationalError as e:
#             code = e.args[0] if e.args else None
#             if code in _RETRY_ERRCODES:
#                 await self._rconn._reacquire_new()
#                 self._cur = await self._rconn._new_raw_cursor(*self._raw_args, **self._raw_kwargs)
#                 return await self._cur.execute(sql, params)
#             raise

#     async def executemany(self, sql, seq_of_params):
#         try:
#             return await self._cur.executemany(sql, seq_of_params)
#         except OperationalError as e:
#             code = e.args[0] if e.args else None
#             if code in _RETRY_ERRCODES:
#                 await self._rconn._reacquire_new()
#                 self._cur = await self._rconn._new_raw_cursor(*self._raw_args, **self._raw_kwargs)
#                 return await self._cur.executemany(sql, seq_of_params)
#             raise


# class _RetryingConnection:
#     """
#     透明重試的 Connection 包裝器：
#     - 提供 cursor() 會回傳 _RetryingCursor（支援 async with）
#     - 借出時 pre-ping；必要時自動換新線
#     - 若執行中遭遇 2006/2013，會在 cursor 層自動重建連線並重試一次
#     """
#     def __init__(self, db):
#         self._db = db
#         self._raw = None
#         self._closed = False

#     async def _acquire_initial(self):
#         self._raw = await self._db._acquire_from_pool()
#         if PRE_PING:
#             try:
#                 await asyncio.wait_for(self._raw.ping(), timeout=PING_TIMEOUT)
#             except Exception:
#                 try:
#                     self._raw.close()
#                 finally:
#                     try:
#                         await self._raw.ensure_closed()
#                     except Exception:
#                         pass
#                 self._raw = await self._db._acquire_from_pool()

#     async def _reacquire_new(self):
#         old = self._raw
#         try:
#             if old:
#                 old.close()
#                 try:
#                     await old.ensure_closed()
#                 except Exception:
#                     pass
#         finally:
#             self._raw = await self._db._acquire_from_pool()
#             if PRE_PING:
#                 try:
#                     await asyncio.wait_for(self._raw.ping(), timeout=PING_TIMEOUT)
#                 except Exception:
#                     self._raw.close()
#                     try:
#                         await self._raw.ensure_closed()
#                     except Exception:
#                         pass
#                     self._raw = await self._db._acquire_from_pool()

#     async def _new_raw_cursor(self, *args, **kwargs):
#         # 相容寫法：cursor(dictionary=True) / cursorclass=... / cursor=...
#         if kwargs.pop("dictionary", False):
#             args = (DictCursor,) + args
#         if "cursorclass" in kwargs:
#             cls = kwargs.pop("cursorclass")
#             args = (cls,) + args
#         if "cursor" in kwargs:
#             cls = kwargs.pop("cursor")
#             args = (cls,) + args

#         if PRE_PING:
#             try:
#                 await asyncio.wait_for(self._raw.ping(), timeout=PING_TIMEOUT)
#             except Exception:
#                 await self._reacquire_new()

#         return await self._raw.cursor(*args, **kwargs)

#     def cursor(self, *args, **kwargs):
#         return _RetryingCursor(self, args, kwargs)

#     async def ensure_closed(self):
#         if self._raw:
#             try:
#                 await self._raw.ensure_closed()
#             except Exception:
#                 pass

#     def close(self):
#         if self._raw and not self._closed:
#             self._raw.close()
#             self._closed = True

#     def __getattr__(self, name):
#         return getattr(self._raw, name)


# class Database:
#     _pool = None

#     @classmethod
#     async def init_pool(cls):
#         """初始化連線池"""
#         if cls._pool is not None:
#             return
#         cls._pool = await create_pool(
#             host=DB_HOST,
#             port=DB_PORT,                   # ✅ 傳入 port
#             user=DB_USER,
#             password=DB_PASSWORD,
#             db=DB_NAME,
#             minsize=DB_MINSIZE,
#             maxsize=DB_MAXSIZE,
#             autocommit=True,
#             pool_recycle=POOL_RECYCLE,
#             connect_timeout=CONNECT_TIMEOUT,  # ✅ 握手逾時
#         )
#         print(f"[✅] Connection pool initialized "
#               f"(host={DB_HOST}:{DB_PORT}, size={DB_MINSIZE}-{DB_MAXSIZE}, "
#               f"recycle={POOL_RECYCLE}s, pre_ping={PRE_PING}, acquire_timeout={POOL_TIMEOUT if USE_POOL_TIMEOUT else 'off'})")

#     @classmethod
#     async def close_pool(cls):
#         """關閉連線池"""
#         if cls._pool:
#             cls._pool.close()
#             await cls._pool.wait_closed()
#             print("[❎] Connection pool closed")

#     @classmethod
#     async def _acquire_from_pool(cls):
#         if cls._pool is None:
#             raise RuntimeError("❌ Connection pool not initialized")
#         if USE_POOL_TIMEOUT:
#             return await asyncio.wait_for(cls._pool.acquire(), timeout=POOL_TIMEOUT)
#         return await cls._pool.acquire()

#     @classmethod
#     async def get_connection(cls):
#         """
#         取得透明重試版連線（與 aiomysql.Connection 相容）
#         可直接被既有 DAO / Service 使用。
#         """
#         rconn = _RetryingConnection(cls)
#         await rconn._acquire_initial()
#         return rconn

#     @classmethod
#     async def release_connection(cls, conn):
#         """釋放連線（支援包裝器或原生連線）"""
#         if cls._pool and conn:
#             try:
#                 raw = getattr(conn, "_raw", None) or conn
#                 try:
#                     await getattr(raw, "ensure_closed")()
#                 except Exception:
#                     pass
#                 cls._pool.release(raw)
#             except Exception:
#                 pass

#     @classmethod
#     @asynccontextmanager
#     async def connection(cls):
#         """
#         用法：
#             async with Database.connection() as conn:
#                 async with conn.cursor() as cur:
#                     await cur.execute("SELECT 1")
#         """
#         rconn = await cls.get_connection()
#         try:
#             yield rconn
#         finally:
#             try:
#                 await cls.release_connection(rconn)
#             except Exception:
#                 import os
import asyncio
from dotenv import load_dotenv
from aiomysql import create_pool, OperationalError
from contextlib import asynccontextmanager
from aiomysql.cursors import DictCursor


load_dotenv()

USE_POOL_TIMEOUT = os.getenv("USE_POOL_TIMEOUT", "false").lower() == "true"
POOL_TIMEOUT = int(os.getenv("POOL_TIMEOUT", 2))  # acquire 逾時（秒）

# 連線最長存活（秒）：應小於 MySQL wait_timeout；避免 NAT/防火牆閒置回收
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", 1800))

# 借出時 pre-ping，避免用到壞連線
PRE_PING = os.getenv("DB_PRE_PING", "true").lower() == "true"
PING_TIMEOUT = float(os.getenv("DB_PING_TIMEOUT", 1.0))

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
        if self._raw:
            try:
                await self._raw.ensure_closed()
            except Exception:
                pass

    def close(self):
        if self._raw and not self._closed:
            self._raw.close()
            self._closed = True

    def __getattr__(self, name):
        return getattr(self._raw, name)


class Database:
    _pool = None

    @classmethod
    async def init_pool(cls):
        """初始化連線池"""
        if cls._pool is not None:
            return
        cls._pool = await create_pool(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME"),
            minsize=1,
            maxsize=10,
            autocommit=True,
            pool_recycle=POOL_RECYCLE,   # ✅ 定期回收舊連線
            # connect_timeout=5,
        )
        print(f"[✅] Connection pool initialized (recycle={POOL_RECYCLE}s, pre_ping={PRE_PING})")

    @classmethod
    async def close_pool(cls):
        """關閉連線池"""
        if cls._pool:
            cls._pool.close()
            await cls._pool.wait_closed()
            print("[❎] Connection pool closed")

    @classmethod
    async def _acquire_from_pool(cls):
        if cls._pool is None:
            raise RuntimeError("❌ Connection pool not initialized")
        if USE_POOL_TIMEOUT:
            return await asyncio.wait_for(cls._pool.acquire(), timeout=POOL_TIMEOUT)
        return await cls._pool.acquire()

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
        """釋放連線（支援包裝器或原生連線）"""
        if cls._pool and conn:
            try:
                raw = getattr(conn, "_raw", None) or conn
                try:
                    await getattr(raw, "ensure_closed")()
                except Exception:
                    pass
                cls._pool.release(raw)
            except Exception:
                pass

    @classmethod
    @asynccontextmanager
    async def connection(cls):
        """
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
            except Exception:
                pass

    @classmethod
    def debug_status(cls):
        """列出目前 pool 狀態（僅供除錯用）"""
        if cls._pool:
            print(f"[🌀] Pool size      : {cls._pool.size}")
            print(f"[🔒] Used          : {cls._pool._used}")
            print(f"[🆓] Free          : {cls._pool.freesize}")
        else:
            print("❌ Pool not initialized.")


#     @classmethod
#     def debug_status(cls):
#         """列出目前 pool 狀態（僅供除錯用）"""
#         if cls._pool:
#             print(f"[🌀] Pool size : {cls._pool.size}")
#             print(f"[🔒] Used     : {cls._pool._used}")
#             print(f"[🆓] Free     : {cls._pool.freesize}")
#         else:
#             print("❌ Pool not initialized.")
