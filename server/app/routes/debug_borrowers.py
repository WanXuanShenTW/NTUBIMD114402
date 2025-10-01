from fastapi import APIRouter
from ..db import Database

borrowers_router = APIRouter(tags=["_debug"])

@borrowers_router.get("/_db/borrowers")
async def borrowers():
    return Database.borrowers_snapshot()