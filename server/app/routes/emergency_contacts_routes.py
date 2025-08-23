from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from ..service.emergency_contacts_service import (
    add_contact_by_phone,
    get_contact_relations,
    remove_contact
)

contact_router = APIRouter()

class CreateContactRequest(BaseModel):
    user_phone: str
    contact_phone: str
    priority: int
    relationship: str

class DeleteContactRequest(BaseModel):
    user_phone: str
    contact_phone: str

@contact_router.post("/contact")
async def create_contact(data: CreateContactRequest):
    try:
        message = await add_contact_by_phone(
            data.user_phone, data.contact_phone, data.priority, data.relationship
        )
        return {"message": message}
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@contact_router.get("/contact")
async def get_contacts(
    user_phone: str = Query(..., description="使用者電話"),
    role: int = Query(None, description="要過濾的角色ID（選填）")
):
    try:
        print(f"[DEBUG] 查詢 user_phone={user_phone}, role={role}")
        contacts = await get_contact_relations(user_phone, role)
        return contacts
    except ValueError as ve:
        print(f"[ERROR] ValueError: {ve}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        import traceback
        print(f"[EXCEPTION] 發生例外:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@contact_router.delete("/contact")
async def delete_contact(data: DeleteContactRequest):
    try:
        message = await remove_contact(data.user_phone, data.contact_phone)
        return {"message": message}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")
    
