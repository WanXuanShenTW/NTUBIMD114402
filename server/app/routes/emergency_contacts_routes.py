from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from ..service.emergency_contacts_service import (
    add_contact_by_phone,
    get_contacts_by_elder,
    get_contacts_by_caregiver,
    remove_contact
)

contact_router = APIRouter(tags=["緊急聯絡關係"])

class CreateContactRequest(BaseModel):
    elder_phone: str
    caregiver_phone: str
    relationship: str

class DeleteContactRequest(BaseModel):
    elder_phone: str
    caregiver_phone: str

@contact_router.post("/contact")
async def create_contact(data: CreateContactRequest):
    """
        新增緊急聯絡關係
    """
    try:
        message = await add_contact_by_phone(
            data.elder_phone, data.caregiver_phone, data.relationship
        )
        return {"message": message}
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@contact_router.get("/contacts/caregiver")
async def get_contacts_by_elder_id(
    elder_phone: str = Query(..., description="長者電話")
):
    """
        查詢對應照護者
    """
    try:
        contacts = await get_contacts_by_elder(elder_phone)
        return contacts
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@contact_router.get("/contacts/elder")
async def get_contacts_by_caregiver_id(
    caregiver_phone: str = Query(..., description="照護者電話")
):
    """
        查詢對應長者
    """
    try:
        contacts = await get_contacts_by_caregiver(caregiver_phone)
        return contacts
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@contact_router.delete("/contact")
async def delete_contact(data: DeleteContactRequest):
    """
        刪除緊急聯絡關係
    """
    try:
        message = await remove_contact(data.elder_phone, data.caregiver_phone)
        return {"message": message}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")