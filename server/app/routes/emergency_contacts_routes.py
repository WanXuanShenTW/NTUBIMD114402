from fastapi import APIRouter, Query
from pydantic import BaseModel
from ..service.emergency_contacts_service import (
    add_contact_by_phone,
    get_contacts_by_elder,
    get_contacts_by_caregiver,
    remove_contact
)
from ..utils.response_util import make_json_response

contact_router = APIRouter(tags=["緊急聯絡人"])

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
        新增緊急聯絡人
    """
    try:
        message = await add_contact_by_phone(
            data.elder_phone, data.caregiver_phone, data.relationship
        )
        return await make_json_response(data=None, code=200, message=message)
    except ValueError as ve:
        return await make_json_response(data=None, code=409, message=str(ve))
    except Exception as e:
        return await make_json_response(data=None, code=500, message=str(e))

@contact_router.get("/contacts/elder")
async def get_contacts_by_elder_route(
    elder_phone: str = Query(..., description="長者電話")
):
    """
        根據長者電話查詢所有的照護者
    """
    try:
        contacts = await get_contacts_by_elder(elder_phone)
        return await make_json_response(data=contacts, code=200, message="查詢成功")
    except ValueError as ve:
        return await make_json_response(data=None, code=404, message=str(ve))
    except Exception as e:
        return await make_json_response(data=None, code=500, message=str(e))

@contact_router.get("/contacts/caregiver")
async def get_contacts_by_caregiver_route(
    caregiver_phone: str = Query(..., description="照護者電話")
):
    """
        根據照護者電話查詢所有的長者
    """
    try:
        contacts = await get_contacts_by_caregiver(caregiver_phone)
        return await make_json_response(data=contacts, code=200, message="查詢成功")
    except ValueError as ve:
        return await make_json_response(data=None, code=404, message=str(ve))
    except Exception as e:
        return await make_json_response(data=None, code=500, message=str(e))

@contact_router.delete("/contact")
async def delete_contact(data: DeleteContactRequest):
    """
        刪除緊急聯絡人
    """
    try:
        message = await remove_contact(data.elder_phone, data.caregiver_phone)
        return await make_json_response(data=None, code=200, message=message)
    except ValueError as ve:
        return await make_json_response(data=None, code=404, message=str(ve))
    except Exception as e:
        return await make_json_response(data=None, code=500, message=str(e))