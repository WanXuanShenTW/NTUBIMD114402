from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Optional
from ..service.user_service import (
    add_user,
    update_user_info,
    change_user_password,
    get_user_info,
    delete_user_account
)
from ..service.emergency_contacts_service import remove_all_contacts_by_caregiver, get_contacts_by_caregiver
from ..utils.response_util import make_json_response
from ..exceptions import AlreadyExistsError, NotFoundError

user_router = APIRouter(tags=["使用者資料"])

class RegisterRequest(BaseModel):
    name: str
    phone: str
    password: str
    role_id: int
    gender: str
    address: Optional[str] = None

class UpdateUserRequest(BaseModel):
    phone: str
    name: Optional[str] = None
    role_id: Optional[int] = None
    address: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    phone: str
    old_password: str
    new_password: str
    confirm_password: str

class DeleteUserRequest(BaseModel):
    phone: str

@user_router.post("/register")
async def register(data: RegisterRequest):
    """
        使用者註冊
    """
    try:
        user_id = await add_user(
            name=data.name,
            phone=data.phone,
            password=data.password,
            role_id=data.role_id,
            gender=data.gender,
            address=data.address,
        )
        return await make_json_response(data={"user_id": user_id}, message="註冊成功", code=200)
    except AlreadyExistsError as ae:
        return await make_json_response(code=409, message="此號碼已被註冊", success=False)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)

@user_router.patch("/user")
async def update_user(data: UpdateUserRequest):
    """
        更新使用者資料
    """
    if not data.phone:
        raise HTTPException(status_code=400, detail="缺少 phone 參數")

    update_data = {key: value for key, value in data.dict().items() if key != "phone" and value is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="請提供至少一個欄位")

    try:
        success = await update_user_info(data.phone, **update_data)
        return await make_json_response(message="更新成功" if success else "更新失敗", code=200)
    except NotFoundError as ne:
        return await make_json_response(code=404, message=str(ne), success=False)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)

@user_router.patch("/user/password")
async def change_password(data: ChangePasswordRequest):
    """
        修改使用者密碼
    """
    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="新密碼與確認密碼不一致")

    try:
        success = await change_user_password(data.phone, data.old_password, data.new_password)
        return await make_json_response(message="密碼修改成功" if success else "密碼修改失敗", code=200)
    except NotFoundError as ne:
        return await make_json_response(code=404, message=str(ne), success=False)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)

@user_router.get("/user")
async def get_user(phone: str = Query(..., description="使用者電話")):
    """
        查詢使用者資料
    """
    try:
        user = await get_user_info(phone)
        return await make_json_response(data=user, message="查詢成功", code=200)
    except NotFoundError as ne:
        return await make_json_response(code=200, message="找不到使用者", success=True)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)

@user_router.delete("/user")
async def delete_user(data: DeleteUserRequest):
    """
        刪除使用者帳號
    """
    try:
        await get_user_info(data.phone)
        
        # 檢查是否存在與該使用者相關的緊急聯絡人關係
        contacts = await get_contacts_by_caregiver(data.phone)
        if contacts:
            await remove_all_contacts_by_caregiver(data.phone)
        
        # 再刪除使用者帳號
        await delete_user_account(data.phone)
        return await make_json_response(code=200, message="刪除成功", success=True)
    except NotFoundError as e:
        return await make_json_response(code=404, message="刪除失敗，使用者不存在", success=False)
    except Exception as e:
        return await make_json_response(code=500, message=str(e), success=False)