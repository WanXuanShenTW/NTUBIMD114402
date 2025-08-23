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

user_router = APIRouter()

class RegisterRequest(BaseModel):
    name: str
    phone: str
    password: str
    role_id: int
    line_id: Optional[str] = None

class UpdateUserRequest(BaseModel):
    phone: str
    name: Optional[str] = None
    role_id: Optional[int] = None
    line_id: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    phone: str
    old_password: str
    new_password: str
    confirm_password: str

class DeleteUserRequest(BaseModel):
    phone: str

@user_router.post("/register")
async def register(data: RegisterRequest):
    try:
        user_id = await add_user(
            name=data.name,
            phone=data.phone,
            password=data.password,
            role_id=data.role_id,
            line_id=data.line_id
        )
        return {"message": "註冊成功", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

@user_router.patch("/user")
async def update_user(data: UpdateUserRequest):
    if not data.phone:
        raise HTTPException(status_code=400, detail="缺少 phone 參數")

    update_data = {key: value for key, value in data.dict().items() if key != "phone" and value is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="請提供至少一個欄位")

    try:
        success = await update_user_info(data.phone, **update_data)
        return {"message": "更新成功"} if success else {"message": "更新失敗"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@user_router.patch("/user/password")
async def change_password(data: ChangePasswordRequest):
    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="新密碼與確認密碼不一致")

    try:
        success = await change_user_password(data.phone, data.old_password, data.new_password)
        return {"message": "密碼修改成功"} if success else {"message": "密碼修改失敗"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@user_router.get("/user")
async def get_user(phone: str = Query(..., description="使用者電話")):
    try:
        user = await get_user_info(phone)
        return user
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@user_router.delete("/user")
async def delete_user(data: DeleteUserRequest):
    try:
        success = await delete_user_account(data.phone)
        if success:
            return {"message": "刪除成功"}
        else:
            raise HTTPException(status_code=404, detail="刪除失敗，使用者不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))