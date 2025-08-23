from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ..service.user_service import get_user_info

auth_router = APIRouter()

class LoginRequest(BaseModel):
    phone: str
    password: str

@auth_router.post("/login")
async def login(data: LoginRequest):
    phone = data.phone
    password = data.password

    if not phone.isdigit() or len(phone) != 10:
        raise HTTPException(status_code=400, detail="phone 必須是10位數字")
    
    try:
        user = await get_user_info(phone)
        if user is None:
            raise HTTPException(status_code=404, detail="使用者不存在")
        if user["password"] != password:
            raise HTTPException(status_code=401, detail="密碼錯誤")
        
        return {"message": "登入成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")