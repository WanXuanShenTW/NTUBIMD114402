from fastapi import APIRouter
from pydantic import BaseModel
from ..service.user_service import get_user_info
from ..utils.response_util import make_json_response

auth_router = APIRouter()

class LoginRequest(BaseModel):
    phone: str
    password: str

@auth_router.post("/login")
async def login(data: LoginRequest):
    phone = data.phone
    password = data.password

    if not phone.isdigit() or len(phone) != 10:
        return make_json_response(code=400, message="phone 必須是10位數字")
    
    try:
        user = await get_user_info(phone)
        if user is None:
            return await make_json_response(code=404, message="使用者不存在")
        if user["password"] != password:
            return await make_json_response(code=401, message="密碼錯誤")
        
        return await make_json_response(data={"user": user}, message="登入成功")
    except Exception as e:
        return await make_json_response(code=500, message=f"伺服器錯誤: {str(e)}")