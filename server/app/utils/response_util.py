from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from .time_utils import now_str

async def make_json_response(data=None, code=200, message="", success=None):
    print(f"Time : {now_str()}")
    if success is None:
        success = (200 <= int(code) < 300)  # 2xx 視為成功
    if success != True:
        print(f"[ERROR] {message}")
    else:
        print(f"[INFO] {message}")
    payload = {
        "success": success,
        "code": code,
        "message": message,
        "data": data
    }
    return JSONResponse(content=jsonable_encoder(payload), status_code=int(code))