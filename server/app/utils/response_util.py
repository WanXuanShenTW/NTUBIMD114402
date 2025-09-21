from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

async def make_json_response(data=None, code=200, message="", success=None):
    if success is None:
        success = (200 <= int(code) < 300)  # 2xx 視為成功
    payload = {
        "success": success,
        "code": code,
        "message": message,
        "data": data
    }
    return JSONResponse(content=jsonable_encoder(payload), status_code=int(code))