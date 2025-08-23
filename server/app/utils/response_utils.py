async def make_json_response(data=None, code=200, message="", success=None):
    if success is None:
        success = (code == 200)
    return {
        "success": success,
        "code": code,
        "message": message,
        "data": data
    }