from flask import Blueprint, request, jsonify
from ..service.user_service import register_user

user_bp = Blueprint("user_bp", __name__)

@user_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    required = ["name", "phone", "password", "role_id"]
    missing = [key for key in required if key not in data]

    if missing:
        return jsonify({"error": f"缺少參數: {', '.join(missing)}"}), 400

    result = register_user(
        name=data["name"],
        phone=data["phone"],
        password=data["password"],
        role_id=data["role_id"],
        line_id=data.get("line_id")
    )

    if "error" in result:
        return jsonify({"error": result["error"]}), result.get("status", 409)
    
    return jsonify({"message": "註冊成功", "user_id": result["user_id"]})
