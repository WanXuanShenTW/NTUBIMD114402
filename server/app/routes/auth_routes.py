from flask import Blueprint, request, jsonify
from ..service.user_service import get_user_info

auth_bp = Blueprint("auth_bp", __name__)

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    phone = data.get("phone")
    password = data.get("password")

    if not phone or not password:
        return jsonify({"error": "缺少 phone 或 password"}), 400
    elif not phone.isdigit() or len(phone) != 10:
        return jsonify({"error": "phone 必須是10位數字"}), 400 
    
    try:
        user = get_user_info(phone)
        if user is None:
            return jsonify({"error": "使用者不存在"}), 404
        if user["password"] != password:
            return jsonify({"error": "密碼錯誤"}), 401
        
        return jsonify({"message": "登入成功"})
    except Exception as e:
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500
