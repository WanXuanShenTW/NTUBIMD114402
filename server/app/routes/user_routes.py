from flask import Blueprint, request, jsonify
from ..service.user_service import (
    add_user,
    update_user_info,
    change_user_password,
    get_user_info,
    delete_user_account
)

user_bp = Blueprint("user_bp", __name__)

@user_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    required = ["name", "phone", "password", "role_id"]
    missing = [key for key in required if key not in data]

    if missing:
        return jsonify({"error": f"缺少參數: {', '.join(missing)}"}), 400

    try:
        user_id = add_user(
            name=data["name"],
            phone=data["phone"],
            password=data["password"],
            role_id=data["role_id"],
            line_id=data.get("line_id")
        )
        return jsonify({"message": "註冊成功", "user_id": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 409

@user_bp.route("/user", methods=["PATCH"])
def update_user():
    data = request.get_json()
    phone = data.get("phone")
    if not phone:
        return jsonify({"error": "缺少 phone 參數"}), 400

    allowed_fields = ["name", "role_id", "line_id"]
    update_data = {key: data[key] for key in allowed_fields if key in data}

    if not update_data:
        return jsonify({"error": "請提供至少一個欄位"}), 400

    try:
        success = update_user_info(phone, **update_data)
        return jsonify({"message": "更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@user_bp.route("/user/password", methods=["PATCH"])
def change_password():
    data = request.get_json()
    phone = data.get("phone")
    old_password = data.get("old_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not all([phone, old_password, new_password, confirm_password]):
        return jsonify({"error": "缺少必要參數"}), 400
    if new_password != confirm_password:
        return jsonify({"error": "新密碼與確認密碼不一致"}), 400

    try:
        # 這裡呼叫你 service 層的 change_password 函式
        change_user_password(phone, old_password, new_password)
        return jsonify({"message": "密碼修改成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@user_bp.route("/user", methods=["GET"])
def get_user():
    phone = request.args.get("phone")
    if not phone:
        return jsonify({"error": "缺少 phone 參數"}), 400

    try:
        user = get_user_info(phone)
        return jsonify(user)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route("/user", methods=["DELETE"])
def delete_user():
    data = request.get_json()
    if not data:
        return jsonify({"error": "缺少必要參數"}), 400
    
    missing = []
    if not data.get("phone"):
        missing.append("phone")
    if missing:
        return jsonify({"error": f"缺少參數: {', '.join(missing)}"}), 400

    try:
        success = delete_user_account(data.get("phone"))
        if success:
            return jsonify({"message": "刪除成功"})
        else:
            return jsonify({"error": "刪除失敗，使用者不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500