from flask import Blueprint, request, jsonify
from ..service.emergency_contacts_service import (
    add_contact_by_phone,
    get_contact_relations,
    remove_contact
)

contact_bp = Blueprint("contact_bp", __name__)

@contact_bp.route("/contact", methods=["POST"])
def create_contact():
    data = request.get_json()
    user_phone = data.get("user_phone")
    contact_phone = data.get("contact_phone")
    priority = data.get("priority")
    relationship = data.get("relationship")

    if not user_phone or not contact_phone:
        return jsonify({"error": "缺少 user_phone 或 contact_phone"}), 400

    try:
        message = add_contact_by_phone(user_phone, contact_phone, priority, relationship)
        return jsonify({"message": message})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@contact_bp.route("/contact", methods=["GET"])
def get_contacts():
    user_phone = request.args.get("user_phone")
    if not user_phone:
        return jsonify({"error": "缺少 user_phone"}), 400

    try:
        contacts = get_contact_relations(user_phone)
        return jsonify(contacts)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@contact_bp.route("/contact", methods=["DELETE"])
def delete_contact():
    data = request.get_json()
    user_phone = data.get("user_phone")
    contact_phone = data.get("contact_phone")

    if not user_phone or not contact_phone:
        return jsonify({"error": "缺少 user_phone 或 contact_phone"}), 400

    try:
        message = remove_contact(user_phone, contact_phone)
        return jsonify({"message": message})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
