from flask import Blueprint, request, jsonify
from ..service.video_watchlist_service import add_video_to_watchlist, remove_video_from_watchlist, get_watchlist_video_data_by_id

watchlist_bp = Blueprint("watchlist_bp", __name__)

@watchlist_bp.route("/watchlist", methods=["GET"])
def get_watchlist_data():
    data = request.args
    missing = []
    if "user_id" not in data:
        missing.append("user_id")
    if "video_type" not in data:
        missing.append("video_type")
    if missing:
        return jsonify({"error": f"缺少參數: {', '.join(missing)}"}), 400
    
    user_id = request.args.get("user_id")
    video_type = request.args.get("video_type")
    
    try:
        watchlist_data = get_watchlist_video_data_by_id(int(user_id), video_type)
        if not watchlist_data:
            return jsonify({"message": "沒有找到任何收藏"}), 200
        return jsonify(watchlist_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@watchlist_bp.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    data = request.get_json()
    if not data:
        return jsonify({"error": "缺少必要參數"}), 400
    missing = []
    if not data.get("user_id"):
        missing.append("user_id")
    if not data.get("record_id"):
        missing.append("record_id")
    if not data.get("video_type"):
        missing.append("video_type")
    if missing:
        return jsonify({"error": f"缺少參數: {', '.join(missing)}"}), 400
    user_id = data["user_id"]
    record_id = data["record_id"]
    video_type = data["video_type"]
    try:
        add_video_to_watchlist(record_id, user_id, video_type)
        return jsonify({"message": "新增收藏成功", "user_id": user_id, "record_id": record_id, "video_type": video_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@watchlist_bp.route("/watchlist", methods=["DELETE"])
def delete_from_watchlist():
    data = request.get_json()
    if not data:
        return jsonify({"error": "缺少必要參數"}), 400
    missing = []
    if not data.get("user_id"):
        missing.append("user_id")
    if not data.get("record_id"):
        missing.append("record_id")
    if not data.get("video_type"):
        missing.append("video_type")
    if missing:
        return jsonify({"error": f"缺少參數: {', '.join(missing)}"}), 400
    user_id = data["user_id"]
    record_id = data["record_id"]
    video_type = data["video_type"]
    try:
        success = remove_video_from_watchlist(user_id, record_id, video_type)
        if success:
            return jsonify({"message": "刪除收藏成功", "user_id": user_id, "record_id": record_id, "video_type": video_type})
        else:
            return jsonify({"error": "找不到符合條件的收藏"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500