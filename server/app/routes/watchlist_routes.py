from flask import Blueprint, request, jsonify
from ..service.video_watchlist_service import add_video_to_watchlist, remove_video_from_watchlist

watchlist_bp = Blueprint("watchlist_bp", __name__)

@watchlist_bp.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    data = request.get_json()
    if not data:
        return jsonify({"error": "缺少必要參數"}), 400
    user_id = data.get("user_id")
    record_id = data.get("record_id")
    video_type = data.get("video_type")
    if not all([user_id, record_id, video_type]):
        return jsonify({"error": "缺少必要參數"}), 400
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
    user_id = data.get("user_id")
    record_id = data.get("record_id")
    video_type = data.get("video_type")
    if not all([user_id, record_id, video_type]):
        return jsonify({"error": "缺少必要參數"}), 400
    try:
        success = remove_video_from_watchlist(user_id, record_id, video_type)
        if success:
            return jsonify({"message": "刪除收藏成功", "user_id": user_id, "record_id": record_id, "video_type": video_type})
        else:
            return jsonify({"error": "找不到符合條件的收藏"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500