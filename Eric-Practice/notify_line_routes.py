from flask import Blueprint, request, jsonify
from datetime import datetime
import requests

notify_line_bp = Blueprint('notify_line_bp', __name__)

LINE_CHANNEL_ACCESS_TOKEN = "JXA4r5Tp2n7UF7mRcP2qWY1PzXM4f7FvCPYbTwaoKQoz2hU6916oxiADO8oMUDJBrHmGjL5WRCK8KR1+GAxtA+7Hr7uGLpFd1HuLVWRvo3CvBLlm7iNQqY2vJpRX8F9dgBaVrtn0JGW0KxzVrwa1iQdB04t89/1O/w1cDnyilFU="

def push_line_message(user_id, message):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    print(f"[LINE] 傳送給 {user_id}，狀態碼：{response.status_code}")
    if response.status_code != 200:
        print(f"[LINE] 回應內容：{response.text}")
    return response.status_code

@notify_line_bp.route('/notify_line', methods=['POST'])
def notify_line():
    try:
        data = request.get_json()
        user_ids = data.get('user_id', [])
        if not isinstance(user_ids, list):
            user_ids = [user_ids]

        # 使用系統時間取代傳進來的時間
        detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"""⚠️ 注意！偵測到跌倒事件 ⚠️
        時間：{detected_time}
        請立即檢查爺爺奶奶們的狀況，確保他們安全無恙。"""

        success_list, fail_list = [], []
        for uid in user_ids:
            status = push_line_message(uid, message)
            (success_list if status == 200 else fail_list).append(uid)

        return jsonify({
            "status": "ok",
            "sent": success_list,
            "failed": fail_list,
            "message": message
        }), 200

    except Exception as e:
        print(f"[錯誤] 發送失敗：{str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 400