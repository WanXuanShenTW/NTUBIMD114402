from flask import Flask, request, abort
from flask_ngrok import run_with_ngrok 
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)  # 初始化Flask應用
run_with_ngrok(app)  # 啟動ngrok，將Flask應用暴露到公網

###########################################################################

# Line API 設定
line_bot_api = LineBotApi('JXA4r5Tp2n7UF7mRcP2qWY1PzXM4f7FvCPYbTwaoKQoz2hU6916oxiADO8oMUDJBrHmGjL5WRCK8KR1+GAxtA+7Hr7uGLpFd1HuLVWRvo3CvBLlm7iNQqY2vJpRX8F9dgBaVrtn0JGW0KxzVrwa1iQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('9d8429a492ae28a6faa1f6f08f006d34')


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  # 如果簽名錯誤，回傳 400 錯誤

    return 'OK', 200  # 確保回傳 200 OK 狀態碼

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # 回應收到的文字訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text)
    )

if __name__ == "__main__":
    app.run()
