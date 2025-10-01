from ckiptagger import data_utils
# data_utils.download_data_gdown("./")
data_utils.download_data_url("./")

from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request, abort
from linebot import  LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from ckiptagger import WS

app = Flask(__name__)
run_with_ngrok(app)
ws = WS("./data")

line_bot_api = LineBotApi('JXA4r5Tp2n7UF7mRcP2qWY1PzXM4f7FvCPYbTwaoKQoz2hU6916oxiADO8oMUDJBrHmGjL5WRCK8KR1+GAxtA+7Hr7uGLpFd1HuLVWRvo3CvBLlm7iNQqY2vJpRX8F9dgBaVrtn0JGW0KxzVrwa1iQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('9d8429a492ae28a6faa1f6f08f006d34')

@app.route("/test", methods=['GET'])
def test():
  return 'test'

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  # If there's an error, return 400
    return 'OK', 200  # Ensure it returns status code 200


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text1=[event.message.text]
    return_test = ""
    ckiplist = ws(text1)
    for tag in ckiplist:
      return_test += str(tag) + ","
    print(return_test)
    line_bot_api.reply_message(event.reply_token,TextSendMessage(return_test))

if __name__ == "__main__":
    app.run()