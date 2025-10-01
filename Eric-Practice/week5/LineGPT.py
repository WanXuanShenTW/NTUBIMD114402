from ckiptagger import data_utils  # 匯入CKIP的資料下載工具

# 下載CKIP資料（這行被註解掉了，可以根據需要啟用）
# data_utils.download_data_gdown("./")
data_utils.download_data_url("./")  # 下載CKIP標註資料到當前目錄

from flask_ngrok import run_with_ngrok  # 匯入Flask與ngrok整合的工具
from flask import Flask  # 匯入Flask框架
from flask import request, abort  # 匯入request（處理請求）與abort（終止請求）方法
from linebot import LineBotApi, WebhookHandler  # 匯入LINE的API與Webhook處理器
from linebot.exceptions import InvalidSignatureError  # 匯入LINE簽名驗證錯誤處理
from linebot.models import MessageEvent, TextMessage, TextSendMessage  # 匯入LINE消息模型

from ckiptagger import WS  # 匯入CKIP的斷詞工具

# 設定OpenAI API金鑰，用於ChatGPT互動
key = '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'

# 設定OpenAI的API金鑰，並進行ChatGPT問答的初始化
from openai import OpenAI

client = OpenAI(api_key=key)

def chatgpt_QA(Q):
    """
    這是用來和ChatGPT進行對話的函數
    會將問題傳給ChatGPT，並返回ChatGPT的回答
    """
    response =client.chat.completions.create(model="gpt-4o-mini",  # 使用GPT-4的mini版模型
    messages=[
        {"role":"system", "content":"以下對話請用繁體中文回答問題"},  # 系統角色指定使用繁體中文回答
        {"role":"user","content":Q}  # 使用者的問題
    ])
    return response.choices[0].message.content.strip()  # 返回回答並去除多餘空格

ws = WS("./data")  # 初始化CKIP的斷詞工具

# 設定LINE Bot的API金鑰
line_bot_api = LineBotApi('JXA4r5Tp2n7UF7mRcP2qWY1PzXM4f7FvCPYbTwaoKQoz2hU6916oxiADO8oMUDJBrHmGjL5WRCK8KR1+GAxtA+7Hr7uGLpFd1HuLVWRvo3CvBLlm7iNQqY2vJpRX8F9dgBaVrtn0JGW0KxzVrwa1iQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('9d8429a492ae28a6faa1f6f08f006d34')  # 設定LINE的Webhook Handler

app = Flask(__name__)  # 初始化Flask應用
run_with_ngrok(app)  # 啟動ngrok，將Flask應用暴露到公網

@app.route("/test", methods=['GET'])  # 測試用的GET路由
def test():
    return 'test'  # 回傳簡單的'test'訊息

@app.route("/callback", methods=['POST'])  # 處理LINE平台發送過來的POST請求
def callback():
    signature = request.headers['X-Line-Signature']  # 從HTTP標頭獲取LINE簽名

    body = request.get_data(as_text=True)  # 獲取請求的body內容（即LINE平台發送的消息）
    app.logger.info("Request body: " + body)  # 輸出請求內容，便於調試

    try:
        handler.handle(body, signature)  # 處理LINE消息
    except InvalidSignatureError:  # 如果簽名無效，返回400錯誤
        abort(400)

    return 'OK'  # 返回200狀態碼，表示成功處理請求

@handler.add(MessageEvent, message=TextMessage)  # 處理來自LINE平台的文本消息
def handle_message(event):
    """
    處理用戶發送的文本消息
    根據訊息內容進行處理，若符合條件，呼叫ChatGPT或CKIP進行處理
    """
    text1 = [event.message.text]  # 取得用戶發送的訊息
    return_test = ""  # 儲存回應的變數
    temp = text1[0]  # 取得訊息中的第一條內容
    if "Q:" in temp[:2]:  # 判斷訊息是否以"Q:"開頭
        Q = temp[2:]  # 若是，則將問題去掉"Q:"部分，傳給ChatGPT
        return_test = chatgpt_QA(Q)  # 呼叫ChatGPT回應
    else:
        ckiplist = ws(text1)  # 若不是問題，則進行中文斷詞處理
        for tag in ckiplist:
            return_test += str(tag) + ","  # 逐一將斷詞結果串接起來

    print(return_test)  # 輸出回應內容
    line_bot_api.reply_message(event.reply_token, TextSendMessage(return_test))  # 回覆LINE平台

if __name__ == "__main__":
    app.run()  # 啟動Flask應用
