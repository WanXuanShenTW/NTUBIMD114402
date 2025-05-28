import os
from flask import Flask, request
from pyngrok import ngrok

port = 5000  # 設定Flask應用的端口

# 設定ngrok的認證金鑰並啟動隧道
ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "2uWcR13TLSNfoEVw89AzjYvPrXd_2EGwrAbDvoYvDjgkQFEPA")
ngrok.set_auth_token(ngrok_token)
public_url = ngrok.connect(f"http://localhost:{port}").public_url
print(" * ngrok tunnel:", public_url)  # 顯示ngrok隧道的公共URL

############################################################################################

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])  # 同時處理GET和POST請求
def submit():
    if request.method == "POST":  # 如果是POST請求，則處理表單資料
        username = request.values['username']  # 從表單中獲取使用者名稱
        password = request.values['password']  # 從表單中獲取密碼
        if username == 'Eric' and password == '930111':  # 檢查帳號密碼
            return '歡迎光臨本網站！'  # 如果帳號密碼正確，顯示歡迎訊息
        else:
            return '帳號或密碼錯誤！'  # 如果帳號密碼錯誤，顯示錯誤訊息
    return """
            <form method='post' action=''>  # 顯示登入表單，這是GET請求的回應
                <p>帳號：<input type='text' name='username' /></p>
                <p>密碼：<input type='text' name='password' /></p>
                <p><button type='submit'>確定</button></p>
            </form>
    """

if __name__ == '__main__':
    app.run()  # 啟動Flask應用
