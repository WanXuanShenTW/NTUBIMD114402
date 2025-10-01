import os
from flask import Flask, request
from pyngrok import ngrok

# 設定 Flask 監聽的 Port
port = 5000

# 設定 ngrok Token（使用環境變數存儲，避免暴露）
ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "2uWcR13TLSNfoEVw89AzjYvPrXd_2EGwrAbDvoYvDjgkQFEPA")
ngrok.set_auth_token(ngrok_token)

# 建立 ngrok 通道
public_url = ngrok.connect(f"http://localhost:{port}").public_url
print(" * ngrok tunnel:", public_url)

# 初始化 Flask 應用
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    name = request.args.get('name', 'None')  # 預設 name 為 'None'
    return f"Hello, {name}!"

if __name__ == '__main__':
    app.run()


