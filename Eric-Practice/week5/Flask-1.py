ngrok_key = "2uWcR13TLSNfoEVw89AzjYvPrXd_2EGwrAbDvoYvDjgkQFEPA"
port = 5000
from pyngrok import ngrok
ngrok.set_auth_token(ngrok_key)
ngrok.connect(port).public_url

from flask import Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "index"

@app.route('/hello', methods=['GET'])
def hello():
    return "Welcome!"

if __name__ == '__main__':
    app.run()