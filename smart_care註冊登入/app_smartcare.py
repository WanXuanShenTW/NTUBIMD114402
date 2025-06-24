from flask import Flask
from db_connection import  get_db_connection
from service.video import video_bp
from service.fall import fall_bp
from service.sleep import sleep_bp
from service.news import news_bp
from service.gait_instability import gait_instability_bp
from service.user import user_bp,login_manager



app = Flask(__name__)
app.secret_key = "your_secret_key" 

login_manager.init_app(app)

# 註冊所有 Blueprint
app.register_blueprint(video_bp)
app.register_blueprint(fall_bp)
app.register_blueprint(sleep_bp)
app.register_blueprint(news_bp)
app.register_blueprint(gait_instability_bp)
app.register_blueprint(user_bp, url_prefix="/user")

if __name__ == '__main__':
    app.run(debug=True)

    