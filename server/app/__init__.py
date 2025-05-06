from flask import Flask
from .routes.video_routes import video_bp
from .routes.fall_routes import fall_bp
from .routes.gait_routes import gait_bp
from .routes.fall_detection_routes import fall_detection_bp
from .routes.Test_route import test_bp  # 新增 Test_route 的 Blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(video_bp)
    app.register_blueprint(fall_bp)
    app.register_blueprint(gait_bp)
    app.register_blueprint(fall_detection_bp)
    app.register_blueprint(test_bp)  # 註冊 Test_route 的 Blueprint
    return app