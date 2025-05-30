from flask import Flask
from .routes.video_routes import video_bp
from .routes.fall_routes import fall_bp
from .routes.gait_routes import gait_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(video_bp)
    app.register_blueprint(fall_bp)
    app.register_blueprint(gait_bp)
    return app