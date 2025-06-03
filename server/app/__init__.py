from flask import Flask
from .routes.video_routes import video_bp
from .routes.fall_routes import fall_bp
from .routes.gait_routes import gait_bp
from .routes.watchlist_routes import watchlist_bp
from .routes.user_routes import user_bp
from .routes.notify_line_routes import notify_line_bp
from .routes.news_voice_routes import news_voice_bp
from .routes.reels_routes import reels_bp 

def create_app():
    app = Flask(__name__)
    app.register_blueprint(video_bp)
    app.register_blueprint(fall_bp)
    app.register_blueprint(gait_bp)
    app.register_blueprint(watchlist_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(notify_line_bp)
    app.register_blueprint(news_voice_bp)
    app.register_blueprint(reels_bp)
    return app