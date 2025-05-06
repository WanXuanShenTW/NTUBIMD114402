from flask import Blueprint, request, jsonify
import os
from datetime import datetime

test_bp = Blueprint("test_bp", __name__)
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@test_bp.route('/upload', methods=['POST'])
def upload_image():
    image = request.files.get('image')
    if not image:
        return jsonify({'error': 'no image provided'}), 400

    if not image.content_type.startswith('image/'):
        return jsonify({'error': 'invalid file type'}), 400

    filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(save_path)

    return jsonify({
        'status': 'success',
        'filename': filename,
        'path': save_path
    }), 200