import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_super_secret_key_1234')
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/knee_osteoarthritis')
    UPLOAD_FOLDER = os.path.join('static', 'tests')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
