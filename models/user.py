from flask_login import UserMixin
from services.db_service import get_db
from flask_bcrypt import check_password_hash, generate_password_hash
from bson.objectid import ObjectId

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data.get('email', '')
        self.name = user_data.get('name', '')
        self.password_hash = user_data.get('password', '')

    @staticmethod
    def get_by_id(user_id):
        user_data = get_db().users.find_one({"_id": ObjectId(user_id)})
        return User(user_data) if user_data else None

    @staticmethod
    def get_by_email(email):
        user_data = get_db().users.find_one({"email": email})
        return User(user_data) if user_data else None

    @staticmethod
    def create(name, email, password):
        hashed = generate_password_hash(password).decode('utf-8')
        user_data = {
            "name": name,
            "email": email,
            "password": hashed
        }
        res = get_db().users.insert_one(user_data)
        user_data['_id'] = res.inserted_id
        return User(user_data)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
