from pymongo import MongoClient
import os
from config import Config
from datetime import datetime
from bson.objectid import ObjectId

client = MongoClient(Config.MONGO_URI)
db = client.get_default_database(default='knee_osteoarthritis')

def get_db():
    return db

def insert_prediction(user_id, image_path, result, confidence, is_blurry=False):
    db.predictions.insert_one({
        "user_id": user_id,
        "image_path": image_path,
        "result": result,
        "confidence": confidence,
        "is_blurry": is_blurry,
        "timestamp": datetime.now()
    })

def get_user_predictions(user_id):
    return list(db.predictions.find({"user_id": user_id}).sort("timestamp", -1))

def get_prediction_by_id(user_id, prediction_id):
    try:
        return db.predictions.find_one({"_id": ObjectId(prediction_id), "user_id": user_id})
    except:
        return None

def delete_prediction(user_id, prediction_id):
    try:
        db.predictions.delete_one({"_id": ObjectId(prediction_id), "user_id": user_id})
    except:
        pass
