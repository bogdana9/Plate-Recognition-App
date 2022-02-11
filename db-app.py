
from flask import Flask, request, jsonify
import os 
import time
import requests
import tensorflow as tf
from flask_api import status
from flask_mongoengine import MongoEngine
import mongoengine as me

class Car(me.Document):
    number = me.StringField(required=True)
    
app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {
    "db": "myapp",
}
db = MongoEngine(app)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

@app.route('/update_number', methods=['POST'])
def update_number():
    if not request.is_json:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST

    number = request.json.get("number", None)
    if number:
        bttf = Car.objects(number=number)
        if len(bttf) == 0:
            bttf = Car(number=number)
            bttf.save()
            return jsonify({number: 'The plate number is saved.'}), status.HTTP_200_OK
        return jsonify({number: 'The plate number is already in the Database.'}), status.HTTP_200_OK
    else:
        return jsonify({number: "Not a number."}), status.HTTP_400_BAD_REQUEST


@app.route('/get_number', methods=['GET'])
def get_number():
    if not request.is_json:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST

    number = request.json.get("number", None)
    if number:
        bttf = Car.objects(number=number)
        message = False
        if len(bttf) > 0:
            message = True
        return jsonify({'Number exists': message}), status.HTTP_200_OK
    else:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST


if __name__ =="__main__":
    app.run(port=8002, debug=False)