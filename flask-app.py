
from flask import Flask, render_template, request
import os 
import time
import requests
app = Flask(__name__)
from tensorflow.keras.preprocessing.image import load_img
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

@app.route('/', methods=['POST','GET'])
def index ():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = str(time.time()) + '.jpg'
        if upload_file.filename.endswith('.jpg') or upload_file.filename.endswith('.png'):
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            response = requests.request(
                    method='POST',
                    url='http://127.0.0.1:8001/extract_plate',
                    json={"image_path":path_save},
                    timeout=10
                )
            response_json = response.json()
            text = ''
            if response.status_code == 200:
                response = requests.request(
                    method='POST',
                    url='http://127.0.0.1:8001/extract_number',
                    json={"roi_path":response_json['roi_path']},
                    timeout=10
                )
                response_json = response.json()
                text = response_json['number']
            return render_template('index.html', upload=True, upload_image = filename, text = text)
        
            
    return render_template('index.html', upload=False)

if __name__ =="__main__":
    app.run(debug=True)