
from flask import Flask, render_template, request
import os 
import time
import requests
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

@app.route('/', methods=['POST','GET'])
def index ():
    if request.method == 'POST':
        if 'image_name' in request.files:
            upload_file = request.files['image_name']
            filename = str(time.time()) + '.jpg'
            path_save = os.path.join(UPLOAD_PATH,filename)
            if upload_file.filename.endswith('.jpg') or upload_file.filename.endswith('.png'):
                upload_file.save(path_save)
                response = requests.request(
                        method='POST',
                        url='http://127.0.0.1:8001/extract_plate',
                        json={"image_path":path_save},
                        timeout=10
                    )
                response_json = response.json() #path of region of interest picture
                text = ''
                check = False
                if response.status_code == 200:
                    response = requests.request(
                        method='POST',
                        url='http://127.0.0.1:8001/extract_number',
                        json={"roi_path":response_json['roi_path']},
                        timeout=10
                    )
                    response_json = response.json()
                    text = response_json['number']
                    if len(text) > 0:
                        response = requests.request(
                                method='GET',
                                url='http://127.0.0.1:8002/get_number',
                                json={"number":text},
                                timeout=10
                            )
                        response_json = response.json()
                        check = response_json['Number exists']
                return render_template('index.html', upload=True, upload_image = filename, text = text, check=check)
        else:
            upload_plate_number = request.form['plate_number']
            response = requests.request(
                    method='POST',
                    url='http://127.0.0.1:8002/update_number',
                    json={"number":upload_plate_number},
                    timeout=10
                )
            response_json = response.json()

            return render_template('index.html', upload_number=True, plate_upload_result=response_json[upload_plate_number])
    return render_template('index.html', upload=False, upload_number=False)

if __name__ =="__main__":
    app.run(debug=False)