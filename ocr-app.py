
from flask import Flask, request, jsonify
import os 
import time
import requests
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import imutils
import matplotlib.pyplot as plt
import tensorflow as tf
from flask_api import status



labels_dic = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: 'Q', 11: 'W',
    12: 'E', 13: 'R', 14: 'T', 15: 'Y', 16: 'U', 17: 'I',
    18: 'O', 19: 'P', 20: 'A', 21: 'S', 22: 'D', 23: 'F', 
    24: 'G', 25: 'H', 26: 'J', 27: 'K', 28: 'L', 29: 'Z', 
    30: 'X', 31: 'C', 32: 'V', 33: 'B', 34: 'N', 35: 'M'}

search_plate_model = tf.keras.models.load_model(r'C:\Users\bogda\Documents\GitHub\Plate-Recognition-App\zASSETS\object_detection_4')
def top_3_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
INPUT_SHAPE = (32, 32, 1)

def CNN_model(activation = 'softmax', 
              loss = 'categorical_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy', top_3_categorical_accuracy]):
    
    model =  tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size = (3, 3),
                     activation = 'relu',
                     input_shape = INPUT_SHAPE))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(tf.keras.layers.Conv2D(128, (4, 4), activation = 'relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(36, activation = activation))
    
    # Compile the model
    model.compile(loss = loss,
                  optimizer = optimizer, 
                  metrics = metrics)
    
    return model

ocr_model = CNN_model()
ocr_model.load_weights(r'C:\Users\bogda\Documents\GitHub\Plate-Recognition-App\zASSETS\OCR\weights.best.letters.hdf5')

def objectDetection(path):
    filename = os.path.basename(path)
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    coords = search_plate_model.predict(test_arr) 
    denorm = np.array([w,w,h,h]) 
    coords = coords * denorm 
    coords = coords.astype(np.int32)
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3) #rgb format
    # convert into bgr
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),image_bgr)
    xmin, xmax, ymin, ymax = coords[0]
    #crop the bounding box -region of interest
    img = np.array(load_img(path))
    
    roi = img [ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    scale_percent = 200 # percent of original size
    
    width = int(roi_bgr.shape[1] * scale_percent / 100)
    height = int(roi_bgr.shape[0] * scale_percent / 100)
    ratio_w_h = width / height
    perfect_height = 158
    dim = (int(ratio_w_h * perfect_height), perfect_height)
    # resize image
    resized = cv2.resize(roi_bgr, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('./static/roi/{}'.format(filename),resized)
    return './static/roi/{}'.format(filename)
   

def objectCharacterRecognition (image_path):
    I = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    _,I = cv2.threshold(I,0.,255.,cv2.THRESH_OTSU)
    I = cv2.bitwise_not(I)
    height, width = I.shape
    _,labels,_,_ = cv2.connectedComponentsWithStats(I) 

    counters = []
    counters_extra = []
    for i in range(0,labels.max()+1):
        mask = cv2.compare(labels,i,cv2.CMP_EQ)
        ctrs,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        for cnt in ctrs:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if (w >= 13 and w <= 150) and (h >= height/3 and h <= 100) and (h >= w):
                counters.append((x, y, w, h))
            elif (w >= 10 and w <= 150) and (h >= 35 and h <= 100) and (h + 5 >= w):
                counters_extra.append((x, y, w, h))
    if len(counters) < 6:
        counters += counters_extra
    counters.sort(key = lambda c: c[0])
    number = ''
    for i, c in enumerate(counters):
        x, y, w, h = c
        roi = I[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        # if the width is greater than the height, resize along the
        # width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)

        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=32)

        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)

        # pad the image and force 32x32 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255))
        img_arr = cv2.resize(padded, (32, 32))
        img_arr = img_arr.reshape(1, 32, 32, 1)
        preds = ocr_model.predict(img_arr)
        i = np.argmax(preds)
        number += labels_dic[i]
    return number

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

@app.route('/extract_plate', methods=['POST'])
def extract_plate():
    if not request.is_json:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST

    image_path = request.json.get("image_path")
    roi_path = objectDetection(image_path)
    if roi_path:
        return jsonify({'roi_path': roi_path}), status.HTTP_200_OK
    else:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST

@app.route('/extract_number', methods=['POST'])
def extract_number():
    if not request.is_json:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST

    image_path = request.json.get("roi_path")
    number = objectCharacterRecognition(image_path)
    if image_path:
        return jsonify({'number': number}), status.HTTP_200_OK
    else:
        return jsonify({"err": "Not a json request."}), status.HTTP_400_BAD_REQUEST

if __name__ =="__main__":
    app.run(port=8001, debug=False)