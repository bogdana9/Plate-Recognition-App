import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

model = tf.keras.models.load_model(r'C:\Users\bogda\Documents\GitHub\Plate-Recognition-App\static\models\object_detection_4')

def objectDetection(path, filename):
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    coords = model.predict(test_arr) 
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
    return coords


import keras_ocr
import matplotlib.pyplot as plt


def objectCharacterRecognition (path, filename):
    img = np.array(load_img(path))
    coords = objectDetection(path,filename)
    xmin, xmax, ymin, ymax = coords[0]
    #crop the bounding box -region of interest
    roi= img [ymin:ymax, xmin:xmax]
    images = keras_ocr.tools.read(roi)
    roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
    text = pt.image_to_string(roi)
    pipeline = keras_ocr.pipeline.Pipeline()
    predictions = pipeline.recognize(images=[roi])[0]
    #drawn = keras_ocr.tools.drawBoxes(
    #    image=img, boxes=predictions, boxes_format='predictions'
    #)
    #print(
    #    'Predicted:', [text for text, box in predictions]
    #)
    #plt.imshow(drawn)
    #plt.show()
    text = [text for text, _ in predictions]
    return  "".join(text)
#import os
#for root, dirs, files in os.walk(r'C:\Users\bogda\Documents\GitHub\Plate-Recognition-App\Labelling\dataset'):
    #for filename in files:
        #if filename.endswith('.jpeg'):
            #print(filename)
            #try:
               # file_path = os.path.join(root, filename)
                #objectCharacterRecognition(file_path,filename)
           # except:
               # pass 



               