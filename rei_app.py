import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import pickle

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__, template_folder="templates")


with open("resnet_2_new", "r") as json_file:
    model = json_file.read()
model = model_from_json(model)
model.load_weights('resnet_new.h5')
model.compile(optimizer= 'adam' ,loss="categorical_crossentropy",metrics=["accuracy"])

with open("class_dicts", "rb") as loc:
  vocab_dict, vocab_dict_op = pickle.load(loc)

def model_predict(img, model):

    img = img.resize((224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    pred_label = vocab_dict_op[np.argmax(pred)]
    confidence = round(np.max(pred)*100)

    return pred_label, confidence


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
      try:
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds, confi = model_predict(img, model)

        return jsonify(result=  "'{}'".format(preds.capitalize())+ " (confidence:{}%) ".format(confi))
      except:
        return jsonify(result= "OOPS! Invalid image type")

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
