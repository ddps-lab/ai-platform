from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import os
import requests
import json
import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

# 학습된 모델 로드
inference_model = load_model('flower_model.keras')

def getPrediction(filename):
    image = keras.utils.load_img('static/uploads/' + filename, target_size=(180, 180))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = inference_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    
    result = [(label, str(np.round(acc * 100, 4)) + '%') for (label, acc) in zip(class_names, score)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result


def get_instance_info():
    try:
        instance_id = requests.get("http://169.254.169.254/latest/meta-data/instance-id", timeout=2).text
        instance_type = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=2).text
        avail_zone = requests.get("http://169.254.169.254/latest/meta-data/placement/availability-zone", timeout=2).text
    except:
        instance_id = "Error"
        instance_type = "Error"
        avail_zone = "Error"

    try:
        geo_info = requests.get('http://ipapi.co/json')
        geo_json = json.loads(geo_info.text)
        geo_ip = geo_json['ip']
        geo_country_name = geo_json['country_name']
        geo_region_name = geo_json['region']
        geo_lat_lon = f"{geo_json['latitude']} / {geo_json['longitude']}"
    except:
        geo_ip = "Error"
        geo_country_name = "Error"
        geo_region_name = "Error"
        geo_lat_lon = "Error"

    try:
        geo_info = requests.get('http://ipinfo.io/json')
        geo_json = json.loads(geo_info.text)
        geo_time_zone = geo_json['timezone']
    except:
        geo_time_zone = "Error"

    for info in [geo_ip, instance_id, instance_type, avail_zone, geo_country_name, geo_region_name, geo_time_zone,
                 geo_lat_lon]:
        flash(info)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    for i in range(10):
        flash('')
    get_instance_info()
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            # flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = getPrediction(filename)
            for top_result in result:
                flash(top_result[0])
                flash(top_result[1])
            get_instance_info()
            return render_template('index.html', filename=filename)
        else:
            # flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
