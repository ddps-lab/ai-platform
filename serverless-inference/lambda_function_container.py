import json
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from requests_toolbelt.multipart import decoder

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# 학습된 모델 로드
model = load_model('flower_model.keras')

def multipart_to_input(multipart_data):
    binary_content = []
    for part in multipart_data.parts:
        binary_content.append(part.content)
        
    img = BytesIO(binary_content[0])
    img = Image.open(img)
    img = img.resize((180, 180), Image.ANTIALIAS)
    img = np.array(img)
    
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    return img


def inference_model(img):
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    result = [(label, str(np.round(acc * 100, 4)) + '%') for (label, acc) in zip(class_names, score)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result


def handler(event, context):
    body = event['body-json']
    
    # 람다 생성 확인용 코드
    if body == "test":
        return {
           'statusCode': 200,
           'body' : "함수가 정상적으로 배포되었습니다."
        }
    
    body = base64.b64decode(body)
    
    boundary = body.split(b'\r\n')[0]
    boundary = boundary.decode('utf-8')
    content_type = f"multipart/form-data; boundary={boundary}"
    
    multipart_data = decoder.MultipartDecoder(body, content_type)
    
    img = multipart_to_input(multipart_data)
    result = inference_model(img)
    
    return {
        'statusCode': 200,
        'body': json.dumps(f"{result[0][0]}&{result[0][1]}&{result[1][0]}&{result[1][1]}&{result[2][0]}&{result[2][1]}&{result[3][0]}&{result[3][1]}&{result[4][0]}&{result[4][1]}")
    }
