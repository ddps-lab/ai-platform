import json
import os
import base64
import sys
from io import BytesIO

# sys.path.append("/mnt/efs/packages")
import numpy as np
from PIL import Image
from requests_toolbelt.multipart import decoder

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# model = load_model("/mnt/efs/packages/mobilenetv2")
model = load_model('/var/task/ai-platform/serverless-inference/model/mobilenet_v2')

def multipart_to_input(multipart_data):
    binary_content = []
    for part in multipart_data.parts:
        binary_content.append(part.content)

    img = BytesIO(binary_content[0])
    img = Image.open(img)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img)
    
    # 1, 224, 224, 3
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    return img

def decode_predictions(preds, top=5):
#     with open('/mnt/efs/packages/imagenet_class_index.json') as f:
#         CLASS_INDEX = json.load(f)
    with open('/var/task/ai-platform/serverless-inference/model/imagenet_class_index.json') as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def inference_model(img):
    result = model.predict(img)
    result = decode_predictions(result)[0]
    result = [(img_class, label, str(round(acc * 100, 4)) + '%') for img_class, label, acc in result]
    return result
    
def handler(event, context):
    
    body = event['body-json']
    
    # 람다 생성 확인용 코드
    if body == "test":
        return {
           'statusCode': 200,
           'body' : "함수가 정상적으로 배포 되었습니다."
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
        'body': json.dumps(f"{result[0][1]}&{result[0][2]}&{result[1][1]}&{result[1][2]}&{result[2][1]}&{result[2][2]}&{result[3][1]}&{result[3][2]}&{result[4][1]}&{result[4][2]}")
    }
