from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

# Load age detection model
AGE_PROTOTXT = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)

# Load face detector
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

def predict_age_from_image(image_bytes):
    # Convert bytes to OpenCV image
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = np.array(img)[:, :, ::-1].copy()  # PIL RGB → OpenCV BGR

    h, w = img.shape[:2]

    # Detect faces
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    ages = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)

            face = img[startY:endY, startX:endX]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
            age_net.setInput(face_blob)
            preds = age_net.forward()
            age = AGE_LIST[preds[0].argmax()]
            ages.append(age)

    if ages:
        return ages  # Returns list of ages if multiple faces
    else:
        return ["No face detected"]

@app.route('/predict_age', methods=['POST'])
def predict_age():
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_b64 = data['image']
    if image_b64.startswith('data:'):
        comma = image_b64.find(',')
        if comma >= 0:
            image_b64 = image_b64[comma+1:]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return jsonify({'error': 'Invalid image/base64: ' + str(e)}), 400

    try:
        ages = predict_age_from_image(image_bytes)
    except Exception as e:
        return jsonify({'error': 'Model error: ' + str(e)}), 500

    return jsonify({'age': ages}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
