from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def predict_age_from_image(image_bytes):
    # Example: always returns 65 for demo — replace with ML prediction.
    return 65

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
        img = Image.open(BytesIO(image_bytes))  # sanity check
    except Exception as e:
        return jsonify({'error': 'Invalid image/base64: ' + str(e)}), 400

    try:
        age = predict_age_from_image(image_bytes)
    except Exception as e:
        return jsonify({'error': 'Model error: ' + str(e)}), 500

    return jsonify({'age': age}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
