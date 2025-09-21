import logging

logging.basicConfig(level=logging.INFO)

@app.route('/predict_age', methods=['POST'])
def predict_age():
    data = request.get_json(silent=True)
    logging.info(f"Received request data keys: {data.keys() if data else 'No data'}")

    if not data or 'image' not in data:
        logging.error('No image key found in request')
        return jsonify({'error': 'No image provided'}), 400

    image_b64 = data['image']
    try:
        image_bytes = base64.b64decode(image_b64)
        logging.info(f"Decoded image bytes length: {len(image_bytes)}")
    except Exception as e:
        logging.error(f"Image decoding failed: {str(e)}")
        return jsonify({'error': 'Invalid image/base64'}), 400

    try:
        age = predict_age_from_image(image_bytes)
        logging.info(f"Predicted age: {age}")
    except Exception as e:
        logging.error(f"Model error: {str(e)}")
        return jsonify({'error': 'Model error'}), 500

    return jsonify({'age': age}), 200
