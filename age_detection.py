import cv2
import numpy as np
from PIL import Image
import os

# ----------------------
# 1. Load Models
# ----------------------

# Face Detection Model (TensorFlow)
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

# Age Prediction Model (Caffe)
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

# Mean values and age ranges
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# ----------------------
# 2. Detect Faces
# ----------------------
def detect_faces(frame, conf_threshold=0.7):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes

# ----------------------
# 3. Predict Age
# ----------------------
def predict_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    return age

# ----------------------
# 4. Convert PNG to JPG if needed
# ----------------------
def convert_to_jpg(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        img = Image.open(image_path).convert("RGB")
        new_path = os.path.splitext(image_path)[0] + ".jpg"
        img.save(new_path)
        print(f"Converted PNG to JPG: {new_path}")
        return new_path
    return image_path

# ----------------------
# 5. Process Image
# ----------------------
def process_image(image_path):
    image_path = convert_to_jpg(image_path)  # Convert PNG to JPG if needed
    print(f"Trying to open image: '{image_path}'")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Image not found or cannot be read at '{image_path}'")
        return

    face_boxes = detect_faces(frame)
    if not face_boxes:
        print("No faces detected in the image.")
        return

    for (x1, y1, x2, y2) in face_boxes:
        face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1),
                     max(0, x1-20):min(x2+20, frame.shape[1]-1)]
        age = predict_age(face)
        cv2.putText(frame, f"Age: {age}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Age Prediction", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------
# 6. Main
# ----------------------
if __name__ == "__main__":
    image_path = input("Enter the image file name (with extension, e.g., oldman_image.jpg): ").strip()
    process_image(image_path)
