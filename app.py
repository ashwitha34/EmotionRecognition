from flask import Flask, request, jsonify, send_from_directory
from keras.models import model_from_json
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load model architecture and weights
with open(r"C:/Users/ashwitha reddy/Desktop/emotion_recognition/backend/model.json", 'r') as json_file:
    loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights('model.weights.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def process_image(image_data):
    # Convert base64 image data to numpy array
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale and resize to 48x48
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(gray_frame, (48, 48))

    # Preprocess the image for the model
    roi = roi_gray.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension if needed

    return roi

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    data = request.get_json()
    image_data = data['image']
    processed_image = process_image(image_data)

    # Predict the emotion
    emotion_prediction = emotion_model.predict(processed_image)[0]
    max_index = np.argmax(emotion_prediction)
    emotion_label = emotion_labels[max_index]

    return jsonify({'emotion': emotion_label})

if __name__ == '__main__':
    app.run(debug=True)
