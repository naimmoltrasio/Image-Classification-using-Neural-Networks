from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('mnist_cnn_model.h5')


def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image', None)

        if image_data is None:
            return jsonify({'error': 'No image data found'}), 400

        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image).argmax()

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
