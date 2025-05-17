from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('betting_detector_final.keras')

@app.route('/')
def home():
    return 'Socly API Çalışıyor!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Görsel dosyası gerekli'}), 400

        file = request.files['file']

        # Dosyayı diske yazmadan bellekte işle
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = 'RİSKLİ HESAP' if prediction >= 0.5 else 'TEMİZ HESAP'

        return jsonify({
            'prediction': label,
            'risk_score': round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
