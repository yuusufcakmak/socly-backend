from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)  # ← BU satır en üstte olacak, decorator'lardan önce

model = None  # lazy load

@app.route('/')
def home():
    return 'Socly API Çalışıyor!'
@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            print("📦 Model yükleniyor...")
            model = load_model('betting_detector_final.keras')  # classifier model

        if 'file' not in request.files:
            return jsonify({'error': 'Görsel dosyası gerekli'}), 400

        file = request.files['file']
        print("⏳ Görsel alındı:", file.filename)

        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        label = "TEHLİKELİ (bahis içerikli)" if prediction >= 0.5 else "NORMAL (tehlikeli değil)"
        confidence = round(float(prediction) * 100, 2) if prediction >= 0.5 else round((1 - prediction) * 100, 2)

        return jsonify({
            'prediction': label,
            'confidence_score': confidence,
            'raw_score': round(float(prediction), 4)
        })

    except Exception as e:
        print("❌ HATA:", str(e))
        return jsonify({'error': str(e)}), 500
