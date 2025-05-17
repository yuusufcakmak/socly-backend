@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            print("📦 Model yükleniyor...")
            model = load_model('betting_detector_final.keras', compile=False)

        if 'file' not in request.files:
            return jsonify({'error': 'Görsel dosyası gerekli'}), 400

        file = request.files['file']
        print("⏳ Görsel alındı:", file.filename)

        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        reconstructed = model.predict(img_array)
        reconstruction_error = np.mean(np.square(img_array - reconstructed))

        threshold = 0.03  # Sabit eşik

        if reconstruction_error <= threshold:
            label = "TEHLİKELİ (bahis içerikli)"
            confidence = (1 - reconstruction_error / threshold) * 100
        else:
            label = "NORMAL (tehlikeli değil)"
            confidence = (reconstruction_error / threshold) * 100

        return jsonify({
            'prediction': label,
            'confidence_score': round(float(confidence), 2),
            'reconstruction_error': round(float(reconstruction_error), 6)
        })

    except Exception as e:
        print("❌ HATA:", str(e))
        return jsonify({'error': str(e)}), 500
