from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from dotenv import load_dotenv

load_dotenv() # .env dosyasındaki değişkenleri yükler

app = Flask(__name__)  # BU SATIR ŞART ‼️
CORS(app)  # Mobil ve web erişimi için şart

model = None  # Lazy load

@app.route('/')
def home():
    return 'Socly API Çalışıyor!'

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            print("📦 Model yükleniyor...")
            model = load_model('betting_detector_final.keras')  # Classifier model

        if 'file' not in request.files:
            return jsonify({'error': 'Görsel dosyası gerekli'}), 400

        file = request.files['file']
        print("⏳ Görsel alındı:", file.filename)

        # Görseli işle
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin yap
        prediction = float(model.predict(img_array)[0][0])
        label = "TEHLİKELİ (bahis içerikli)" if prediction >= 0.5 else "NORMAL (tehlikeli değil)"
        confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

        return jsonify({
            'prediction': label,
            'confidence_score': round(confidence, 2),
            'raw_score': round(prediction, 4)
        })

    except Exception as e:
        print("❌ HATA:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/report-to-meta', methods=['POST'])
def report_to_meta():
    try:
        if 'screenshot' not in request.files:
            return jsonify({'message': 'Ekran görüntüsü dosyası gerekli'}), 400

        screenshot = request.files['screenshot']
        prediction = request.form.get('prediction', 'Unknown')
        score = request.form.get('score', 'N/A')
        analyzedUser = request.form.get('analyzedUser', 'Bilinmeyen Hesap')
        reporterUserId = request.form.get('reporterUserId', 'anonymous')

        # E-posta Ayarları
        sender_email = os.environ.get('SENDER_EMAIL') # Ortam değişkeninden al
        sender_password = os.environ.get('SENDER_PASSWORD') # Ortam değişkeninden al
        
        receiver_email = "yusuf.cakmak.2000@gmail.com"

        if not sender_email or not sender_password:
            print("❌ HATA: Gönderici e-posta veya şifre ayarlanmamış.")
            return jsonify({'message': 'E-posta gönderimi için sunucu konfigürasyonu eksik.'}), 500

        # E-posta Oluşturma
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"Şüpheli Hesap Bildirimi: {analyzedUser}"

        body = f"""
        Merhaba Meta Destek Ekibi,

        Aşağıdaki hesap, uygulamamız aracılığıyla bir kullanıcı tarafından şüpheli olarak bildirilmiştir:

        Analiz Edilen Hesap: {analyzedUser}
        Risk Tahmini: {prediction}
        Risk Skoru: {score}
        Raporu Yapan Kullanıcı ID: {reporterUserId}

        Bu hesap, uygulamamız tarafından yapılan analiz sonucunda potansiyel olarak riskli bulunmuştur ve ekteki ekran görüntüsü ile birlikte incelenmesi için bildirilmektedir.

        Saygılarımızla,
        Socly Uygulaması Destek Ekibi
        """
        msg.attach(MIMEText(body, 'plain'))

        # Ekran Görüntüsü Ekleme
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(screenshot.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={screenshot.filename}")
        msg.attach(part)

        # E-posta Gönderme
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587) # Örnek: Gmail SMTP sunucusu
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            print("✅ Rapor e-postası başarıyla gönderildi.")
            return jsonify({'message': 'Rapor başarıyla gönderildi.'}), 200
        except Exception as e:
            print(f"❌ E-posta gönderme hatası: {str(e)}")
            return jsonify({'message': f'Rapor gönderilirken bir e-posta sunucu hatası oluştu: {str(e)}'}), 500

    except Exception as e:
        print(f"❌ Raporlama endpoint hatası: {str(e)}")
        return jsonify({'message': f'Rapor işlenirken bir hata oluştu: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
