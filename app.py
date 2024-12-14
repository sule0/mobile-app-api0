from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import io
from PIL import Image  # Resimleri açmak için kullanılır

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)
# TFLite modelini yükle
interpreter = tf.lite.Interpreter('./inceptionv3last.tflite')
interpreter.allocate_tensors()

# Giriş ve çıkış tensörlerini alın
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sınıf etiketlerini tanımla
class_labels = ['Kıyafet', 'Cam', 'Metal', 'Kağıt', 'Plastik', 'Ayakkabı']

# Resim boyutları
img_height = 224
img_width = 224

# Ana sayfa
@app.route('/')
def index():
    return "Atık Tanıma API'sine Hoş Geldiniz"

# Tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Dosyayı okuyun ve resmi yükleyin
        img = Image.open(io.BytesIO(file.read())).resize((img_width, img_height))  # Resmi yeniden boyutlandırın
        img_array = np.array(img, dtype=np.float32)  # Resmi array'e dönüştürün
        img_array = np.expand_dims(img_array, axis=0)  # 4 boyutlu hale getirin (1, img_height, img_width, 3)
        img_array /= 255.0  # Normalizasyon

        # Model ile tahmin yap
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # En yüksek tahmin edilen sınıfın indeksini alın
        predicted_index = np.argmax(predictions, axis=1)[0]

        # Sınıf adını belirleyin
        predicted_class = class_labels[predicted_index]

        # Confidence skorunu alın
        confidence = float(predictions[0][predicted_index]) * 100  # Yüzdelik format
        
        

        # Sonuçları JSON formatında döndürün
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence  })

    except Exception as e:
        return jsonify({'error': str(e)})

# Uygulamayı çalıştırın
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
