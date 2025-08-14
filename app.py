from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS so React frontend can access

# ✅ Model Path
model = None
model_path = os.path.join("model", "my_model.keras")
print("🔍 Checking if model exists:", os.path.exists(model_path))

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", str(e))
else:
    print("❌ Model file not found at:", model_path)

# ✅ Your class labels (same order as during training)
class_labels = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# ✅ Home route (for testing)
@app.route('/')
def home():
    return "🚀 Flask Waste Classifier API is running!"

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    try:
        # 🖼️ Preprocess image
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # 🔮 Prediction
        prediction = model.predict(image_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
