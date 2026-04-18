import os
import sqlite3
import numpy as np
import pickle
import logging
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'database.db')
CROP_MODEL_FILE = os.path.join(BASE_DIR, 'crop_pipeline.pkl')
SOIL_MODEL_FILE = os.path.join(BASE_DIR, 'soil_model.pkl')
SOIL_CLASSES_FILE = os.path.join(BASE_DIR, 'soil_classes.pkl')

# Helper to pre-populate DB with 150 items so analytics works instantly
def seed_database(conn):
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM soil_data')
    if c.fetchone()[0] == 0:
        logging.info("Seeding DB with static fallback data for Analytics...")
        from dataset_generator import generate_crop_data
        import pandas as pd
        if not os.path.exists('dataset.csv'):
            generate_crop_data()
            
        df = pd.read_csv('dataset.csv')
        # Just grab random 150 samples
        df_sample = df.sample(n=150, random_state=42)
        
        for _, row in df_sample.iterrows():
            c.execute('''
                INSERT INTO soil_data (temperature, humidity, rainfall, soil_type, recommended_crop)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                float(row['temperature']), float(row['humidity']), 
                float(row['rainfall']), str(row['soil_type']), 
                str(row['label'])
            ))
        conn.commit()
    
def init_db():
    try:
        # We will drop the old schema and recreate if N exists
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("PRAGMA table_info(soil_data)")
        columns = [info[1] for info in c.fetchall()]
        if 'N' in columns:
            logging.info("Old DB Schema detected. Dropping old soil_data table.")
            c.execute("DROP TABLE soil_data")
            
        c.execute('''
            CREATE TABLE IF NOT EXISTS soil_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                rainfall REAL NOT NULL,
                soil_type TEXT NOT NULL,
                recommended_crop TEXT
            )
        ''')
        conn.commit()
        seed_database(conn)
        conn.close()
        logging.info("Database initialized.")
    except Exception as e:
        logging.error(f"DB Init Error: {e}")

with app.app_context():
    init_db()

# Load Models
def load_models():
    crop_pipeline, soil_model, soil_classes = None, None, None
    if os.path.exists(CROP_MODEL_FILE):
        with open(CROP_MODEL_FILE, 'rb') as f:
            crop_pipeline = pickle.load(f)
    if os.path.exists(SOIL_CLASSES_FILE):
        with open(SOIL_CLASSES_FILE, 'rb') as f:
            soil_classes = pickle.load(f)
    if os.path.exists(SOIL_MODEL_FILE):
        with open(SOIL_MODEL_FILE, 'rb') as f:
            soil_model = pickle.load(f)
    
    return crop_pipeline, soil_model, soil_classes

crop_pipeline, soil_model, soil_classes = load_models()

# Load PyTorch Validation Model
print("Loading PreTrained PyTorch ImageValidator...")
validation_weights = MobileNet_V3_Small_Weights.DEFAULT
validation_model = mobilenet_v3_small(weights=validation_weights)
validation_model.eval()
validation_preprocess = validation_weights.transforms()
imagenet_categories = validation_weights.meta["categories"]

def is_soil_image(image_bytes):
    """Uses a pre-trained ImageNet model to verify if the uploaded picture is actually soil/ground/dirt"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    batch = validation_preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        prediction = validation_model(batch).squeeze(0).softmax(0)
        
    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(prediction, 5)
    
    # Check if any of the top 5 descriptions relate to ground/soil/dirt/mud/sand/field
    valid_keywords = ['soil', 'dirt', 'earth', 'ground', 'sand', 'mud', 'field', 'pot', 'plow', 'tractor']
    for i in range(top5_prob.size(0)):
        category_name = imagenet_categories[top5_catid[i]].lower()
        if any(keyword in category_name for keyword in valid_keywords):
            return True, category_name
            
    # If we got here, it's likely a face, car, animal, etc.
    top_prediction = imagenet_categories[top5_catid[0]]
    return False, top_prediction

def process_image(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    return img_array.flatten().reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded for Soil Analysis'}), 400
        
        file = request.files['image']
        temp = request.form.get('temperature')
        file_bytes = file.read()

        # 0. Pre-Validation
        # is_valid, detected_object = is_soil_image(file_bytes)
        # if not is_valid:
        #     return jsonify({
        #         'error': f'Invalid Image: This looks like a "{detected_object}". Please choose a correct image of soil for prediction.'
        #     }), 400

        # 1. Decision Tree Classification for Soil Type
        if not soil_model or not soil_classes:
            return jsonify({'error': 'Decision Tree Soil Model not found on the server.'}), 500
        
        img_array = process_image(file_bytes)
        humidity = request.form.get('humidity')
        rainfall = request.form.get('rainfall')
        
        if not temp or not humidity or not rainfall:
            return jsonify({'error': 'Missing environmental inputs'}), 400
            
        temp = float(temp)
        humidity = float(humidity)
        rainfall = float(rainfall)


        # Decision Tree Predict
        soil_class_idx = soil_model.predict(img_array)[0]
        try:
            probs = soil_model.predict_proba(img_array)[0]
            soil_confidence = float(max(probs)) * 100
        except:
            soil_confidence = 85.0
            
        if soil_confidence <= 50.0:
            return jsonify({'error': 'choose a correct image of soil for prediction'}), 400
            
        predicted_soil = soil_classes[soil_class_idx]

        # 2. Decision Tree for Crop Recommendation
        if not crop_pipeline:
            return jsonify({'error': 'ML crop model not loaded.'}), 500

        # Construct input dataframe matching training shape
        import pandas as pd
        input_df = pd.DataFrame([{
            'temperature': temp,
            'humidity': humidity,
            'rainfall': rainfall,
            'soil_type': predicted_soil
        }])

        crop_prediction = crop_pipeline.predict(input_df)[0]
        
        # Calculate confidence if method exists
        try:
            crop_probs = crop_pipeline.predict_proba(input_df)[0]
            crop_confidence = float(max(crop_probs)) * 100
        except:
            crop_confidence = 85.0 # fallback

        logging.info(f"Soil: {predicted_soil} ({soil_confidence:.1f}%), Crop: {crop_prediction}")
        return jsonify({
            'soil_type': predicted_soil,
            'soil_confidence': f"{soil_confidence:.2f}%",
            'recommended_crop': str(crop_prediction).capitalize(),
            'crop_confidence': f"{crop_confidence:.2f}%"
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/soil', methods=['GET'])
def get_soil_data():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM soil_data ORDER BY id DESC')
        rows = c.fetchall()
        conn.close()
        
        return jsonify([dict(row) for row in rows]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/soil', methods=['POST'])
def add_soil_data():
    try:
        data = request.get_json()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO soil_data (temperature, humidity, rainfall, soil_type, recommended_crop)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            float(data['temperature']), float(data['humidity']), 
            float(data['rainfall']), str(data['soil_type']), 
            str(data['recommended_crop'])
        ))
        conn.commit()
        inserted_id = c.lastrowid
        conn.close()
        return jsonify({'message': 'Record added', 'id': inserted_id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/soil/<int:id>', methods=['PUT'])
def update_soil_data(id):
    try:
        data = request.get_json()
        fields, values = [], []
        for key in ['temperature', 'humidity', 'rainfall', 'soil_type', 'recommended_crop']:
            if key in data:
                fields.append(f"{key} = ?")
                values.append(data[key])
                
        if not fields: return jsonify({'error': 'No fields provided'}), 400
        values.append(id)
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(f"UPDATE soil_data SET {', '.join(fields)} WHERE id = ?", tuple(values))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/soil/<int:id>', methods=['DELETE'])
def delete_soil_data(id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('DELETE FROM soil_data WHERE id = ?', (id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT recommended_crop, COUNT(*) as count FROM soil_data GROUP BY recommended_crop')
        crop_dist = {row['recommended_crop']: row['count'] for row in c.fetchall()}
        
        c.execute('SELECT soil_type, COUNT(*) as count FROM soil_data GROUP BY soil_type')
        soil_dist = {row['soil_type']: row['count'] for row in c.fetchall()}
        
        c.execute('SELECT AVG(temperature) as avg_T, AVG(humidity) as avg_H, AVG(rainfall) as avg_R FROM soil_data')
        avg_row = c.fetchone()
        averages = dict(avg_row) if avg_row else {'avg_T': 0, 'avg_H': 0, 'avg_R': 0}
            
        conn.close()
        
        return jsonify({
            'crop_distribution': crop_dist,
            'soil_distribution': soil_dist,
            'averages': averages,
            'total_records': sum(crop_dist.values())
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
