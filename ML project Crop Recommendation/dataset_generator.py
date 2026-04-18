import pandas as pd
import numpy as np
import random

def generate_crop_data():
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic ranges for each crop based on agricultural standards
    # columns: temperature, humidity, rainfall, soil_type, label
    crop_profiles = {
        'rice': {'soil_types': ['alluvial', 'clay'], 'temperature': (20, 27), 'humidity': (80, 85), 'rainfall': (180, 260)},
        'maize': {'soil_types': ['red', 'black', 'alluvial'], 'temperature': (18, 27), 'humidity': (55, 75), 'rainfall': (60, 110)},
        'chickpea': {'soil_types': ['sandy', 'alluvial'], 'temperature': (17, 21), 'humidity': (14, 20), 'rainfall': (65, 95)},
        'kidneybeans': {'soil_types': ['alluvial', 'red'], 'temperature': (15, 25), 'humidity': (18, 24), 'rainfall': (60, 150)},
        'pigeonpeas': {'soil_types': ['alluvial', 'sandy'], 'temperature': (18, 37), 'humidity': (30, 70), 'rainfall': (90, 200)},
        'mothbeans': {'soil_types': ['sandy', 'red'], 'temperature': (24, 32), 'humidity': (40, 65), 'rainfall': (30, 75)},
        'mungbean': {'soil_types': ['alluvial', 'sandy'], 'temperature': (27, 30), 'humidity': (80, 90), 'rainfall': (36, 60)},
        'blackgram': {'soil_types': ['alluvial', 'black'], 'temperature': (25, 35), 'humidity': (60, 68), 'rainfall': (60, 75)},
        'lentil': {'soil_types': ['alluvial', 'clay'], 'temperature': (18, 30), 'humidity': (60, 69), 'rainfall': (35, 55)},
        'pomegranate': {'soil_types': ['red', 'alluvial'], 'temperature': (18, 24), 'humidity': (85, 95), 'rainfall': (100, 115)},
        'banana': {'soil_types': ['alluvial', 'clay'], 'temperature': (25, 30), 'humidity': (75, 85), 'rainfall': (90, 120)},
        'mango': {'soil_types': ['alluvial', 'red', 'black'], 'temperature': (27, 36), 'humidity': (45, 55), 'rainfall': (85, 105)},
        'grapes': {'soil_types': ['sandy', 'alluvial'], 'temperature': (8, 42), 'humidity': (80, 84), 'rainfall': (65, 75)},
        'watermelon': {'soil_types': ['sandy', 'alluvial'], 'temperature': (24, 26), 'humidity': (80, 90), 'rainfall': (40, 60)},
        'muskmelon': {'soil_types': ['sandy', 'alluvial'], 'temperature': (27, 30), 'humidity': (90, 95), 'rainfall': (20, 30)},
        'apple': {'soil_types': ['alluvial', 'red'], 'temperature': (21, 24), 'humidity': (90, 95), 'rainfall': (100, 125)},
        'orange': {'soil_types': ['alluvial', 'red'], 'temperature': (10, 35), 'humidity': (90, 95), 'rainfall': (100, 120)},
        'papaya': {'soil_types': ['alluvial', 'black'], 'temperature': (23, 44), 'humidity': (90, 95), 'rainfall': (40, 250)},
        'coconut': {'soil_types': ['sandy', 'alluvial'], 'temperature': (25, 30), 'humidity': (90, 100), 'rainfall': (130, 230)},
        'cotton': {'soil_types': ['black', 'clay'], 'temperature': (22, 26), 'humidity': (75, 85), 'rainfall': (60, 100)},
        'jute': {'soil_types': ['alluvial', 'clay'], 'temperature': (23, 27), 'humidity': (70, 90), 'rainfall': (150, 200)},
        'coffee': {'soil_types': ['red', 'alluvial'], 'temperature': (23, 28), 'humidity': (50, 70), 'rainfall': (115, 200)}
    }

    data = []
    # Generate 150 samples per crop
    for label, ranges in crop_profiles.items():
        for _ in range(150):
            row = {
                'temperature': np.random.uniform(ranges['temperature'][0], ranges['temperature'][1]),
                'humidity': np.random.uniform(ranges['humidity'][0], ranges['humidity'][1]),
                'rainfall': np.random.uniform(ranges['rainfall'][0], ranges['rainfall'][1]),
                'soil_type': random.choice(ranges['soil_types']),
                'label': label
            }
            data.append(row)

    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('dataset.csv', index=False)
    print("New dataset.csv successfully generated with {} rows.".format(len(df)))

if __name__ == '__main__':
    generate_crop_data()
