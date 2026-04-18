import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Soil Categories
SOIL_TYPES = ['red', 'black', 'clay', 'sandy', 'alluvial']

def generate_synthetic_soil_images(samples_per_class=200, img_size=(64, 64)):
    """Generate synthetic 1D flattened images to represent different soil types to bypass downloading massive datasets."""
    X = []
    y = []

    # Base RGB colors approximately matching soil types
    base_colors = {
        'red': [150, 50, 30],       # Reddish-brown
        'black': [40, 40, 40],      # Very dark
        'clay': [180, 130, 90],     # Light smooth brown
        'sandy': [210, 180, 140],   # Yellow-brown
        'alluvial': [120, 100, 80]  # Dark grey-brown
    }

    print("Generating synthetic images for Deep Neural Network...")
    np.random.seed(42)
    
    for i, soil in enumerate(SOIL_TYPES):
        color = base_colors[soil]
        for _ in range(samples_per_class):
            # Create base image
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            img[:, :] = color
            
            # Add random noise/texture to make it realistic
            noise = np.random.normal(0, 20, (img_size[0], img_size[1], 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Flatten for MLP Classifier
            X.append(img.flatten())
            y.append(i)

    # Normalize images
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    return X, y

def train_dnn():
    X, y = generate_synthetic_soil_images(samples_per_class=300)
    
    print(f"Generated {len(X)} images of size {X[0].shape}. Training Deep Neural Network (MLP)...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Deep Neural Network Setup
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"DNN Model Accuracy: {acc * 100:.2f}%")
    
    # Save Model
    with open('soil_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved DNN model to 'soil_model.pkl'!")
    
    # Save categories map
    with open('soil_classes.pkl', 'wb') as f:
        pickle.dump(SOIL_TYPES, f)
    print("Saved 'soil_classes.pkl' mapping.")

if __name__ == "__main__":
    train_dnn()
