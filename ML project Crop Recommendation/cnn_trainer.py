import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Soil Categories
SOIL_TYPES = ['red', 'black', 'clay', 'sandy', 'alluvial']

def generate_synthetic_soil_images(samples_per_class=100, img_size=(64, 64)):
    """Generate synthetic images to represent different soil types to bypass downloading massive datasets."""
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

    print("Generating synthetic images for CNN...")
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
            
            X.append(img)
            y.append(i)

    # Normalize images
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    return X, y

def train_cnn():
    X, y = generate_synthetic_soil_images(samples_per_class=200)
    
    print(f"Generated {len(X)} images of size {X[0].shape}.")
    
    # Define lightweight CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(SOIL_TYPES), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training Convolutional Neural Network (CNN) for Soil Image Classification...")
    # Train
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save Model
    model.save('soil_model.h5')
    print("Saved CNN model to 'soil_model.h5'!")
    
    # Save categories map
    with open('soil_classes.pkl', 'wb') as f:
        pickle.dump(SOIL_TYPES, f)
    print("Saved 'soil_classes.pkl' mapping.")

if __name__ == "__main__":
    train_cnn()
