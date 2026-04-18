import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

def train_and_save_model():
    print("Loading new dataset...")
    if not os.path.exists('dataset.csv'):
        print("dataset.csv not found! Run dataset_generator.py first.")
        return

    df = pd.read_csv('dataset.csv')

    # Features and Target
    X = df[['temperature', 'humidity', 'rainfall', 'soil_type']]
    y = df['label']

    print(f"Dataset shape: {df.shape}")
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing for categorical features
    categorical_features = ['soil_type']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep temp, humidity, rainfall unchanged
    )

    print("Training Decision Tree Classifier Pipeline...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy Score: {acc * 100:.2f}%")

    # Save the pipeline
    with open('crop_pipeline.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nPipeline saved successfully as 'crop_pipeline.pkl'!")

if __name__ == '__main__':
    train_and_save_model()
