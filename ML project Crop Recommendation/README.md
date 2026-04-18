# Soil Nutrient Based Crop Recommendation Using Decision Tree Algorithm

Welcome to AgriPredict, a complete system for recommending optimal crops based on soil nutrients and environmental factors.

## Overview
This system utilizes a highly accurate **Decision Tree Algorithm** to analyze the relationships between soil properties, weather parameters, and crop yields. By securely analyzing a soil sample image alongside environmental data such as Temperature, Humidity, and Rainfall, the application recommends the best crop to cultivate.

## Features
* **Decision Tree AI Model:** An advanced decision tree determines the best crop by finding optimal split points among the environmental parameters.
* **Instant Analysis:** Input your temperature, humidity, rainfall, and a picture of the soil to get an immediate recommendation, backed by decision tree confidence scores.
* **Full CRUD Database:** You can log your soil tests and recommendations into the built-in SQLite database for future reference. View, edit, and delete records seamlessly.
* **Analytics Dashboard:** Summarize your past recommendations and environmental averages into an intuitive pie-chart and radar map based on historical decision tree data.
* **Dynamic Crop Previews:** Instantly view beautiful, dynamically generated images of the suggested crop.

## Running the Application
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open a web browser and navigate to `http://127.0.0.1:5000`

## Built With
* Python / Flask
* Scikit-Learn (**Decision Tree Machine Learning**)
* SQLite3
* Vanilla JS, HTML, CSS, Bootstrap

_No Neural Networks or Deep Learning models were utilized in this build; all intelligence relies entirely on structured Decision Tree Machine Learning paradigms._
