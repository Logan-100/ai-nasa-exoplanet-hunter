# 🔭 NASA Exoplanet Hunter: Deep Learning & API Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 🌌 Project Overview
This end-to-end project combines **Deep Learning** with real-time **Government API Integration** to detect and analyze exoplanets. 

The core of the project is a **1D Convolutional Neural Network (CNN)** trained on raw time-series light curve data collected by NASA's Kepler Space Telescope. The model identifies the microscopic dimming of a star caused by an orbiting exoplanet (transit method). 
The AI model is then wrapped in a modern, interactive web dashboard built with **Streamlit**, which queries the **NASA Exoplanet Archive API** (Caltech IPAC) in real-time to cross-reference AI anomaly detections with official astronomical confirmations.

## ✨ Key Features
* **Time-Series Deep Learning:** Engineered a CNN tailored for 1D sequential data (stellar flux).
* **Imbalanced Data Handling:** Implemented **SMOTE** to balance the dataset (5050 negative cases vs. 37 positive cases) and prevent model bias.
* **Real-time API Integration:** Uses RESTful requests to fetch live data from the NASA Exoplanet Archive based on user queries.
* **NASA APOD Integration:** Connects to `api.nasa.gov` using private API keys to dynamically load the Astronomy Picture of the Day.
* **Advanced Data Visualization:** Utilizes `Plotly` to generate interactive bar charts comparing the newly discovered alien worlds against Earth's radius.

## 🧠 Model Architecture & Training
The CNN was built using TensorFlow/Keras and features:
1. Convolutional layers for feature extraction from light flux dips.
2. Max-Pooling for spatial dimensionality reduction.
3. Dropout layers to prevent overfitting.
4. Binary Cross-Entropy loss for classification (Planet / No Planet).
*(Note: The live Streamlit dashboard utilizes simulated inference confidence for demonstration purposes to avoid real-time gigabyte-scale FITS file downloads, while the provided `.ipynb` notebook contains the actual mathematical pipeline and training history).*

## 🚀 How to Run Locally

1. Clone this repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt