# 💖 Relationship Compatibility Predictor

Predict relationship compatibility using survey responses with the help of machine learning. This project combines **unsupervised clustering** to generate labels and a **supervised Artificial Neural Network (ANN)** to perform the final classification.

---

## 📌 Project Overview

The Relationship Compatibility Predictor is designed to analyze answers from a relationship-based questionnaire and predict whether two individuals would make a good match. The project pipeline includes:

- 📊 **Survey Data Collection** via Google Forms
- 🏷️ **Label Generation using KMeans Clustering (Unsupervised Learning)**
- 🧠 **Model Training using an Artificial Neural Network (Supervised Learning)**

This hybrid approach allows the model to:
- Discover natural clusters in respondent preferences (unsupervised),
- Learn complex patterns for accurate classification (supervised).

The final model is deployed through a **Streamlit** app, allowing users to input their answers and receive real-time compatibility predictions.


---



## 📊 1. Data Collection

- Responses gathered using a **Google Form** with 12 questions.
- Questions were designed to capture preferences and personality traits.
- **149 responses** collected and stored in CSV format.

---

## 🏷️ 2. Data Labeling

- Used **KMeans Clustering (k=2)** to label the responses as either a "match" or "no match".
- Avoided rule-based methods to reduce bias and discover natural groupings in data.

---

## 🔠 3. Preprocessing

- Applied **Label Encoding** to convert categorical responses into numerical format.
- Chosen for its compactness and order preservation.

---

## 🧠 4. Model Architecture

- Developed an **Artificial Neural Network (ANN)**:
  - 2 Hidden Layers with ReLU
  - Dropout + Batch Normalization
  - Output Layer: Sigmoid activation for binary classification
- Regularization techniques were applied to reduce overfitting.

---

## 💾 5. Model Saving

- Saved the trained model using Keras’s `.h5` format (`model (1).h5`).
- ModelCheckpoint used during training to retain the best performing weights.

---

## 🧪 6. Inference with Streamlit

- `app.py` is a **Streamlit application** that:
  - Accepts user inputs via form
  - Uses the trained model to predict compatibility
  - Displays results in real-time

---

## ✅ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/peram-mahesh.git
   cd peram-mahesh
   
2.**Install required packages**


pip install -r requirements.txt


3. **Start the Streamlit app**

streamlit run app.py

---


### 🚀 Features Real-time compatibility prediction

Easy-to-use form interface

Model built using real survey data

Compact and efficient model inference


---


## 📁 Repository Structure

```
peram-mahesh/
├── app.py                         # Streamlit app for predictions  
├── model (1).h5                   # Trained Keras model  
├── Assessment - Form Responses.csv  # Survey response dataset  
├── requirements.txt              # Python dependencies  
└── README.md                     # Project overview
```

## 👥 Authors

Developed by Peram Mahesh

• https://www.linkedin.com/in/peram-mahesh-536487269/


• https://github.com/peram-mahesh
