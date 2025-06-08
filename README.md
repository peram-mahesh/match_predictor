# 💖 Relationship Compatibility Predictor

Predict relationship compatibility using survey data and a machine learning model trained with unsupervised clustering.

---

## 📌 Project Overview

This project aims to determine match potential between individuals based on their responses to a compatibility survey. Using **unsupervised learning** (KMeans clustering) and a neural network classifier, it predicts whether two people are likely to be a good match.

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


### 🚀 Features Real-time compatibility prediction

Easy-to-use form interface

Model built using real survey data

Compact and efficient model inference

👥 Authors
Developed by Peram Mahesh

• https://www.linkedin.com/in/peram-mahesh-536487269/


• https://github.com/peram-mahesh
