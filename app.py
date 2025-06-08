import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# --- Page config ---
st.set_page_config(page_title="Relationship Compatibility Predictor", layout="centered")

# --- Title ---
st.title("💘 Relationship Compatibility Predictor")
st.write("Answer the questions to see if you're a potential match!")

# --- Load CSV ---
df = pd.read_csv("Assessment - Form Responses.csv")
df.drop(columns=["Timestamp", "Email Address"], inplace=True)

# --- Define Target Logic ---
reference_data = {
    'How spontaneous are you?': "Balanced",
    'Do you enjoy giving or receiving surprises?\n': "Love it",
    '\nHow important is music taste compatibility to you?\n': "Very important, must match mine",
    'How open are you to trying new things (food, travel, experiences)?': "Mostly open",
    'How much personal space do you need in a relationship?': "Moderate",
    'How emotionally expressive are you?': "Balanced",
    'How important is having similar long-term goals?': "Important",
    'What’s your preferred mode of communication?': "Face to Face",
    'What’s your ideal time to hang out?': "Evening",
    'What is your ideal weekend plan?': "Outdoor Adventures"
}

def calculate_score(row):
    return sum(row.get(key) == value for key, value in reference_data.items())

# Generate target column
df['score'] = df.apply(calculate_score, axis=1)
df['target'] = df['score'].apply(lambda x: "match" if x > 4 else "not match")
df.drop(columns="score", inplace=True)

# --- Features and Labels ---
X = df.drop('target', axis=1)
y = df['target']

# --- Column Categories ---
nominal_cols = [
    'Do you enjoy giving or receiving surprises?\n',
    'What’s your preferred mode of communication?',
    'What is your ideal weekend plan?'
]

ordinal_cols = [
    'How spontaneous are you?',
    '\nHow important is music taste compatibility to you?\n',
    'How open are you to trying new things (food, travel, experiences)?',
    'How much personal space do you need in a relationship?',
    'How emotionally expressive are you?',
    'How important is having similar long-term goals?',
    'What’s your ideal time to hang out?'
]

ordinal_orders = [
    ['Very planned', 'Mostly planned', 'Balanced', 'Mostly spontaneous', 'Very spontaneous'],
    ['Doesn’t matter at all', 'Slightly important', 'Neutral / Okay either way', 'Important but not a deal-breaker', 'Very important, must match mine'],
    ['Not open at all', 'Slightly hesitant', 'Sometimes open', 'Mostly open', 'Very adventurous'],
    ['Very little', 'A little', 'Moderate', 'Quite a bit', 'A lot'],
    ['Very reserved', 'Slightly reserved', 'Balanced', 'Mostly expressive', 'Very expressive'],
    ['Not important', 'Slightly important', 'Neutral', 'Important', 'Very important'],
    ['Morning', 'Afternoon', 'Evening', 'Late night']
]

# --- Pipelines ---
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ordinal_orders))
])

nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('ord', ordinal_pipeline, ordinal_cols),
    ('nom', nominal_pipeline, nominal_cols)
])

# Fit the preprocessor on the full dataset
preprocessor.fit(X)
le = LabelEncoder()
le.fit(y)

# --- Load the Trained Model ---
model = keras.models.load_model("model.h5")

# --- User Input via Sidebar ---
st.sidebar.header("🧠 Answer these to predict your match:")
user_input = {}

for col in ordinal_cols:
    user_input[col] = st.sidebar.selectbox(col.strip(), ordinal_orders[ordinal_cols.index(col)])

for col in nominal_cols:
    unique_vals = df[col].dropna().unique().tolist()
    user_input[col] = st.sidebar.selectbox(col.strip(), unique_vals)

# --- Prediction ---
if st.button("Predict Match"):
    input_df = pd.DataFrame([user_input])
    input_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_transformed)[0][0]

    if prediction > 0.5:
        st.markdown("## 🥳✨ **It’s a Match! Perfect vibes detected!** 💫💕")
        st.image("https://media.giphy.com/media/l3vR85PnGsBwu1PFK/giphy.gif", width=400)
        st.success("Congratulations! Go plan your next adventure together! 🚀")
    else:
        st.markdown("## 😞💔 **Not a Match... Maybe next time.**")
        st.image("https://media.giphy.com/media/3og0IPxMM0erATueVW/giphy.gif", width=400)
        st.error("Sometimes it's just not meant to be. But hey, don't give up! 🌈")



