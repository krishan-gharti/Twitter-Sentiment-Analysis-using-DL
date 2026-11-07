import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences 

import numpy as np

## Loading the tensorflow model fro prediction

model = load_model('rnn_model.h5')

with open("tokenizer.pkl","rb")as f:
    tokenizer=pickle.load(f)

st.title("Tweeter tweets Sentiment Analysis")

tweet=st.text_area("Enter the tweet to predict the sentiment of the tweet: ")

if st.button("Predict Sentiment") and tweet.strip():
    sequence = tokenizer.texts_to_sequences([tweet])
    padded_sequence = pad_sequences(sequence, maxlen=117, padding='pre')
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]

    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    st.write(f"Predicted Sentiment: {sentiment_mapping[predicted_class]}")