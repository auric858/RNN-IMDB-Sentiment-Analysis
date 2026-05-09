import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
# LOAD THE IMDB WORD INDEX
word_index=imdb.get_word_index() ## gets all word indexes
reverse_word_index={value:key for key,value in word_index.items()}

## load pre trained model 
model=load_model('simple_rnn_imdb.h5')


## helper functions
## function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],padding='pre',maxlen=100)
    return padded_review


## prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0]>0.5 else 'negative'
    return sentiment,prediction[0][0]


## DESIGINING STREAMLIT WEB APP
st.title("IMDB MOVIE REVIEW SENTIMENT ANALYSIS")
st.write("Enter a movie review to classify it as positive or negative.")

user_input=st.text_area("Movie Review")

if st.button("Classify"):
    ## make prediction
    sentiment,score=predict_sentiment(user_input)

    ## display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score:.2f}')
else:
    st.write("enter a movie review.")

    


