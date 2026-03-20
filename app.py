import streamlit as st
import librosa
import numpy as np
from textblob import TextBlob
import speech_recognition as sr
from sklearn.ensemble import RandomForestClassifier
import tempfile
import os
st.title("🎤 Speech Sentiment Analyzer")

st.write("Upload an audio file (.wav) and get sentiment + prediction")

audio_file = st.file_uploader("Upload Audio", type=["wav"])

model = RandomForestClassifier()
X_dummy = np.array([[0], [1]])
y_dummy = np.array([0, 1])
model.fit(X_dummy, y_dummy)

if audio_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        temp_audio_path = tmp.name

    st.audio(temp_audio_path)

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        st.subheader("📝 Transcribed Text:")
        st.write(text)

      
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        st.subheader("📊 Sentiment Score:")
        st.write(sentiment)

        if sentiment > 0:
            st.success("Positive 😊")
        elif sentiment < 0:
            st.error("Negative 😠")
        else:
            st.info("Neutral 😐")
        y, sr_rate = librosa.load(temp_audio_path)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr_rate))

      
        prediction = model.predict([[mfcc]])

        st.subheader("ML Prediction:")
        st.write(prediction[0])

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        os.remove(temp_audio_path)
