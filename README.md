# AI-LIE-DETECTION
Lie Detection using Voice + Text ML

AI-Based Lie Detection System
Overview
The AI-Based Lie Detection System is a machine learning-driven application designed to analyze human speech and identify potential deception. By leveraging audio signal processing and natural language analysis, the system evaluates vocal characteristics and linguistic patterns to classify statements as truthful or deceptive.

Objectives
To develop an automated system for detecting deception using speech data
To integrate audio feature extraction with machine learning techniques
To provide a user-friendly interface for real-time or recorded input analysis

Key Features
Audio input via recording or file upload
Speech-to-text conversion for linguistic analysis
Extraction of acoustic features such as MFCCs, pitch, and energy
Classification using a supervised machine learning model
Interactive web interface for ease of use

Technology Stack
Programming Language: Python
Framework: Streamlit
Libraries and Tools:
Librosa (audio processing)
SpeechRecognition (speech-to-text conversion)
Scikit-learn (machine learning algorithms)
TextBlob (text analysis)

AI-Lie-Detection/
│
├── app.py                # Main Streamlit app
├── model.pkl            # Trained ML model
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── sample_audio/        # Test audio files

