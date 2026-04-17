# Project Structure


HearMeSign/
├── as_l_dictionary.py
├── config.py
├── ctc_model.py
├── index.html
├── landmarks.py
├── manifest.json
├── matcher.py
├── mp_holistic.py
├── run.sh
├── setup.sh
├── sign_model.py
├── sw.js
├── text_processing.py
├── train_asl.py
├── train_asl_mlp.py
├── train_continuous.py
├── train_from_feedback.py
├── train_lse.py
├── train_lse_mlp.py
├── tts.py
└── web_bridge.py


---

# File Descriptions

## Core Configuration and Utilities

### config.py
Central configuration file defining parameters such as model paths, thresholds, and runtime settings.

### text_processing.py
Handles post-processing of predicted outputs, including cleaning, formatting, and sentence construction.

### as_l_dictionary.py
Contains mappings between predicted classes and corresponding sign language words or phrases.

---

## Computer Vision and Feature Extraction

### mp_holistic.py
Integrates MediaPipe Holistic for detecting hands, pose, and facial landmarks from video input.

### landmarks.py
Extracts and structures landmark data from MediaPipe outputs for downstream model input.

---

## Models and Inference

### sign_model.py
Defines the primary model architecture for sign classification.

### ctc_model.py
Implements a Connectionist Temporal Classification (CTC) model for sequence-based continuous sign recognition.

### matcher.py
Matches model predictions to known signs or sequences using similarity scoring or alignment.

---

## Training Scripts

### train_asl.py
Trains a model for American Sign Language classification.

### train_asl_mlp.py
Trains an ASL model using a multilayer perceptron (MLP).

### train_lse.py
Trains a model for Spanish Sign Language (LSE).

### train_lse_mlp.py
MLP-based training script for LSE classification.

### train_continuous.py
Trains models for continuous sign recognition using sequence-based approaches.

### train_from_feedback.py
Fine-tunes models using user feedback or corrected predictions.

---

## Speech and Output

### tts.py
Converts recognized text into speech using a text-to-speech system.

---

## Web Interface and Integration

### index.html
Frontend interface for video input and displaying recognition results.

### web_bridge.py
Connects backend Python models with the frontend, handling communication.

### manifest.json
PWA configuration file for installability and metadata.

### sw.js
Service worker enabling offline support and caching.

---

## Scripts and Automation

### setup.sh
Sets up the environment and installs dependencies.

### run.sh
Launches the application.
