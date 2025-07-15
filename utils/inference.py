import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from utils.audio_utils import create_spectrogram, preprocess_spectrogram

disease_classes = ["URTI", "Healthy", "COPD", "Bronchiectasis", "Pneumonia", "Bronchiolitis"]
label_encoder = LabelEncoder()
label_encoder.fit(disease_classes)

CONFIDENCE_THRESHOLD = 0.7

model = None
def load_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model("model/respiratory_cnn_model.h5")
            print("Model loaded successfully")
        except Exception as e:
            print(f" Error loading model: {e}")
            model = None
    return model

def predict_disease(audio_path):
    """Predict disease with confidence thresholding"""
    y,sr=librosa.load(audio_path, sr=None)
    spectrogram=create_spectrogram(y,sr)

    model=load_model()
    if model is None:
        return "Error", 0.0, None, None
    
    spectrogram = create_spectrogram(audio_path)
    if spectrogram is None:
        return None, None, None, None, y, sr
    
    spectrogram_processed = preprocess_spectrogram(spectrogram)
    spectrogram_input = np.expand_dims(spectrogram_processed, axis=0)
    
    prediction = model.predict(spectrogram_input, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    
    # Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence, spectrogram, prediction, y, sr
    
    predicted_disease = label_encoder.inverse_transform(predicted_class)[0]
    return predicted_disease, confidence, spectrogram, prediction,y,sr