import os
import librosa
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import librosa.display
from datetime import datetime

def create_spectrogram(audio_input, sr=None, target_shape=(128, 128)):
    """Create mel spectrogram from an audio file path or raw audio data."""
    try:
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=None)
        elif isinstance(audio_input, np.ndarray):
            y = audio_input
            if sr is None:
                raise ValueError("Sample rate 'sr' must be provided for raw audio data.")
        else:
            raise TypeError(f"audio_input must be a file path or numpy array, not {type(audio_input)}")

        s = librosa.feature.melspectrogram(y=y, sr=sr)
        s_db = librosa.amplitude_to_db(s, ref=np.max)
        s_resized = cv2.resize(s_db, target_shape)
        return np.expand_dims(s_resized, axis=-1)
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def preprocess_spectrogram(spectrogram):
    """Normalize and scale spectrogram if needed"""
    scaler = MinMaxScaler()
    original_shape = spectrogram.shape
    spectrogram_flat = spectrogram.reshape(-1, 1)
    spectrogram_scaled = scaler.fit_transform(spectrogram_flat)
    spectrogram_normalized = spectrogram_scaled.reshape(original_shape)
    return spectrogram_normalized
def create_waveform_canvas(y, sr, disease, confidence):
    """Return a FigureCanvas with waveform and prediction info."""
    figure = Figure(figsize=(8, 3))
    canvas = FigureCanvas(figure)
    ax = figure.add_subplot(111)
    time = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(time, y)
    ax.set_title(f"Waveform - Prediction: {disease} ({confidence:.2f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    figure.tight_layout()
    canvas.draw()
    return canvas
def create_spectrogram_canvas(spectrogram, disease, confidence):
    figure=Figure(figsize=(8, 3))
    canvas=FigureCanvas(figure)
    ax=figure.add_subplot(111)

    if spectrogram is not None:
        img=ax.imshow(spectrogram[:,:,0], aspect='auto', origin='lower', cmap='viridis')
        figure.colorbar(img, ax=ax,format="%+2.0f dB")
        ax.set_title(f"Spectrogram - Prediction: {disease} ({confidence:.2f}%)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
    else:
        ax.set_title("Spectrogram not available")
        ax.text(0.5, 0.5, "No Spectrogram Data", horizontalalignment='center', verticalalignment='center')

                
    figure.tight_layout()
    canvas.draw()
    return canvas




def visualize_prediction_in_widget(widget, prediction, spectrogram, disease, confidence, label_encoder, y, sr):    
    """Visualize both spectrogram and class probabilities"""
    #fig, ax=plt.subplots(figsize=(6, 4))
    
    classes = label_encoder.classes_
    figure=Figure(figsize=(12, 4))
    canvas=FigureCanvas(figure)
    #widget.plot_area.addWidget(canvas)
    
    ax1=figure.add_subplot(1, 3, 1)
    ax2=figure.add_subplot(1, 3, 2)
    ax3=figure.add_subplot(1, 3, 3)
    
    #plot time varying waveform
    time=np.linspace(0, len(y)/sr, num=len(y))
    ax1.plot(time,y)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
        
    # Spectrogram plot
    if spectrogram is not None:
        ax2.imshow(spectrogram[:,:,0], aspect='auto', origin='lower',cmap='viridis')
        ax2.set_title(f"Predicted: {disease} ({confidence:.2%})")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Mel Frequency")
    else:
        ax2.set_title("Spectrogram not available")

    # Class probabilities plot       
    classes = label_encoder.classes_
    probabilities = prediction[0] if prediction is not None else [0]*len(classes)
    ax3.barh(classes,probabilities, color='blue')
    ax3.set_title("Class Probabilitites")
    ax3.set_xlim([0, 1])
    ax3.set_xlabel("Probability")

    figure.tight_layout()
    canvas.draw()
    #canvas=FigureCanvas(fig)
    return canvas
    
def save_prediction_results(audio_path, disease, confidence, spectrogram):
    """Save results with formatted filenames"""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = "Predicted_Spectrograms"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save spectrogram
    spectrogram_filename = f"{base_name}_{disease}.npy"
    spectrogram_path = os.path.join(output_dir, spectrogram_filename)
    np.save(spectrogram_path, spectrogram)
    print(f"\nSaved spectrogram as: {spectrogram_filename}")
