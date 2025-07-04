import os
import librosa
import numpy as np
import tensorflow as tf
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import librosa.display
from datetime import datetime

def create_spectrogram(audio_path, target_shape=(128, 128)):
    """Create mel spectrogram matching training process"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        s = librosa.feature.melspectrogram(y=y, sr=sr)
        s_db = librosa.amplitude_to_db(s, ref=np.max)
        s_resized = cv2.resize(s_db, target_shape)
        return np.expand_dims(s_resized, axis=-1)
    except Exception as e:
        print(f"Error creating spectrogram: {str(e)}")
        return None

def preprocess_spectrogram(spectrogram):
    """Normalize and scale spectrogram if needed"""
    scaler = MinMaxScaler()
    original_shape = spectrogram.shape
    spectrogram_flat = spectrogram.reshape(-1, 1)
    spectrogram_scaled = scaler.fit_transform(spectrogram_flat)
    spectrogram_normalized = spectrogram_scaled.reshape(original_shape)
    return spectrogram_normalized

def visualize_prediction_in_widget(widget, prediction, spectrogram, disease, confidence, label_encoder, y, sr):    
    """Visualize both spectrogram and class probabilities"""
    classes = label_encoder.classes_
    figure=Figure(figsize=(12, 4))
    canvas=FigureCanvas(figure)
    widget.layout().addWidget(canvas)
    
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
    ax2.imshow(spectrogram[:,:,0], aspect='auto', origin='lower',cmap='viridis')
    ax2.set_title(f"Predicted: {disease} ({confidence:.2%})")

    # Class probabilities plot       
    classes = label_encoder.classes_
    probabilities = prediction[0]
    ax3.barh(classes,probabilities, color='blue')
    ax3.set_title("Class Probabilitites")
    ax3.set_xlim([0, 1])

    figure.tight_layout()
    canvas.draw()
    """"
    colors = ['green' if prob == max(probabilities) else 'blue' for prob in probabilities]
    bars = plt.barh(classes, probabilities, color=colors)
    plt.xlim([0, 1])
    plt.title('Class Probabilities')
    plt.xlabel('Probability')  
    
    # Annotate probabilities
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2%}', ha='left', va='center')    
    plt.tight_layout()
    plt.show()"""

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


