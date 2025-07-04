import sys
import os
import librosa
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QMessageBox, QCheckBox
from utils.inference import load_model, predict_disease, label_encoder
from utils.audio_utils import create_spectrogram, preprocess_spectrogram, visualize_prediction_in_widget, save_prediction_results
class StethoscopeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline AI Stethoscope")
        self.setGeometry(50, 50, 1200, 600)
        layout = QVBoxLayout()
        self.label = QLabel("Load an audio file to analyze")
        self.button = QPushButton("Load Audio")
        self.result = QLabel("Prediction will appear here")
        self.button.clicked.connect(self.load_audio)
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.result)
        self.setLayout(layout)        
        self.stack_checkbox=QCheckBox("Stack Views")
        self.stack_checkbox.setChecked(False)
        layout.addWidget(self.stack_checkbox)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav)")

        y,sr=librosa.load(file_path,sr=None)
        spectrogram=create_spectrogram(file_path)


        if file_path:  # Check if a file was selected
            self.label.setText(f"Loaded: {os.path.basename(file_path)}")
            #mel_input=load_and_preprocess_audio(file_path)
            #prediction=classify_audio(mel_input)
            #self.result.settext(f"Prediction: {prediction}")  
            # Run prediction
            disease, confidence, spectrogram, prediction = predict_disease(file_path)
            # Show result
            if disease == "Unknown":
                self.result.setText(f"Prediction: Low confidence, {disease} ({confidence:.2%})")
            elif disease == "Error":
                self.result.setText("Error processing audio")
            else:
                self.result.setText(f"Prediction: {disease} ({confidence:.2%})")
                                            
            # Optional: visualize results in a pop-up
            if spectrogram is not None and prediction is not None:
                visualize_prediction_in_widget(self, prediction, spectrogram, disease, confidence, label_encoder, y, sr)
                reply=QMessageBox.question(self,
                                       "Save Spectogram?",
                                       "Do you want to save the spectrogram?",
                                       QMessageBox.Yes|QMessageBox.No,
                                       QMessageBox.No
                                        )
                if reply == QMessageBox.Yes:
                    # Save spectrogram and results
                    save_prediction_results(file_path, disease, confidence, spectrogram)
                
                if not self.stack_checkbox.isChecked():
                    while self.layout().count()>3:
                        child=self.layout().takeAt(3)

                        if child.widget():
                            child.widget().deleteLater()
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StethoscopeApp()
    window.show()
    sys.exit(app.exec())
