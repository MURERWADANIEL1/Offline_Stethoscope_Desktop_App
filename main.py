import sys
import os
import librosa
import numpy as np
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QMessageBox, QCheckBox,QSizePolicy, QScrollArea
from utils.inference import load_model, predict_disease, label_encoder
from utils.audio_utils import create_spectrogram, preprocess_spectrogram, visualize_prediction_in_widget, save_prediction_results

class Worker(QObject):
    finished=pyqtSignal(str, float, object, object, object, int, str)

    def __init__(self,audio_path, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        #self.model=model

    def run(self):
        disease, confidence, spectrogram, prediction,y,sr=predict_disease(self.audio_path)
        self.finished.emit(disease,confidence, spectrogram,prediction,y,sr,self.audio_path)
class StethoscopeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline AI Stethoscope")
        self.setGeometry(50, 50, 1200, 600)
        main_layout = QVBoxLayout()
        self.label = QLabel("Load an audio file to analyze")
        self.button = QPushButton("Load Audio")
        self.button.clicked.connect(self.load_audio)
        self.result = QLabel("Prediction will appear here")

        self.label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.result.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)

        main_layout.addWidget(self.label)
        main_layout.addWidget(self.button)   
        main_layout.addWidget(self.result)   

        self.scroll_area=QScrollArea()  
        self.scroll_area.setWidgetResizable(True)
        self.plot_widget=QWidget()
        self.plot_area=QVBoxLayout(self.plot_widget)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setWidget(self.plot_widget)
        main_layout.addWidget(self.scroll_area) 

        self.setLayout(main_layout) 
        
        self.spinner_label=QLabel()
        self.spinner=QMovie("assets/loading.gif")
        self.spinner_label.setMovie(self.spinner)
        self.spinner_label.hide()
        main_layout.addWidget(self.spinner_label)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav)")

        y,sr=librosa.load(file_path,sr=None)
        spectrogram=create_spectrogram(file_path)

        self.spinner_label.show()
        self.spinner.start()
        self.button.setEnabled(False)
        self.result.setText("Proessing audio, please wait...")

        #background processing
        self.thread=QThread()
        self.worker=Worker(file_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_processing_done)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def add_visualization(self, title, canvas):
        group_box=QGroupBox()
        vbox=QVBoxLayout()

        #create title with X buttons
        title_bar=QHBoxLayout()
        title_label=QLabel(title)
        close_button=QPushButton("X")
        close_button.setFixedSize(20,20)

        #close logic
        def close_view():
            group_box.setParent(None)
            group_box.deleteLater()

        close_button.clicked.connect(close_view)

        title_bar.addWidget(title_label)
        title_bar.addStretch()
        title_bar.addWidget(close_button)

        #adding title bar and canvas
        vbox.addLayout(title_bar)
        vbox.addWidget(canvas)
        group_box.setLayout(vbox)
        group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plot_area.addWidget(group_box)

    def on_processing_done(self, disease,confidence, spectrogram, prediction,y,sr, file_path):
        self.spinner.stop()
        self.spinner_label.hide()
        self.button.setEnabled(True)
        self.result.setText(f"Prediction: {disease} ({confidence:.2%})")
        visualize_prediction_in_widget(self, prediction, spectrogram, disease, confidence, label_encoder, y, sr)

    
        if file_path:  # Check if a file was selected
            self.label.setText(f"Loaded: {os.path.basename(file_path)}")
            
            # Run prediction
            disease, confidence, spectrogram, prediction, y, sr = predict_disease(file_path)
            # Show result
            if disease == "Unknown":
                self.result.setText(f"Prediction: Low confidence, {disease} ({confidence:.2%})")
            elif disease == "Error":
                self.result.setText("Error processing audio")
            else:
                self.result.setText(f"Prediction: {disease} ({confidence:.2%})")
                                            
            # Optional: visualize results in a pop-up
            if spectrogram is not None and prediction is not None:
                canvas=visualize_prediction_in_widget(self, prediction, spectrogram, disease, confidence, label_encoder, y, sr)
                self.add_visualization(f"Spectrogram", canvas)

                #visualize_prediction_in_widget(self, prediction, spectrogram, disease, confidence, label_encoder, y, sr)
                reply=QMessageBox.question(self,
                                       "Save Spectogram?",
                                       "Do you want to save the spectrogram?",
                                       QMessageBox.Yes|QMessageBox.No,
                                       QMessageBox.No
                                        )
                if reply == QMessageBox.Yes:
                    # Save spectrogram and results
                    save_prediction_results(file_path, disease, confidence, spectrogram)
                               
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StethoscopeApp()
    window.show()
    sys.exit(app.exec())
