import sys
import os
import librosa
import numpy as np
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QFileDialog, QVBoxLayout, QMessageBox, QCheckBox,QSizePolicy, QScrollArea
from utils.inference import load_model, predict_disease, label_encoder
from utils.audio_utils import create_spectrogram, create_spectrogram_canvas, create_waveform_canvas, preprocess_spectrogram, visualize_prediction_in_widget, save_prediction_results

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
        self.setStyleSheet("background-color: #eafaf1;")
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

        self.switch_view_button = QPushButton("Switch View")
        self.switch_view_button.setEnabled(False)
        self.switch_view_button.clicked.connect(self.switch_view)
        main_layout.addWidget(self.switch_view_button)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav)")
        if not file_path:
            return #Use cancelled, nothing to do
    

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

    def add_visualization(self, title, canvas, file_path=None, extra_data=None):
        group_box=QGroupBox()
        vbox=QVBoxLayout()

        #create title with X buttons
        title_bar=QHBoxLayout()
        #title_label=QLabel(title)
        display_text=title
        if file_path:
            display_text = f"{title} - {os.path.basename(file_path)}"
        title_label=QLabel(display_text)

        save_button=QPushButton("Save")
        save_button.setFixedSize(60, 20)
        save_button.setToolTip("Save this Visualization")

        close_button=QPushButton("X")
        close_button.setFixedSize(20,20)
        close_button.setToolTip("Close this Visualization")

        #close logic
        def close_view():
            group_box.setParent(None)
            group_box.deleteLater()

        def save_view():
            if extra_data:
                disease=extra_data.get("disease", "Unknown")
                confidence=extra_data.get("confidence", 0.0)
                original_name=os.path.splitext(os.path.basename(file_path))[0]
                filename=f"{original_name}_{disease}_{confidence:.2f}.npy"
            
            else:
                filename=f"{title.lower().replace(' ', '_')}.npy"
            save_path, _ =QFileDialog.getSaveFileName(
                self,
                f"Save{title}",
                filename, # Default filename
                "Numpy Files (*.npy);;All Files (*)" # Filter
            ) 
            if save_path:
                spectrogram=extra_data.get("spectrogram") if extra_data else None
                if spectrogram is not None:
                    np.save(save_path, spectrogram)
                    QMessageBox.information(self, "Saved", f"Spectrogram saved as:\n{save_path}")
            else:
                QMessageBox.warning(self, "Error", "Spectrogram not Saved")

        close_button.clicked.connect(close_view)
        save_button.clicked.connect(save_view)

        title_bar.addWidget(title_label)
        title_bar.addStretch()
        title_bar.addWidget(save_button)
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

        self.last_processed={
            "disease": disease,
            "confidence": confidence,
            "spectrogram": spectrogram,
            "prediction": prediction,
            "y": y,
            "sr": sr,
            "file_path": file_path
        }   

        if disease is None:
            self.result.setText("Could not process the audio file.")
            return
        self.result.setText(f"Prediction: {disease} ({confidence:.2%})")

        if file_path:  # Check if a file was selected
            self.label.setText(f"Loaded: {os.path.basename(file_path)}")
            view, ok =QInputDialog.getItem(
                self,
                "Select View",
                "Choose which view to display:",
                ["Waveform", "Spectrogram"],
                1, 
                False
            )
            if not ok:
                return #user cancelled
            
            if view == "Waveform":
                canvas=create_waveform_canvas(y, sr, disease, confidence*100)
                
            else:
                canvas=create_spectrogram_canvas(spectrogram, disease, confidence*100)
                      
            # Optional: visualize results in a pop-up
            if canvas:
                self.add_visualization(
                    view,
                    canvas,
                    file_path=file_path,
                    extra_data={
                        "disease": disease, # for filename
                        "confidence": confidence, # for filename
                        "spectrogram": spectrogram                        
                    }
                )  

            self.switch_view_button.setEnabled(True)    
    def switch_view(self):
        if not hasattr(self, "last_processed"):
            return
        data=self.last_processed
        view, ok=QInputDialog.getItem(
            self,
            "Switch View",
            "Choose which view to display:",
            ["Waveform", "Spectrogram"],
            1,
            False                
        )
        if not ok:
            return
                
        if view == "Waveform":
            canvas=create_waveform_canvas(data["y"], data["sr"], data["disease"], data["confidence"]*100)

        else:
            canvas= create_spectrogram_canvas(data["spectrogram"], data["disease"], data["confidence"]*100)

        self.add_visualization(
            view,
            canvas,
            file_path=data["file_path"],
            extra_data={
                "disease": data["disease"],
                "confidence": data["confidence"],
                "spectrogram":data["spectrogram"]
            }
        )
                    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StethoscopeApp()
    window.show()
    sys.exit(app.exec())
