# Offline AI Stethoscope

This project is a desktop application that uses a deep learning model to analyze respiratory audio recordings and predict potential respiratory conditions. It serves as a software interface for an AI-powered stethoscope, allowing for offline analysis of `.wav` audio files.

The application is built with Python, using PyQt5 for the graphical user interface and TensorFlow/Keras for the machine learning model inference.

## Features

*   **Audio File Analysis**: Load and process `.wav` audio files.
*   **AI-Powered Prediction**: Predicts one of several respiratory conditions (URTI, Healthy, COPD, Bronchiectasis, Pneumonia, Bronchiolitis) from an audio recording.
*   **Confidence Score**: Displays the model's confidence in its prediction.
*   **Data Visualization**:
    *   View the audio waveform.
    *   View the corresponding mel spectrogram used by the model.
*   **Save Results**: Export the generated spectrogram as a `.npy` file for further analysis.
*   **Responsive UI**: A loading indicator provides feedback during processing, and the interface is designed to handle analysis in the background without freezing.

## Getting Started

### Prerequisites

*   Python 3.8+
*   The required Python packages as listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/MURERWADANIEL1/Offline_Stethoscope_Desktop_App>
    cd offline_stethoscope
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To start the application, run the `main.py` script from the root directory:

```bash
python main.py
```

## Usage

1.  Launch the application.
2.  Click the **"Load Audio"** button and select a `.wav` file.
3.  The application will process the file and display the prediction and confidence score.
4.  You will be prompted to choose a visualization: **Waveform** or **Spectrogram**.
5.  The selected visualization will appear. You can save the visualization data or close it.
6.  Use the **"Switch View"** button to open another visualization of the same audio file.
