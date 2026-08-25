# 🎨 MNIST Digit Recognizer with Active Learning

An interactive machine learning web application that allows users to draw a digit (0-9) on a canvas and uses a Random Forest model to predict the drawn number. 

Unlike standard MNIST classifiers that boast 99% accuracy in a notebook but fail on real-world web canvas inputs due to dataset distribution shifts, this project is built for the real world. It features a custom computer vision preprocessing pipeline and a built-in **Active Learning Feedback Loop** that allows the model to continuously learn and adapt to your specific handwriting style.

## 🌟 Key Features
* **Active Learning Pipeline:** If the model guesses incorrectly, you can flag the mistake in the UI, save the raw drawing, and seamlessly retrain the model with your new data.
* **Data-Shift Preprocessing:** Standard UI canvases produce hard, digital pixels on a white background. This app uses a custom OpenCV pipeline to dynamically invert, crop, center-of-mass align, and blur user drawings to flawlessly mimic the physics of original 1990s scanned MNIST ink.
* **Lightweight Architecture:** Built using traditional machine learning (Scikit-Learn Random Forest) instead of heavy neural networks, making training and inference lightning fast on any CPU.

**Tech Stack:** `Python`, `Scikit-Learn`, `OpenCV`, `Gradio`, `NumPy`

---

## 📁 Project Structure

```text
Number_images_prediction/
│
├── logs/                       # Auto-generated: Stores training_metrics.txt evaluation logs
├── saved_models/               # Directory for the compiled Random Forest model (.pkl)
│
├── src/                        # Core pipeline modules
│   ├── __init__.py             
│   ├── data_preprocess.py           # Handles OpenCV image inversion, centering, and flattening
│   └── inference.py            # Loads the .pkl model and outputs predictions
│
├── train.py                    # Downloads original MNIST, trains, and saves the initial base model
├── retrain.py                  # Merges MNIST with your custom UI corrections and retrains the model
├── main.py                      # Main Gradio web application with the feedback UI
├── requirements.txt            # List of Python package dependencies
├── .gitignore                  # Keeps large model files and environments out of Git
└── README.md

