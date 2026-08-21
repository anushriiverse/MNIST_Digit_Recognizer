# MNIST Digit Recognizer

A lightweight machine learning web application that allows users to draw a digit (0-9) on a canvas, and uses a trained Random Forest model to predict the drawn number. 

This project bridges the gap between a raw machine learning model and a user-friendly interface, featuring a custom preprocessing pipeline to perfectly center and scale hand-drawn input to match the MNIST dataset format.

It uses active learning loop where if the model makes incorrect predictions, the correct label entered by the user along with the drawn number image will be used for re-training to fine-tune the model.

The project is hosted locally. 

## Project Structure

```text
MNIST/
│
├── models/               # Directory for the compiled model
│   └── mnist_rf_model.pkl      # Generated after running train.py
│
├── src/                        # Core pipeline modules
│   ├── __init__.py             
│   ├── data_preprocess.py           # Handles image inversion, centering, and flattening
│   └── inference.py            # Loads the .pkl model and outputs predictions
│
├── train.py                    # Downloads MNIST, trains the Random Forest, and saves it
├── main.py                      # Main Gradio web application
└── README.md                   # Project documentation