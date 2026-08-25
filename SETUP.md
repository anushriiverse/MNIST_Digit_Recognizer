# Setup & Installation Guide

Follow these steps to clone the repository, install dependencies, and run the Active Learning web app on your local machine.

## 1. Clone the Repository
Open your terminal and pull down the project:
```bash
git clone [https://github.com/anushriiverse/Number_images_prediction.git](https://github.com/anushriiverse/Number_images_prediction.git)
cd Number_images_prediction

2. Set Up a Virtual Environment (Recommended)

Keep your system clean by creating an isolated Python environment for this project.
Windows:

Bash

python -m venv .venv
.venv\Scripts\activate


Mac/Linux:

Bash

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies

Install all the required Python libraries using the requirements file:

Bash

pip install -r requirements.txt

4. Train the Initial Base Model

Because trained model files are too large for standard GitHub hosting, you need to generate the initial model locally. Run the training script:

Bash

python train.py


Note: This will download the 70,000-image MNIST dataset, train a Random Forest classifier, and save the model to saved_models/mnist_rf_model.pkl**. It usually takes about 1-2 minutes.

5. Launch the Web App

Once the model is successfully trained, start the local Gradio web server:

Bash

python app.py


The terminal will output a local URL (e.g., http://127.0.0.1:7860**). Click it to open the interactive canvas in your browser and start drawing!

🔄 How to Use the Active Learning Loop

This application gets smarter the more you use it. To teach the model your handwriting style:

Draw a number on the canvas and click Predict Digit.
If the prediction is wrong, type the correct number into the Actual Digit (0-9) box.
Click Save Correction to Dataset. The app saves your raw drawing to the collected_corrections/ folder.

Stop the web server (Ctrl+C) and run:

Bash

python retrain.py


This script merges your custom drawings with the original MNIST dataset, trains an updated model, and logs the new evaluation metrics in the logs/ folder so you can track improvements over time.

Created the SETUP.md file with cleaned-up Markdown, proper code blocks, troubleshooting, project directories, and a quick-start section. Humanity survives another README-adjacent task.

Download SETUP.md 

SETUP.md
Document

Library
/
SETUP.md
Setup & Installation Guide

Follow these steps to clone the repository, install dependencies, train the initial model, and run the Active Learning web app locally.

1. Clone the Repository

Open your terminal and clone the project:

git clone https://github.com/anushriiverse/Number_images_prediction.git
cd Number_images_prediction
2. Set Up a Virtual Environment (Recommended)

A virtual environment keeps the project's Python dependencies isolated from the rest of your system.

Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies

Install all required Python libraries from the requirements.txt file:

pip install -r requirements.txt
4. Train the Initial Base Model

The trained model file is not included in the repository because model files can be large and are not ideal for standard GitHub hosting.

Generate the initial model locally by running:

python train.py

This script will:

Download the 70,000-image MNIST dataset.
Train a Random Forest classifier.
Save the trained model to saved_models/mnist_rf_model.pkl.

The initial training usually takes around 1–2 minutes, depending on your system.

5. Launch the Web App

Once the model has been trained successfully, start the local Gradio web server:

python app.py

The terminal will display a local URL similar to:

http://127.0.0.1:7860

Open the URL in your browser to access the interactive drawing canvas and start making predictions.

🔄 How to Use the Active Learning Loop

The application supports an active learning workflow that allows the model to learn from corrections made by users.

Step 1: Draw a Digit

Draw a handwritten digit on the canvas and click Predict Digit.

Step 2: Correct an Incorrect Prediction

If the prediction is incorrect, enter the correct digit in the Actual Digit (0-9) field.

Step 3: Save the Correction

Click Save Correction to Dataset.

The application will save your drawing and its corrected label in the:

collected_corrections/

folder.

Step 4: Retrain the Model

Stop the running web server with Ctrl+C, then run:

python retrain.py

The retraining script will:

Load the original MNIST dataset.
Merge it with your collected corrections.
Train an updated Random Forest model.
Save the updated model.
Record evaluation metrics in the logs/ directory.

You can repeat this cycle as more corrections are collected, allowing the model to gradually adapt to additional handwriting styles.

📁 Important Project Directories
saved_models/
    mnist_rf_model.pkl       # Trained model

collected_corrections/
    ...                      # User-provided corrected drawings

logs/
    ...                      # Evaluation metrics and retraining logs
Troubleshooting
python is not recognized

On some systems, Python may be available as python3 instead:

python3 --version

If that works, use python3 instead of python when running the commands above.

Virtual environment is not activated

Make sure your terminal shows the virtual environment name, typically .venv, before running the installation or application commands.

On Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate
Dependencies fail to install

Make sure you are using a supported Python version and that pip is up to date:

python -m pip install --upgrade pip
pip install -r requirements.txt
Quick Start

Once the repository has been cloned, the shortest path to a running application is:

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python train.py
python app.py

Then open the local Gradio URL shown in the terminal.