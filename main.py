import time
import os
import cv2
import numpy as np
import gradio as gr
from src.data_preprocess import format_for_model
from src.inference import load_model, predict_digit

MODEL_PATH = "models/mnist_rf_model.pkl"

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

FEEDBACK_DIR = "collected_corrections"

# Define the Gradio interface
def recognize_digit(image):
    # this function takes raw drawing
    if image is None:
        return "No image provided, please draw again."

    if isinstance(image, dict):
        image_array = image.get("composite", image.get("image"))
    else:
        image_array = image

    # Preprocess the image
    try:
        preprocessed_array = format_for_model(image_array)
        prediction = predict_digit(preprocessed_array, model)
        return f"Predicted Digit: {prediction}", image_array
    except Exception as e:
        return f"Error during prediction: {e}", None

def save_correction(image_array, true_label):
    if image_array is None:
        return "No image to save."

    if not true_label or not true_label.isdigit() or not (0 <= int(true_label) <= 9):
        return "Invalid label. Please enter a digit between 0 and 9."

    label_dir = os.path.join(FEEDBACK_DIR, str(true_label))
    os.makedirs(label_dir, exist_ok=True)

    filename = f"{int(time.time() * 1000)}.png"
    filepath = os.path.join(label_dir, filename)

    if image_array.dtype != np.uint8:
            image_array = np.uint8(image_array)

    # array into standard memory
    img_clean = np.ascontiguousarray(image_array, dtype=np.uint8)

    # retuns RGB or RGBA
    if len(img_clean.shape) == 3:
        if img_clean.shape[2] == 4:
            img_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGBA2GRAY)
        elif img_clean.shape[2] == 3:
            img_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)

    success = cv2.imwrite(filepath, img_clean)
    if success:
        return f"Correction saved successfully at {filepath}."
    else:
        return "Failed to save the correction. Please try again."


#gradio's Blocks for UI
with gr.Blocks() as demo:
    gr.Markdown("MNIST Digit Classifier with active learning loop")
    gr.Markdown("Draw a digit (0-9) in the box below and click 'Submit' to see the prediction.")

    current_image = gr.State()

    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(type="numpy", height=280, width=280 , label = "Drawing Canvas")
            predict_button = gr.Button("Predict Digit", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(label="Model Prediction", text_align="center")
            gr.Markdown("If the prediction is incorrect, please provide the correct label (0-9) below and click 'Submit Correction'.")
            correct_label_ip = gr.Textbox(label="Correct Label (0-9)", placeholder="Enter the correct digit here")
            save_button = gr.Button("Submit Correction")
            status_box = gr.Textbox(label="Status", interactive=False)

    predict_button.click(fn=recognize_digit, inputs=canvas, outputs=[output_box, current_image])
    save_button.click(fn=save_correction, inputs=[current_image, correct_label_ip], outputs=status_box)


if __name__ == "__main__":
    demo.launch(share=True)