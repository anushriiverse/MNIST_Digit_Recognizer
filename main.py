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
        return f"Predicted Digit: {prediction}"
    except Exception as e:
        return f"Error during prediction: {e}"

#gradio's sketchpad
demo = gr.Interface(
    fn = recognize_digit,
    inputs = gr.Sketchpad(type='numpy', width=280, height=280, interactive=True), 
    outputs = gr.Textbox(label="Prediction Result", text_align="center"),
    title = "Handwritten Digit Recognition",
    description = "Draw a digit (0-9) in the sketchpad and click 'Submit' to see the model's prediction.",
)

if __name__ == "__main__":
    demo.launch()