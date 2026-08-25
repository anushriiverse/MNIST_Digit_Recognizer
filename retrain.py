import os
import cv2
import joblib
import numpy as np
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt

# Import your existing preprocessing pipeline
from src.data_preprocess import format_for_model

def load_custom_data():
    custom_X = []
    custom_y = []

    # check if data directory exists for corrections
    data_dir = "collected_corrections"
    if not os.path.exists(data_dir):
        print ("No custom data directory found")
        return np.array([]), np.array([])

    count = 0
    for digit in range(10):
        folder_path = os.path.join(data_dir, str(digit))
        if not os.path.exists(folder_path):
            continue

        # get filename
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                filepath = os.path.join(folder_path, filename)
                # save in grayscale
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    # preprocess the image
                    preprocessed_array = format_for_model(img)[0]
                    custom_X.append(preprocessed_array)
                    custom_y.append(str(digit))
                    count += 1
    print(f"Loaded {count} custom images for retraining.")
    return np.array(custom_X), np.array(custom_y)

def main():
    print("Load MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = np.array(mnist.data), np.array(mnist.target)

    # Normal Pixel values
    X = X / 255.0

    # load custom data
    X_custom, y_custom = load_custom_data()

    if len(X_custom) > 0:
        # combine MNIST and custom data
        X_combined = np.vstack((X, X_custom))
        y_combined = np.concatenate((y, y_custom))
    else:
        X_combined, y_combined = X, y

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined , test_size=0.2, random_state=42)

    # train
    print("Train Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, n_jobs = -1, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    print("Evaluate the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # print the evaluation metrics
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")

    report = classification_report(y_test, predictions)

    # save model
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mnist_rf_model.pkl")

    print(f"Saving the model to {save_path}...")
    joblib.dump(model, save_path)

    # Create a logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "training_metrics.txt")
    
    # Open the file in "a" (append) mode so we don't delete old logs
    with open(log_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d || %H:%M:%S")
        log_file.write(f"=== Training Run: {timestamp} ===\n")
        log_file.write(f"Custom Images Added: {len(X_custom)}\n")
        log_file.write(f"Validation Accuracy: {accuracy * 100:.4f}%\n")
        log_file.write("Classification Report:\n")
        log_file.write(report)
        log_file.write("\n" + "="*55 + "\n\n")
        
    print(f"Metrics report saved to: {log_path}")

    # cm = confusion_matrix(y_test, predictions)
    # plt.figure(figsize=(10, 7))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

if __name__ == "__main__":
    main()
