import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    print("Load MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target

    # Normal Pixel values
    X = X / 255.0

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

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
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    # save model
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mnist_rf_model.pkl")

    print(f"Saving the model to {save_path}...")
    joblib.dump(model, save_path)

if __name__ == "__main__":
    main()