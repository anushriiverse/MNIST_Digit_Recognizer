import joblib

def load_model(model_path):
    """"
    Loads model from .pkl file
    of skilearn
    """
    model = joblib.load(model_path)
    return model

def predict_digit(preprocessed_img, model):

    prediction = model.predict(preprocessed_img)
    return str(prediction)