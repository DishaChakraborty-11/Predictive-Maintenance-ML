import numpy as np

def validate_inputs(values):
    """
    Validate user inputs before prediction.
    Ensures no missing or invalid values are passed.
    """
    for v in values:
        if v is None or v == "":
            return False
    return True


def preprocess_input(scaler, input_values):
    """
    Takes raw input values from Streamlit and applies scaling using the trained scaler.
    Returns a numpy array ready for prediction.
    """
    input_array = np.array(input_values).reshape(1, -1)
    scaled_data = scaler.transform(input_array)
    return scaled_data


def get_prediction_label(prediction):
    """
    Converts 0/1 model output into a readable label for the UI.
    """
    return "Failure Likely" if prediction == 1 else "Normal Operation"
