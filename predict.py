import json
import os

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved Keras model (SavedModel format)
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "model", "tf_model"))

# Load class names; JSON keys are strings ("0", "1", ...)
with open(os.path.join(BASE_DIR, "model", "classes.json"), "r") as f:
    classes = json.load(f)


def predictions(text: str):
    """
    Predicts the disease based on the symptoms.
    :param text: The symptoms
    :return: (predicted disease, list of [other disease, score] pairs)
    """
    input_dict = {"text": tf.convert_to_tensor([text])}
    scores = model.predict(input_dict, verbose=0)[0]
    best = int(tf.argmax(scores).numpy())
    predicted_disease = classes[str(best)]
    other_diseases = [
        (classes[str(i)], float(scores[i])) for i in range(len(scores)) if i != best
    ]
    return predicted_disease, other_diseases
