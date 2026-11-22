# symptoms-disease-predictor-api

Small Flask API that takes a free-text list of symptoms and returns a predicted disease from a pre-trained TensorFlow text classifier.

## Features

- `GET /` health message.
- `GET /predict/<symptoms>` returns the most likely disease and the scores of all other classes.
- 24 disease classes (listed in `model/classes.json`), for example Acne, Allergy, Dengue, Diabetes.
- Model is loaded once at startup.

## Tech stack

Python, Flask, TensorFlow / Keras (SavedModel with a text vectorizer vocabulary).

## Project structure

```
app.py               Flask routes
predict.py           loads the model and runs inference
model/tf_model/      saved Keras model (about 70 MB)
model/classes.json   class index to disease name
requirements.txt
```

## Setup and run

Requirements: Python 3.8+ and a TensorFlow version compatible with the saved model. No environment variables.

```bash
pip install -r requirements.txt
python app.py        # serves on http://localhost:8000
```

## Usage

```bash
curl "http://localhost:8000/predict/itching%20skin%20rash"
```

Response shape:

```json
{"predicted_disease": "...", "other_diseases": [["Acne", 0.01], ["Allergy", 0.02]]}
```

## Limitations

- Not medical advice. This is a demo model, trained on data whose source is not documented in this repo, and it has no reported accuracy.
- The model file was saved with an older TensorFlow/Keras, so a much newer TensorFlow may fail to load it. This was not run during cleanup (TensorFlow was not installed).
- The Flask development server is used, no authentication, no input validation.
