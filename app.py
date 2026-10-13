import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
import traceback
from fastapi.responses import PlainTextResponse
from tensorflow import keras

app = FastAPI()

preprocessor = joblib.load("models/preprocessor.joblib")
model = keras.models.load_model("models/model.keras")
try:
    y_scaler = joblib.load("models/y_scaler.joblib")
except FileNotFoundError:
    y_scaler = None

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    try:
        X = pd.DataFrame([payload])
        prediction = model.predict(preprocessor.transform(X))
        if y_scaler is not None:
            prediction = y_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()
        return {"predicted_number_of_products_sold": float(prediction[0])}
    except Exception:
        return PlainTextResponse(traceback.format_exc(), status_code=500)