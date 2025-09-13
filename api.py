from fastapi import FastAPI
from api_models import PredictionPayload
import ml_logic
import pandas as pd

app = FastAPI(title="Power Consumption Prediction API")


@app.get("/")
def welcome():

    return "This is a welcome message"


@app.post("/predict")
def predict(payload: PredictionPayload):
    """
    Takes historical power consumption data and returns a forecast
    for the next hour.
    """
    # 1. FastAPI has already validated and parsed the payload for us.
    # We can access the data with simple dot notation.
    model_type = payload.model_options.model_type
    timeseries_detail = payload.model_options.timeseries_detail

    # Pydantic models can be easily converted to dictionaries,
    # which is perfect for creating a Pandas DataFrame.
    data_list = [point.dict() for point in payload.data]
    df = pd.DataFrame(data_list)

    # 2. Now, call the functions you already built and tested!
    model, feature_scaler, target_scaler = ml_logic.load_artifacts(model_type)
    preprocessed_tensor = ml_logic.preprocess_input(model_type, timeseries_detail, df, feature_scaler)
    forecast = ml_logic.make_prediction(model, preprocessed_tensor, target_scaler)

    # The model returns a tensor, let's get the raw value
    prediction_value = forecast

    # 3. Return a clean JSON response
    return {"model_used": model_type, "forecast": prediction_value}