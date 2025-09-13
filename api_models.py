# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal # We use Literal for fixed string options

# This defines the shape of a single data point in our list
class PowerDataPoint(BaseModel):
    Date: str
    Time: str
    Global_active_power: float
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: float
    Sub_metering_2: float
    Sub_metering_3: float

# This defines the "model_options" part of our JSON
class ModelOptions(BaseModel):
    model_type: Literal['FFN', 'LSTM']
    timeseries_detail: Literal['minutes', 'hours']

# This is the main input model for our entire JSON payload
class PredictionPayload(BaseModel):
    model_options: ModelOptions
    data: List[PowerDataPoint]