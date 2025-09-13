# api.py
import pandas as pd
import json
import numpy as np
import torch
import joblib # A common library for saving/loading Python objects like scalers
from pytorchmodels import customFFN, customLSTM, reshape_for_LSTM

MODEL_PATH_FFN = r"artifacts\ffnnet.pt"
MODEL_PATH_LSTM = r"artifacts\lstmnet.pt"
FEATURE_SCALER_PATH_FFN = r"artifacts\feature_scaler_ffn.pkl"
FEATURE_SCALER_PATH_LSTM = r"artifacts\feature_scaler_lstm.pkl"
TARGET_SCALER_PATH_FFN = r"artifacts\target_scaler_ffn.pkl"
TARGET_SCALER_PATH_LSTM = r"artifacts\target_scaler_lstm.pkl"
INPUT_DATA_PATH = "test.json"

def parse_json_from_payload(json_path):
    with open(json_path, "r") as jsonfile:
        thedict = json.load(jsonfile)

    model_options = thedict["model_options"]
    model_type = model_options["model_type"]
    timeseries_detail = model_options["timeseries_detail"]
    data = thedict["data"]
    df = pd.DataFrame(data)
    return model_type, timeseries_detail, df

def load_artifacts(model_type):
    """
    Loads the trained model and the scaler from disk.
    """
    print("Loading model")
    if model_type == "FFN":
        model = customFFN(11, 4, 128, 1)
        model.load_state_dict(torch.load(MODEL_PATH_FFN, map_location=torch.device('cpu')))
        model.eval()
        feature_scaler_path = FEATURE_SCALER_PATH_FFN
        target_scaler_path = TARGET_SCALER_PATH_FFN
    elif model_type == "LSTM":
        model = customLSTM(16, 128, 2, 1)
        model.load_state_dict(torch.load(MODEL_PATH_LSTM, map_location=torch.device('cpu')))
        model.eval()
        feature_scaler_path = FEATURE_SCALER_PATH_LSTM
        target_scaler_path = TARGET_SCALER_PATH_LSTM
    print("Loading scaler")
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print("Artifacts loaded.")
    return model, feature_scaler, target_scaler


# In api.py

def preprocess_input(model_type, timeseries_detail, df, scaler):
    """
    Validates, standardizes, and preprocesses raw data for prediction.
    """

    # Ensure the timestamp column is a datetime object and set it as the indexdf["Timestamp"] = df["Date"]+ " " + df["Time"]
    df["Timestamp"] = df["Date"] + " " + df["Time"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
    timestampcol = df.pop("Timestamp")
    df.insert(0, "Timestamp", timestampcol)

    # Convert the columns to numeric explicitly
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['Global_reactive_power'] = pd.to_numeric(df['Global_reactive_power'], errors='coerce')
    df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
    df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
    df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')
    # Conver Sub-meterings to KWH (no particular reason other that I am more familiar with this UoM)
    df['Sub_metering_1'] = df['Sub_metering_1'] / 1000
    df['Sub_metering_2'] = df['Sub_metering_2'] / 1000
    df['Sub_metering_3'] = df['Sub_metering_3'] / 1000
    df.dropna(inplace=True)
    # Calculate energy consumption in Kwh and RAE(Remaining Active energy)
    df['Energy Consumption Kwh'] = df['Global_active_power'] / 60
    df['RAE'] = df['Energy Consumption Kwh'] - df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].sum(axis=1)

    # Set the index to the timeseries columns
    df.set_index("Timestamp", inplace=True)
    # --- 1. Standardize to a clean, hourly DataFrame ---
    df_hourly = None
    if timeseries_detail == 'minutes':
        # Rule: Only include hours that have a full 60 minutes of data.
        # Hint: You can use df.resample('h').size() to check this.
        # After filtering, you would aggregate the valid hours (e.g., using .sum() or .mean())
        complete_hours_indices = (df['Global_active_power'].resample("h").size() == 60).values

        # create all hourly columns
        active_power_column = df['Global_active_power'].resample("h").mean()
        energy_consumption_column = df['Energy Consumption Kwh'].resample("h").sum()
        sub_metering_1_column = df['Sub_metering_1'].resample("h").sum()
        sub_metering_2_column = df['Sub_metering_2'].resample("h").sum()
        sub_metering_3_column = df['Sub_metering_3'].resample("h").sum()
        rae_column = df['RAE'].resample("h").sum()
        voltage_column = df['Voltage'].resample("h").mean()
        global_intensity_column = df['Global_intensity'].resample("h").mean()
        # concatenate in dataframe
        df_hourly = pd.concat(
            [active_power_column, energy_consumption_column, sub_metering_1_column, sub_metering_2_column,
             sub_metering_3_column, rae_column, voltage_column, global_intensity_column], axis=1)
        df_hourly = df_hourly[complete_hours_indices]

        print("Resampling minute-level data to complete hours.")

    elif timeseries_detail == 'hours':
        # The data is already hourly, we can use it directly.
        df_hourly = df
        print("Using provided hourly data.")
    else:
        raise ValueError("Invalid timeseries_detail specified.")

    # add next hour for ffn only
    if model_type == "FFN":
        #last valid idx
        idx = df_hourly.tail(1).index[0] + pd.Timedelta(hours=1)
        print(f"previous last idx {df_hourly.tail(1).index}, new last index {idx}")
        df_hourly.loc[idx] = None


    # feature engineering (independent of model type)
    df_hourly["Day"] = df_hourly.index.day_name()
    df_hourly["hour"] = df_hourly.index.hour
    df_hourly["month"] = df_hourly.index.month
    df_hourly["year"] = df_hourly.index.year
    df_hourly["day_of_year"] = df_hourly.index.dayofyear
    df_hourly["day_of_week"] = df_hourly.index.dayofweek
    df_hourly["week_of_year"] = df_hourly.index.isocalendar().week
    # months
    df_hourly['month_sin'] = np.sin(2 * np.pi * (df_hourly['month'] - 1) / 12.0)
    df_hourly['month_cos'] = np.cos(2 * np.pi * (df_hourly['month'] - 1) / 12.0)
    # weekdays
    df_hourly['day_of_week_sin'] = np.sin(2 * np.pi * (df_hourly['day_of_week'] - 1) / 7.0)
    df_hourly['day_of_week_cos'] = np.cos(2 * np.pi * (df_hourly['day_of_week'] - 1) / 7.0)
    # hours
    df_hourly['hour_sin'] = np.sin(2 * np.pi * (df_hourly['hour'] - 1) / 24.0)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * (df_hourly['hour'] - 1) / 24.0)
    # days of year
    df_hourly['day_of_year_sin'] = np.sin(2 * np.pi * (df_hourly['day_of_year'] - 1) / 364.25)
    df_hourly['day_of_year_cos'] = np.cos(2 * np.pi * (df_hourly['day_of_year'] - 1) / 364.25)


    # --- 2. Preprocess depending on model type ---
    if model_type == "FFN":
        required_hours = 3
        if len(df_hourly) < required_hours:
            raise ValueError(f"Insufficient data. Model '{model_type}' requires {required_hours} complete, consecutive hours.")
        # lag features (only relevant for FFN)
        df_lag1 = df_hourly[['Global_active_power']].shift(1)
        df_lag1.columns = ["Global_active_power_lag1"]
        df_lag2 = df_hourly[['Global_active_power']].shift(2)
        df_lag2.columns = ["Global_active_power_lag2"]
        df_lag3 = df_hourly[['Global_active_power']].shift(3)
        df_lag3.columns = ["Global_active_power_lag3"]
        columns_to_keep_nolag = ['Global_active_power', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                                 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
        df_hourly = pd.concat([df_hourly[columns_to_keep_nolag], df_lag1, df_lag2, df_lag3], axis=1)
        df_hourly["Global_active_power"] = df_hourly.pop('Global_active_power')
        df_hourly.dropna(inplace=True)
        X_np = np.array(df_hourly.drop(columns=['Global_active_power']))
        X_np_scaled = scaler.transform(X_np)
        X_tensor = torch.tensor(X_np_scaled, dtype=torch.float32)
        X_tensor= X_tensor[-1]

    elif model_type == "LSTM":
        required_hours = 24
        if len(df_hourly) < required_hours:
            raise ValueError(f"Insufficient data. Model '{model_type}' requires {required_hours} complete, consecutive hours.")



        columns_to_keep = ['Global_active_power', 'Energy Consumption Kwh', 'Sub_metering_1', 'Sub_metering_2',
                           'Sub_metering_3', "Voltage", "Global_intensity", 'RAE', 'hour_sin', 'hour_cos',
                           'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_of_year_sin',
                           'day_of_year_cos']

        df_hourly = df_hourly[columns_to_keep]
        df_hourly['Global_active_power'] = df_hourly.pop('Global_active_power')
        df_hourly.dropna(inplace=True)

        X_np = np.array(df_hourly)
        print(f"X_np shape = {X_np.shape}")
        X_reshaped = reshape_for_LSTM(X_np, 24)
        print(f"X reshaped shape = {X_reshaped.shape}")
        nsamples, nx, ny = X_reshaped.shape
        unreshaped_for_scaling = X_reshaped.reshape((nsamples * nx, ny))
        X_scaled = scaler.transform(unreshaped_for_scaling)
        X_final = X_scaled.reshape(X_reshaped.shape)
        X_tensor = torch.tensor(X_final, dtype=torch.float32)
        X_tensor = X_tensor[-1].view(1,24,-1)

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Valid model types: 'FFN', 'LSTM")


    # --- 3. Select the final data slice ---
    return X_tensor
    # --- 4. Now, apply feature engineering and scaling ---
    # On the 'final_data' DataFrame...
    # ... create lag features for FFN, scale the data, create tensors ...
    # preprocessed_tensor = ...

    # return preprocessed_tensor

def make_prediction(model, tensor, scaler):
    """
    Takes the model and a preprocessed tensor, returns the prediction.
    """
    prediction = model(tensor).detach().numpy()
    prediction = prediction.reshape((-1,1))
    # print(prediction, type(prediction), prediction.shape)
    final_prediction = scaler.inverse_transform(prediction).reshape(-1).tolist()
    #print(final_prediction, type(final_prediction), final_prediction.shape)
    return final_prediction

# This is a standard Python convention to make a script runnable
if __name__ == "__main__":
    # 1. Define paths to your saved artifacts
    model_type, timeseries_detail, data = parse_json_from_payload(INPUT_DATA_PATH)
    print(f"Model type: {model_type}, timeseries_detail:{timeseries_detail} data:{data.head()}")
    # 2. Execute the pipeline
    model, scaler = load_artifacts(model_type)

    preprocessed_tensor = preprocess_input(model_type, timeseries_detail, data, scaler)
    print(f"Preprocessed tensor shape: {preprocessed_tensor.shape}")

    # 3. Print the result

    forecast = make_prediction(model, preprocessed_tensor)
    print(f"forecast: {forecast}")
