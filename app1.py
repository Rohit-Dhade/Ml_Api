from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import joblib
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # You can specify frontend domain here instead of ""
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
with open("model_ac_compressed.joblib", "rb") as f:
    model_ac = joblib.load(f)

with open("model_dc_compressed.joblib", "rb") as f:
    model_dc = joblib.load(f)

# Global cache for fetched weather data
hourly_dataframe = None
system_kWp = 1000  # You can change or set dynamically based on user input

@app.get("/")
async def home():
    return {"message": "Welcome to the page !!!"}

@app.get("/fetch")
async def fetch_weather():
    global hourly_dataframe

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 18.52,
        "longitude": 73.85,
        "timezone": "auto",
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        "forecast_days": 7,
        "wind_speed_unit": "ms",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    radiation = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": temperature_2m,
        "wind_speed_10m": wind_speed_10m,
        "shortwave_radiation": radiation,
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return {"message": "Weather data fetched and stored successfully."}


@app.get("/predict")
async def predict_power():
    global hourly_dataframe, system_kWp

    if hourly_dataframe is None:
        return JSONResponse(status_code=400, content={"error": "Weather data not fetched yet."})

    # Prepare features
    features_ac = hourly_dataframe.iloc[:, 1:4].values
    ac_hourly = model_ac.predict(features_ac)

    # Prepare features for DC model
    feature_dc = np.concatenate((features_ac, ac_hourly.reshape(-1, 1)), axis=1)
    dc_hourly = model_dc.predict(feature_dc)

    # Set AC/DC output to 0 when radiation is 0
    for i in range(len(ac_hourly)):
        if features_ac[i][2] == 0:
            ac_hourly[i] = 0
    for i in range(len(dc_hourly)):
        if features_ac[i][2] == 0:
            dc_hourly[i] = 0

    # Calculate totals
    ac_total = round((np.sum(ac_hourly) / 1000) * (system_kWp / 350), 2)
    dc_total = round((np.sum(dc_hourly) / 1000) * (system_kWp / 350), 2)

    return {
        "ac_hourly": ac_hourly.tolist(),
        "dc_hourly": dc_hourly.tolist(),
        "ac_total_kWh": ac_total,
        "dc_total_kWh": dc_total
    }
