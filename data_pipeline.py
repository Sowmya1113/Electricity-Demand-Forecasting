# ============================================
# data_pipeline_v7.py
# PURPOSE: PRODUCTION-READY ENERGY FORECASTING (MULTI-SOURCE + EVALUATION)
# ============================================

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# ============================================
# CONFIG
# ============================================

class Config:
    NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    DEMAND_DATA_URL = "https://api.posoco.in/api/shortterm/power-demand"
    GENERATION_DATA_URL = "https://api.posoco.in/api/shortterm/generation"

    LAT = 20.5937
    LON = 78.9629

CONFIG = Config()

# ============================================
# LOGGER
# ============================================

def get_logger():
    logger = logging.getLogger("pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger()

# ============================================
# DATA FETCHER
# ============================================

class DataFetcher:
    def __init__(self):
        self.session = requests.Session()

    def fetch_weather(self, start, end):
        params = {
            "latitude": CONFIG.LAT,
            "longitude": CONFIG.LON,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "parameters": "T2M,WS10M,ALLSKY_SFC_SW_DWN,PRECTOTCORR,CLOUD_AMT",
            "format": "JSON",
        }

        r = self.session.get(CONFIG.NASA_URL, params=params, timeout=30)
        data = r.json()["properties"]["parameter"]

        df = pd.DataFrame({
            "date": pd.to_datetime(list(data["T2M"].keys())),
            "temp": list(data["T2M"].values()),
            "wind": list(data["WS10M"].values()),
            "solar_rad": list(data["ALLSKY_SFC_SW_DWN"].values()),
            "rain": list(data["PRECTOTCORR"].values()),
            "cloud": list(data["CLOUD_AMT"].values()),
        })

        return df.replace([-999, -9999], np.nan).dropna().set_index("date")

    def fetch_demand(self):
        r = self.session.get(CONFIG.DEMAND_DATA_URL, timeout=30)
        df = pd.DataFrame(r.json())
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def fetch_generation(self):
        r = self.session.get(CONFIG.GENERATION_DATA_URL, timeout=30)
        df = pd.DataFrame(r.json())
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

# ============================================
# FEATURE ENGINEERING
# ============================================

class FeatureBuilder:
    def build(self, df):
        df = df.copy()

        df["temp_sq"] = df["temp"] ** 2
        df["wind_sq"] = df["wind"] ** 2

        df["demand_lag1"] = df["demand"].shift(1)
        df["demand_lag7"] = df["demand"].shift(7)

        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        return df.dropna()

# ============================================
# DEMAND MODEL
# ============================================

class DemandModel:
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# ============================================
# GENERATION MODEL (MULTI-SOURCE)
# ============================================

class GenerationModel:
    def __init__(self):
        base_model = GradientBoostingRegressor()
        self.model = MultiOutputRegressor(base_model)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# ============================================
# EVALUATION
# ============================================

def evaluate(y_true, y_pred, name="model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    logger.info(f"{name} MAE: {mae:.2f}")
    logger.info(f"{name} RMSE: {rmse:.2f}")

# ============================================
# PIPELINE
# ============================================

class Pipeline:
    def run(self):
        fetcher = DataFetcher()
        features = FeatureBuilder()

        end = datetime.now()
        start = end - timedelta(days=365)

        weather = fetcher.fetch_weather(start, end)
        demand = fetcher.fetch_demand()
        generation = fetcher.fetch_generation()

        df = weather.join(demand).join(generation)

        if df.empty:
            logger.error("No data available")
            return

        df = features.build(df)

        # SPLIT (time-based)
        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]

        feature_cols = [
            "temp","wind","solar_rad","rain","cloud",
            "temp_sq","wind_sq","demand_lag1","demand_lag7",
            "day_of_week","month"
        ]

        # DEMAND MODEL
        demand_model = DemandModel()
        demand_model.train(train[feature_cols], train["demand"])

        demand_preds = demand_model.predict(test[feature_cols])
        evaluate(test["demand"], demand_preds, "Demand")

        # GENERATION MODEL (multi-source)
        gen_cols = ["solar_gen","wind_gen","hydro_gen","thermal_gen"]

        gen_model = GenerationModel()
        gen_model.train(train[["demand"]], train[gen_cols])

        gen_preds = gen_model.predict(test[["demand"]])
        evaluat
