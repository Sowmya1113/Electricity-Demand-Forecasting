# ============================================
# data_pipeline.py
# PURPOSE: Handle ALL data operations for electricity demand forecasting
# - Fetch weather data from NASA POWER API
# - Load and preprocess electricity demand data
# - Engineer features for NHiTS & iTransformer models
# - Validate and clean datasets
# ============================================

# ============================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================
import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any, Union

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================
# SECTION 9: CONFIGURATION CONSTANTS
# ============================================
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_COMMUNITY = "RE"
NASA_POWER_FORMAT = "JSON"

DEFAULT_LAG_HOURS = [1, 2, 3, 24, 48, 168]
DEFAULT_ROLLING_WINDOWS = [3, 7, 14, 30]
CYCLICAL_FEATURES = ["hour", "day_of_week", "month"]

HEATING_BASE_TEMP = 18
COOLING_BASE_TEMP = 21
WIND_CUT_IN_SPEED = 3
WIND_CUT_OUT_SPEED = 25

MIN_COMPLETENESS_RATIO = 0.95
MAX_ALLOWED_OUTLIERS = 0.05
MAX_TEMP_CHANGE_PER_HOUR = 10

TEMP_MIN, TEMP_MAX = -30, 55
HUMIDITY_MIN, HUMIDITY_MAX = 0, 100
WIND_SPEED_MIN, WIND_SPEED_MAX = 0, 50
SOLAR_MIN, SOLAR_MAX = 0, 12


# ============================================
# SECTION 8: EXCEPTION HANDLING & LOGGING
# ============================================
class DataPipelineError(Exception):
    """Base exception for data pipeline errors"""

    pass


class APIFetchError(DataPipelineError):
    """Raised when NASA POWER API fails"""

    pass


class DataValidationError(DataPipelineError):
    """Raised when data fails validation checks"""

    pass


class MissingDataError(DataPipelineError):
    """Raised when required data is missing"""

    pass


def setup_logger(name: str = "DataPipeline") -> logging.Logger:
    """Configure logging for the data pipeline"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


# ============================================
# SECTION 2: NASA POWER API CLIENT CLASS
# ============================================
class NASAPowerClient:
    """
    Handles all interactions with NASA POWER API
    PURPOSE: Get weather data for electricity forecasting
    """

    PARAMETERS = {
        "T2M": "temperature_2m",
        "T2M_MAX": "temperature_max",
        "T2M_MIN": "temperature_min",
        "RH2M": "relative_humidity",
        "WS10M": "wind_speed_10m",
        "WS50M": "wind_speed_50m",
        "ALLSKY_SFC_SW_DWN": "solar_radiation",
        "PRECTOTCORR": "precipitation",
        "CLOUD_AMT": "cloud_cover",
        "PS": "surface_pressure",
    }

    def __init__(self):
        self.base_url = NASA_POWER_BASE_URL
        self.community = NASA_POWER_COMMUNITY
        self.format = NASA_POWER_FORMAT
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ElectricityForecast/1.0"})

    def fetch_daily_data(
        self,
        latitude: float,
        longitude: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """
        Fetch daily weather data from NASA POWER API
        INPUT: Location coordinates, date range
        OUTPUT: DataFrame with temperature, humidity, wind, solar, precipitation
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y%m%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y%m%d")

        params = {
            "community": self.community,
            "longitude": longitude,
            "latitude": latitude,
            "start": start_date,
            "end": end_date,
            "format": self.format,
            "parameters": ",".join(self.PARAMETERS.keys()),
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "messages" in data:
                logger.info(f"API messages: {data.get('messages')}")

            return self._parse_api_response(data)
        except Exception as e:
            logger.warning(f"API fetch failed: {e}. Using fallback data.")
            return self._generate_fallback_data(start_date, end_date)

    def fetch_current_weather(self, latitude: float, longitude: float) -> Dict:
        """
        Get present weather conditions (near real-time)
        INPUT: Location coordinates
        OUTPUT: Dictionary with current weather values
        NOTE: NASA POWER has ~2 day latency for NRT data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        try:
            df = self.fetch_daily_data(latitude, longitude, start_date, end_date)
            if not df.empty:
                return df.iloc[-1].to_dict()
        except Exception as e:
            logger.warning(f"Current weather fetch failed: {e}")

        return self._generate_fallback_weather()

    def fetch_forecast(
        self, latitude: float, longitude: float, days: int = 90
    ) -> pd.DataFrame:
        """
        Get weather forecast for next N days
        INPUT: Location, forecast horizon (5-90 days)
        OUTPUT: DataFrame with forecasted weather parameters
        USED FOR: Future demand prediction
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days)

        return self.fetch_daily_data(latitude, longitude, start_date, end_date)

    def fetch_climatology(self, latitude: float, longitude: float) -> pd.DataFrame:
        """
        Get historical averages for baseline comparison
        INPUT: Location coordinates
        OUTPUT: Monthly climatology data
        USED FOR: Anomaly detection, seasonal normalization
        """
        current_year = datetime.now().year
        start_date = f"{current_year - 5}0101"
        end_date = f"{current_year - 1}1231"

        try:
            df = self.fetch_daily_data(latitude, longitude, start_date, end_date)
            if not df.empty:
                df["month"] = pd.to_datetime(df["datetime"]).dt.month
                climatology = df.groupby("month").mean()
                return climatology
        except Exception as e:
            logger.warning(f"Climatology fetch failed: {e}")

        return self._generate_default_climatology()

    def _parse_api_response(self, response_json: Dict) -> pd.DataFrame:
        """Convert NASA POWER JSON response to DataFrame"""
        try:
            data = response_json.get("properties", {}).get("parameter", {})
            if not data:
                return pd.DataFrame()

            dates = data.get("ALLSKY_SFC_SW_DWN", {}).keys()
            if not dates:
                return pd.DataFrame()

            records = []
            for date_str in dates:
                record = {"datetime": pd.to_datetime(date_str, format="%Y%m%d")}
                for param, col_name in self.PARAMETERS.items():
                    param_data = data.get(param, {})
                    record[col_name] = param_data.get(date_str, np.nan)
                records.append(record)

            df = pd.DataFrame(records)
            df = df.set_index("datetime").sort_index()
            return df
        except Exception as e:
            logger.error(f"Failed to parse API response: {e}")
            return pd.DataFrame()

    def _generate_fallback_data(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """Generate synthetic weather data when API fails"""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y%m%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y%m%d")

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        n_days = len(date_range)

        np.random.seed(42)
        seasonal_pattern = np.sin(np.linspace(0, 2 * np.pi, 365))[:n_days]

        base_temp = 20 + seasonal_pattern * 10

        data = {
            "datetime": date_range,
            "temperature_2m": base_temp + np.random.normal(0, 2, n_days),
            "temperature_max": base_temp + 5 + np.random.normal(0, 1.5, n_days),
            "temperature_min": base_temp - 5 + np.random.normal(0, 1.5, n_days),
            "relative_humidity": 60
            + seasonal_pattern * 20
            + np.random.normal(0, 10, n_days),
            "wind_speed_10m": 3 + np.random.exponential(2, n_days),
            "wind_speed_50m": 5 + np.random.exponential(3, n_days),
            "solar_radiation": np.maximum(
                0, 5 + seasonal_pattern * 3 + np.random.normal(0, 1, n_days)
            ),
            "precipitation": np.random.exponential(2, n_days),
            "cloud_cover": np.clip(
                30 + seasonal_pattern * 30 + np.random.normal(0, 15, n_days), 0, 100
            ),
            "surface_pressure": 1013 + np.random.normal(0, 10, n_days),
        }

        df = pd.DataFrame(data)
        df = df.set_index("datetime")

        for col in ["relative_humidity", "cloud_cover"]:
            df[col] = df[col].clip(0, 100)

        return df

    def _generate_fallback_weather(self) -> Dict:
        """Generate single fallback weather reading"""
        return {
            "temperature_2m": 20.0,
            "temperature_max": 24.0,
            "temperature_min": 16.0,
            "relative_humidity": 60,
            "wind_speed_10m": 5.0,
            "wind_speed_50m": 8.0,
            "solar_radiation": 4.0,
            "precipitation": 0.0,
            "cloud_cover": 30,
            "surface_pressure": 1013,
        }

    def _generate_default_climatology(self) -> pd.DataFrame:
        """Generate default monthly climatology"""
        months = range(1, 13)
        seasonal = np.sin(2 * np.pi * np.array(months) / 12)

        data = {
            "month": months,
            "temperature_2m": 15 + seasonal * 10,
            "relative_humidity": 60 + seasonal * 15,
            "wind_speed_10m": 5 + np.random.uniform(-1, 1, 12),
            "solar_radiation": 4 + seasonal * 2,
        }
        return pd.DataFrame(data).set_index("month")



# ============================================
# SECTION 2B: EMBER ENERGY API CLIENT CLASS
# ============================================
class EmberEnergyClient:
    """
    Handles all interactions with Ember Energy API
    PURPOSE: Get real-world electricity generation and mix data for India
    """

    BASE_URL = "https://api.ember-energy.org/v1/electricity-generation/monthly"

    def __init__(self, api_key: str = "22a3271e-b37d-3f53-f084-1c5ffab5b64d"):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ElectricityProject/1.0"})

    def fetch_generation_mix(
        self, iso_code: str = "IND", start_date: str = "2021-01-01", end_date: str = "2026-12-31"
    ) -> pd.DataFrame:
        """
        Fetch monthly electricity generation data from Ember Energy API
        INPUT: ISO Alpha-3 country code, date range
        OUTPUT: DataFrame with date, fuel type, generation (TWh), and share (%)
        """
        params = {
            "api_key": self.api_key,
            "entity_code": iso_code,
            "start_date": start_date,
            "end_date": end_date,
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                result = response.json()
                data = result.get("data", []) if isinstance(result, dict) else result
                
                if not data:
                    logger.warning("No data found in Ember API response")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                return df
            else:
                logger.error(f"Ember API error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ember API fetch failed: {e}")
            return pd.DataFrame()

    def get_latest_mix_percentages(self, iso_code: str = "IND") -> Dict[str, float]:
        """
        Get the most recent fuel mix percentages for the given country
        OUTPUT: Dictionary of {fuel_type: percentage}
        """
        df = self.fetch_generation_mix(iso_code)
        if df.empty:
            return {}

        latest_date = df["date"].max()
        latest_data = df[df["date"] == latest_date]
        
        # We only want non-aggregate series for a clean breakdown
        # Common Ember series: Coal, Gas, Hydro, Solar, Wind, Bioenergy, Nuclear, Other fossil, Other renewables
        breakdown = latest_data[latest_data["is_aggregate_series"] == False]
        
        mix = {}
        for _, row in breakdown.iterrows():
            mix[row["series"]] = row["share_of_generation_pct"]
            
        return mix


# ============================================
# SECTION 3: FEATURE ENGINEERING CLASS
# ============================================
class FeatureEngineer:
    """
    Creates all features needed for demand forecasting models
    PURPOSE: Transform raw weather & demand data into model-ready features
    """

    def __init__(self):
        self.feature_list = []
        self.feature_config = {
            "heating_base_temp": HEATING_BASE_TEMP,
            "cooling_base_temp": COOLING_BASE_TEMP,
            "wind_cut_in": WIND_CUT_IN_SPEED,
            "wind_cut_out": WIND_CUT_OUT_SPEED,
        }

    def create_all_features(
        self, weather_df: pd.DataFrame, demand_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        MAIN METHOD: Complete feature engineering pipeline
        INPUT: Weather DataFrame, optional demand DataFrame
        OUTPUT: DataFrame with all engineered features
        """
        df = weather_df.copy()

        df = self._add_temporal_features(df)
        df = self._add_cyclical_features(df)
        df = self._add_energy_features(df)

        if demand_df is not None:
            df = self._add_lag_features(df, demand_df)

        df = self._add_rolling_features(df)
        df = self._add_interaction_features(df)

        self.feature_list = [
            c for c in df.columns if c not in ["datetime", "demand_mw"]
        ]
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from datetime index"""
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")

        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["day_of_year"] = df.index.dayofyear
        df["week_of_year"] = df.index.isocalendar().week

        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_peak_hour"] = (
            (df["hour"] >= 7) & (df["hour"] <= 9)
            | (df["hour"] >= 17) & (df["hour"] <= 21)
        ).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

        df = self._add_holiday_features(df)

        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday indicators - India specific"""
        df = df.copy()

        india_holidays = [
            "01-01",
            "01-26",
            "08-15",
            "10-02",
            "12-25",
            "01-14",
            "03-08",
            "04-02",
            "04-10",
            "04-14",
            "05-01",
            "08-19",
            "09-17",
            "10-20",
            "11-01",
        ]

        df["date_str"] = df.index.strftime("%m-%d")
        df["is_holiday"] = df["date_str"].isin(india_holidays).astype(int)
        df = df.drop(columns=["date_str"])

        df["is_festival_season"] = ((df["month"] >= 10) | (df["month"] <= 1)).astype(
            int
        )

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert circular temporal features to continuous values"""
        df = df.copy()

        cyclical_mappings = {
            "hour": 24,
            "day_of_week": 7,
            "month": 12,
            "day_of_year": 365,
        }

        for col, period in cyclical_mappings.items():
            if col in df.columns:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

        return df

    def _add_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create electricity-sector specific derived features"""
        df = df.copy()

        if "temperature_2m" in df.columns:
            temp = df["temperature_2m"]
            base_heat = self.feature_config["heating_base_temp"]
            base_cool = self.feature_config["cooling_base_temp"]

            df["heating_degree"] = np.maximum(0, base_heat - temp)
            df["cooling_degree"] = np.maximum(0, temp - base_cool)
            df["temp_deviation"] = np.abs(temp - (base_heat + base_cool) / 2)

            df["heat_index"] = (
                temp + 0.5 * (df["relative_humidity"] - 50)
                if "relative_humidity" in df.columns
                else temp
            )

        if "relative_humidity" in df.columns:
            df["humidity_deficit"] = 100 - df["relative_humidity"]
            df["humidity_stress"] = (df["relative_humidity"] > 70).astype(int)

        if "wind_speed_10m" in df.columns:
            wind = df["wind_speed_10m"]
            cut_in = self.feature_config["wind_cut_in"]
            cut_out = self.feature_config["wind_cut_out"]

            df["wind_power_potential"] = np.where(
                (wind >= cut_in) & (wind <= cut_out),
                ((wind - cut_in) / (cut_out - cut_in)) ** 3,
                0,
            )

        if "solar_radiation" in df.columns:
            df["solar_potential"] = df["solar_radiation"] / 12 * 100

        if "temperature_2m" in df.columns and "relative_humidity" in df.columns:
            df["temp_humidity_interaction"] = (
                df["temperature_2m"] * df["relative_humidity"] / 100
            )

        return df

    def _add_lag_features(
        self, df: pd.DataFrame, demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add historical demand values as predictors"""
        df = df.copy()

        if "demand_mw" not in demand_df.columns:
            return df

        demand_df = demand_df.copy()
        if not isinstance(demand_df.index, pd.DatetimeIndex):
            demand_df["datetime"] = pd.to_datetime(demand_df["datetime"])
            demand_df = demand_df.set_index("datetime")

        common_idx = df.index.intersection(demand_df.index)
        if len(common_idx) == 0:
            return df

        demand_aligned = demand_df.loc[common_idx, "demand_mw"]

        lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168]

        for lag in lag_hours:
            df.loc[common_idx, f"demand_lag_{lag}h"] = demand_aligned.shift(lag)

        df.loc[common_idx, "demand_rolling_mean_24h"] = demand_aligned.rolling(
            24
        ).mean()
        df.loc[common_idx, "demand_rolling_std_24h"] = demand_aligned.rolling(24).std()
        df.loc[common_idx, "demand_rolling_mean_168h"] = demand_aligned.rolling(
            168
        ).mean()

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages and standard deviations"""
        df = df.copy()

        rolling_cols = ["temperature_2m", "relative_humidity", "wind_speed_10m"]
        windows = [3, 7, 14, 30]

        for col in rolling_cols:
            if col in df.columns:
                for window in windows:
                    df[f"{col}_ma_{window}d"] = (
                        df[col].rolling(window, min_periods=1).mean()
                    )
                    df[f"{col}_std_{window}d"] = (
                        df[col].rolling(window, min_periods=1).std()
                    )

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined features from multiple variables"""
        df = df.copy()

        if "temperature_2m" in df.columns and "hour" in df.columns:
            df["temp_hour_interaction"] = df["temperature_2m"] * df["hour"]

        if "temperature_2m" in df.columns and "is_peak_hour" in df.columns:
            df["temp_peak_interaction"] = df["temperature_2m"] * df["is_peak_hour"]

        if "wind_speed_10m" in df.columns and "cloud_cover" in df.columns:
            df["wind_cloud_interaction"] = (
                df["wind_speed_10m"] * (100 - df["cloud_cover"]) / 100
            )

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of all created feature names"""
        return self.feature_list


# ============================================
# SECTION 4: DATA LOADER CLASS
# ============================================
class DataLoader:
    """
    Load and preprocess electricity demand datasets
    PURPOSE: Handle historical demand data from CSV files or APIs
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def load_demand_data(
        self,
        filepath: Optional[str] = None,
        date_column: str = "datetime",
        value_column: str = "demand_mw",
    ) -> pd.DataFrame:
        """
        Load electricity demand from CSV file
        INPUT: Path to CSV file
        OUTPUT: DataFrame with datetime index and demand column
        """
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=[date_column])
        else:
            logger.info("No local data found, fetching real demand from Ember API")
            df = self.load_from_api("india")

        df = df.set_index(date_column)
        df = df.sort_index()

        df = self.handle_missing_values(df)

        return df

    def _generate_synthetic_demand(self, days: int = 365) -> pd.DataFrame:
        """Generate synthetic demand data for testing"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), periods=days * 24, freq="H"
        )

        n = len(dates)
        hourly_pattern = np.sin(2 * np.pi * np.arange(n) / 24)
        weekly_pattern = np.sin(2 * np.pi * np.arange(n) / (24 * 7))
        seasonal_pattern = np.sin(2 * np.pi * np.arange(n) / (24 * 365))

        base_demand = 40000
        hourly_effect = hourly_pattern * 8000
        weekly_effect = weekly_pattern * 5000
        seasonal_effect = seasonal_pattern * 10000

        np.random.seed(42)
        weather_noise = np.random.normal(0, 2000, n)
        random_noise = np.random.normal(0, 1000, n)

        demand = (
            base_demand
            + hourly_effect
            + weekly_effect
            + seasonal_effect
            + weather_noise
            + random_noise
        )
        demand = np.maximum(demand, 15000)

        return pd.DataFrame({"datetime": dates, "demand_mw": demand})

    def load_from_api(self, region: str = "india") -> pd.DataFrame:
        """
        Load real demand data from Ember Energy API for India
        """
        logger.info(f"Loading real demand data from Ember API for region: {region}")
        try:
            client = EmberEnergyClient()
            iso_code = "IND" if region.lower() == "india" else region
            df = client.fetch_generation_mix(iso_code)
            
            if df.empty:
                logger.warning("Empty response from Ember API, falling back to synthetic")
                return self._generate_synthetic_demand()
                
            # Filter for Demand series
            demand_df = df[df["series"] == "Demand"].copy()
            
            if demand_df.empty:
                logger.warning("No 'Demand' series found in Ember data, falling back to synthetic")
                return self._generate_synthetic_demand()
                
            # Convert TWh to average MW
            demand_df["demand_mw"] = (demand_df["generation_twh"] * 1000000) / 730
            
            # Map columns to expected names
            demand_df = demand_df.rename(columns={"date": "datetime"})
            
            # Since Ember is monthly, we might need to resample/interpolate for models 
            # that expect hourly data, but for now we return the real trends.
            return demand_df[["datetime", "demand_mw"]]
            
        except Exception as e:
            logger.error(f"Ember API load failed: {e}. Falling back to synthetic.")
            return self._generate_synthetic_demand()

    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent hourly frequency"""
        if df.index.freq is None or df.index.freq != "H":
            df = df.resample("H").mean()

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill or interpolate missing demand values"""
        df = df.copy()

        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Handling {missing_count} missing values")

            df = df.interpolate(method="time")

            df = df.fillna(method="ffill")
            df = df.fillna(method="bfill")

        return df

    def detect_outliers(
        self, df: pd.DataFrame, column: str = "demand_mw"
    ) -> pd.DataFrame:
        """Identify anomalous demand values"""
        if column not in df.columns:
            return df

        df = df.copy()
        df["is_outlier"] = False

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        df.loc[
            (df[column] < lower_bound) | (df[column] > upper_bound), "is_outlier"
        ] = True

        outlier_count = df["is_outlier"].sum()
        if outlier_count > 0:
            logger.warning(f"Detected {outlier_count} outliers")

        return df

    def split_train_val_test(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data for model training/validation/testing"""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(
            f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        return train_df, val_df, test_df


# ============================================
# SECTION 5: DATA VALIDATION CLASS
# ============================================
class DataValidator:
    """
    Validate data quality before model training
    PURPOSE: Ensure >85% accuracy by catching data issues early
    """

    def __init__(self):
        self.validation_results = {}

    def validate_weather_data(self, df: pd.DataFrame) -> Dict:
        """Check weather data for physical plausibility"""
        issues = []

        if "temperature_2m" in df.columns:
            temp_issues = df[
                (df["temperature_2m"] < TEMP_MIN) | (df["temperature_2m"] > TEMP_MAX)
            ]
            if len(temp_issues) > 0:
                issues.append(f"Temperature out of range: {len(temp_issues)} values")

        if "relative_humidity" in df.columns:
            hum_issues = df[
                (df["relative_humidity"] < HUMIDITY_MIN)
                | (df["relative_humidity"] > HUMIDITY_MAX)
            ]
            if len(hum_issues) > 0:
                issues.append(f"Humidity out of range: {len(hum_issues)} values")

        if "wind_speed_10m" in df.columns:
            wind_issues = df[
                (df["wind_speed_10m"] < WIND_SPEED_MIN)
                | (df["wind_speed_10m"] > WIND_SPEED_MAX)
            ]
            if len(wind_issues) > 0:
                issues.append(f"Wind speed out of range: {len(wind_issues)} values")

        if "solar_radiation" in df.columns:
            solar_issues = df[
                (df["solar_radiation"] < SOLAR_MIN)
                | (df["solar_radiation"] > SOLAR_MAX)
            ]
            if len(solar_issues) > 0:
                issues.append(
                    f"Solar radiation out of range: {len(solar_issues)} values"
                )

        status = "PASS" if len(issues) == 0 else "FAIL"

        return {"status": status, "issues": issues, "total_records": len(df)}

    def validate_demand_data(self, df: pd.DataFrame) -> Dict:
        """Check demand data for operational plausibility"""
        issues = []
        quality_score = 100

        if "demand_mw" not in df.columns:
            return {
                "status": "FAIL",
                "issues": ["No demand column found"],
                "quality_score": 0,
            }

        negative_demand = (df["demand_mw"] < 0).sum()
        if negative_demand > 0:
            issues.append(f"Negative demand values: {negative_demand}")
            quality_score -= negative_demand / len(df) * 100

        zero_demand = (df["demand_mw"] == 0).sum()
        if zero_demand > len(df) * 0.01:
            issues.append(f"Excessive zero values: {zero_demand}")
            quality_score -= zero_demand / len(df) * 50

        if len(df) > 1:
            hour_changes = df["demand_mw"].pct_change().abs()
            sudden_jumps = (hour_changes > 0.5).sum()
            if sudden_jumps > 0:
                issues.append(f"Sudden jumps >50%: {sudden_jumps}")
                quality_score -= sudden_jumps / len(df) * 100

        status = "PASS" if quality_score >= 80 else "FAIL"

        return {
            "status": status,
            "issues": issues,
            "quality_score": max(0, quality_score),
        }

    def check_data_completeness(
        self, df: pd.DataFrame, expected_frequency: str = "H"
    ) -> Dict:
        """Check for missing timestamps"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"status": "UNKNOWN", "completeness": 0, "gaps": []}

        expected = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=expected_frequency
        )

        actual = df.index
        missing = expected.difference(actual)

        completeness = len(actual) / len(expected) if len(expected) > 0 else 0

        status = "PASS" if completeness >= MIN_COMPLETENESS_RATIO else "FAIL"

        return {
            "status": status,
            "completeness": completeness,
            "gaps": list(missing[:10]),
            "missing_count": len(missing),
        }

    def check_seasonal_consistency(self, df: pd.DataFrame) -> Dict:
        """Verify data follows expected seasonal patterns"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"status": "UNKNOWN", "anomalies": []}

        df = df.copy()
        df["month"] = df.index.month
        df["hour"] = df.index.hour

        monthly_avg = df.groupby("month").mean(numeric_only=True)

        anomalies = []

        if "demand_mw" in monthly_avg.columns:
            winter_months = [12, 1, 2]
            summer_months = [4, 5, 6]

            winter_avg = monthly_avg.loc[winter_months, "demand_mw"].mean()
            summer_avg = monthly_avg.loc[summer_months, "demand_mw"].mean()

            if winter_avg < summer_avg:
                anomalies.append(
                    "Winter demand should be higher than summer (heating load)"
                )

        return {
            "status": "PASS" if len(anomalies) == 0 else "WARNING",
            "anomalies": anomalies,
        }

    def generate_quality_report(
        self, weather_df: pd.DataFrame, demand_df: pd.DataFrame
    ) -> Dict:
        """Create comprehensive data quality report"""
        report = {
            "weather_validation": self.validate_weather_data(weather_df),
            "demand_validation": self.validate_demand_data(demand_df),
            "weather_completeness": self.check_data_completeness(weather_df),
            "demand_completeness": self.check_data_completeness(demand_df),
            "seasonal_check": self.check_seasonal_consistency(demand_df),
        }

        overall_score = (
            report["weather_validation"].get("quality_score", 100)
            + report["demand_validation"].get("quality_score", 100)
            + report["weather_completeness"].get("completeness", 0) * 100
            + report["demand_completeness"].get("completeness", 0) * 100
        ) / 4

        report["overall_score"] = overall_score

        return report


# ============================================
# SECTION 6: DATA PIPELINE ORCHESTRATOR
# ============================================
class DataPipeline:
    """
    Main orchestrator - connects all data components
    PURPOSE: Single interface for all data operations
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.weather_api = NASAPowerClient()
        self.feature_engineer = FeatureEngineer()
        self.data_loader = DataLoader()
        self.validator = DataValidator()
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.scaler = StandardScaler()

    def prepare_training_data(
        self,
        lat: float,
        lon: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        demand_filepath: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Complete pipeline for preparing training data
        STEPS:
        1. Fetch historical weather data
        2. Load historical demand data
        3. Validate both datasets
        4. Engineer features
        5. Align weather and demand
        6. Return model-ready dataset
        """
        logger.info("Starting training data preparation...")

        weather_df = self.weather_api.fetch_daily_data(lat, lon, start_date, end_date)
        logger.info(f"Fetched {len(weather_df)} weather records")

        demand_df = self.data_loader.load_demand_data(demand_filepath)
        logger.info(f"Loaded {len(demand_df)} demand records")

        weather_df = self.data_loader.resample_to_hourly(weather_df)

        quality_report = self.validator.generate_quality_report(weather_df, demand_df)
        logger.info(f"Data quality score: {quality_report['overall_score']:.1f}")

        aligned_df = self.align_weather_and_demand(weather_df, demand_df)

        features_df = self.feature_engineer.create_all_features(aligned_df, demand_df)

        features_df = features_df.dropna()

        feature_names = self.feature_engineer.get_feature_names()

        if "demand_mw" in features_df.columns:
            X = features_df[feature_names]
            y = features_df["demand_mw"]
        else:
            X = features_df[feature_names]
            y = None

        logger.info(
            f"Prepared {len(X)} training samples with {len(feature_names)} features"
        )

        return X, y, feature_names

    def prepare_forecast_data(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 7,
        last_known_demand: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Prepare data for future prediction
        STEPS:
        1. Fetch weather forecast
        2. Generate temporal features for forecast period
        3. Use last known demand for lags
        4. Return feature matrix for prediction
        """
        logger.info(f"Preparing {forecast_days}-day forecast data...")

        weather_forecast = self.weather_api.fetch_forecast(lat, lon, forecast_days)

        weather_forecast = self.feature_engineer.create_all_features(weather_forecast)

        if last_known_demand is not None and len(last_known_demand) > 0:
            lag_features = []
            for lag in [24, 48, 168]:
                if len(last_known_demand) >= lag:
                    weather_forecast[f"demand_lag_{lag}h"] = last_known_demand.iloc[
                        -lag
                    ]

            if len(last_known_demand) >= 24:
                weather_forecast["demand_rolling_mean_24h"] = last_known_demand.iloc[
                    -24:
                ].mean()
                weather_forecast["demand_rolling_mean_168h"] = (
                    last_known_demand.iloc[-168:].mean()
                    if len(last_known_demand) >= 168
                    else last_known_demand.mean()
                )

        feature_names = self.feature_engineer.get_feature_names()
        available_features = [f for f in feature_names if f in weather_forecast.columns]

        X_forecast = weather_forecast[available_features].fillna(0)

        return X_forecast

    def get_current_conditions(self, lat: float, lon: float) -> Dict:
        """Get present weather for real-time display"""
        weather = self.weather_api.fetch_current_weather(lat, lon)

        if "temperature_2m" in weather:
            temp = weather["temperature_2m"]
            weather["heating_degree"] = max(0, HEATING_BASE_TEMP - temp)
            weather["cooling_degree"] = max(0, temp - COOLING_BASE_TEMP)

        return weather

    def align_weather_and_demand(
        self, weather_df: pd.DataFrame, demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Ensure weather and demand have matching timestamps"""
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            weather_df.index = pd.to_datetime(weather_df.index)
        if not isinstance(demand_df.index, pd.DatetimeIndex):
            demand_df.index = pd.to_datetime(demand_df.index)

        common_index = weather_df.index.intersection(demand_df.index)

        if len(common_index) == 0:
            logger.warning("No overlapping timestamps found, using index alignment")
            min_len = min(len(weather_df), len(demand_df))
            weather_df = weather_df.iloc[:min_len]
            demand_df = demand_df.iloc[:min_len]
            weather_df["demand_mw"] = demand_df["demand_mw"].values
            return weather_df

        aligned = weather_df.loc[common_index].copy()
        aligned["demand_mw"] = demand_df.loc[common_index, "demand_mw"].values

        return aligned

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Cache processed data to avoid recomputation"""
        filepath = os.path.join(self.cache_dir, f"{filename}.parquet")
        df.to_parquet(filepath)
        logger.info(f"Saved processed data to {filepath}")

    def load_processed_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load previously cached processed data"""
        filepath = os.path.join(self.cache_dir, f"{filename}.parquet")
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded cached data from {filepath}")
            return df
        return None


# ============================================
# SECTION 7: HELPER FUNCTIONS
# ============================================
def validate_coordinates(lat: float, lon: float) -> bool:
    """Check if latitude/longitude are within valid ranges"""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def get_timezone_from_coordinates(lat: float, lon: float) -> str:
    """Estimate timezone from coordinates"""
    offset = round(lon / 15)
    return f"UTC{offset:+d}"


def calculate_season(month: int) -> str:
    """Determine season from month number"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def is_holiday(date: datetime, country: str = "India") -> bool:
    """Check if a date is a public holiday"""
    india_holidays = [
        datetime(date.year, 1, 1),
        datetime(date.year, 1, 26),
        datetime(date.year, 8, 15),
        datetime(date.year, 10, 2),
        datetime(date.year, 12, 25),
    ]
    return date in india_holidays


def get_default_parameters() -> Dict:
    """Return default parameter settings for the pipeline"""
    return {
        "forecast_horizon_days": 7,
        "feature_windows": DEFAULT_ROLLING_WINDOWS,
        "lag_periods": DEFAULT_LAG_HOURS,
        "validation_thresholds": {
            "min_completeness": MIN_COMPLETENESS_RATIO,
            "max_outliers": MAX_ALLOWED_OUTLIERS,
        },
        "heating_base_temp": HEATING_BASE_TEMP,
        "cooling_base_temp": COOLING_BASE_TEMP,
    }


# ============================================
# SECTION 10: EXPORTS
# ============================================
__all__ = [
    "NASAPowerClient",
    "EmberEnergyClient",
    "FeatureEngineer",
    "DataLoader",
    "DataValidator",
    "DataPipeline",
    "validate_coordinates",
    "calculate_season",
    "get_default_parameters",
    "DataPipelineError",
    "APIFetchError",
    "DataValidationError",
    "MissingDataError",
]
