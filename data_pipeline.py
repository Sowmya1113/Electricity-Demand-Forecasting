# ============================================
# data_pipeline.py
# PURPOSE: Handle ALL data operations for electricity demand forecasting
# OPTIMIZED VERSION - Faster, Cleaner, More Maintainable
# ============================================

import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================
# SECTION 1: CONFIGURATION (OPTIMIZED)
# ============================================

@dataclass(frozen=True)
class Config:
    """Centralized configuration - immutable and type-safe"""
    # API Settings
    NASA_POWER_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_COMMUNITY: str = "RE"
    NASA_FORMAT: str = "JSON"
    EMBER_API_KEY: str = "22a3271e-b37d-3f53-f084-1c5ffab5b64d"
    EMBER_BASE_URL: str = "https://api.ember-energy.org/v1/electricity-generation/monthly"
    
    # India Location
    INDIA_LAT: float = 20.5937
    INDIA_LON: float = 78.9629
    
    # Wind Standards (IEC 61400-1)
    WIND_CUT_IN: float = 3.0
    WIND_CUT_OUT: float = 25.0
    WIND_OPTIMAL: float = 12.0
    
    # Temperature Standards (ASHRAE 55)
    HEATING_BASE: float = 18.0
    COOLING_BASE: float = 21.0
    
    # Physical Limits
    TEMP_RANGE: Tuple[float, float] = (-30.0, 55.0)
    HUMIDITY_RANGE: Tuple[float, float] = (0.0, 100.0)
    WIND_RANGE: Tuple[float, float] = (0.0, 50.0)
    SOLAR_MAX: float = 12.0
    
    # Quality Rules
    MIN_COMPLETENESS: float = 0.95
    MAX_OUTLIERS: float = 0.05
    MAX_TEMP_CHANGE: float = 10.0
    
    # Feature Settings
    LAG_HOURS: Tuple = (1, 2, 3, 24, 48, 168)
    ROLLING_WINDOWS: Tuple = (3, 7, 14, 30)
    
    # Model Settings
    INPUT_LENGTH: int = 720
    OUTPUT_LENGTH: int = 720
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.0005
    
    # City Profiles (India-specific)
    CITY_PROFILES: Dict = None
    
    def __post_init__(self):
        object.__setattr__(self, 'CITY_PROFILES', {
            "Delhi": {"type": "inland", "solar": 0.9, "wind": 0.3},
            "Mumbai": {"type": "coastal", "solar": 0.8, "wind": 0.9},
            "Chennai": {"type": "coastal", "solar": 0.85, "wind": 0.85},
            "Bengaluru": {"type": "inland", "solar": 0.85, "wind": 0.6},
            "Kolkata": {"type": "coastal", "solar": 0.7, "wind": 0.6},
            "Hyderabad": {"type": "inland", "solar": 0.9, "wind": 0.5},
            "Jaipur": {"type": "inland", "solar": 0.95, "wind": 0.5},
            "Ahmedabad": {"type": "inland", "solar": 0.95, "wind": 0.55},
        })

CONFIG = Config()

# ============================================
# SECTION 2: LOGGING & DECORATORS (OPTIMIZED)
# ============================================

def setup_logger(name: str = "DataPipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}")
                    import time
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

def timed_cache(seconds: int = 3600):
    """Time-based cache decorator"""
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = datetime.now()
            if key in cache:
                result, timestamp = cache[key]
                if (now - timestamp).total_seconds() < seconds:
                    return result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

# ============================================
# SECTION 3: INDIA DEFAULTS (OPTIMIZED)
# ============================================

class IndiaDefaults:
    """Singleton pattern for India defaults - computed once and cached"""
    
    _instance = None
    _cache_path = os.path.join(os.path.dirname(__file__), "cache", "india_defaults.json")
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        self._data = self._build_or_load()
    
    def _build_or_load(self) -> Dict:
        """Build from APIs or load from cache"""
        data = self._fetch_from_apis()
        if data:
            self._save_cache(data)
            return data
        
        cached = self._load_cache()
        if cached:
            logger.info("Using cached India defaults")
            return cached
        
        raise RuntimeError("Cannot build India defaults - APIs unavailable and no cache")
    
    def _fetch_from_apis(self) -> Optional[Dict]:
        """Fetch from both APIs in parallel"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            weather_future = executor.submit(self._fetch_weather)
            demand_future = executor.submit(self._fetch_demand)
            
            weather = weather_future.result()
            demand = demand_future.result()
        
        if weather and demand:
            return {**weather, **demand}
        return None
    
    def _fetch_weather(self) -> Dict:
        """Fetch weather climatology from NASA"""
        current_year = datetime.now().year
        params = {
            "community": CONFIG.NASA_COMMUNITY,
            "longitude": CONFIG.INDIA_LON,
            "latitude": CONFIG.INDIA_LAT,
            "start": f"{current_year - 5}0101",
            "end": f"{current_year - 1}1231",
            "format": CONFIG.NASA_FORMAT,
            "parameters": "T2M,RH2M,WS10M,WS50M,ALLSKY_SFC_SW_DWN,PS",
        }
        try:
            resp = requests.get(CONFIG.NASA_POWER_URL, params=params, timeout=45)
            resp.raise_for_status()
            raw = resp.json().get("properties", {}).get("parameter", {})
            if not raw:
                return {}
            
            def clean(key: str) -> np.ndarray:
                vals = [float(v) for v in raw.get(key, {}).values() 
                       if v not in (-999, -9999, None) and not np.isnan(float(v))]
                return np.array(vals) if vals else np.array([])
            
            t2m = clean("T2M")
            if t2m.size == 0:
                return {}
            
            return {
                "base_temp": round(float(t2m.mean()), 2),
                "temp_amplitude": round(float(t2m.std()), 2),
                "humidity": round(float(clean("RH2M").mean()), 2) if clean("RH2M").size else 60,
                "wind_speed": round(float(clean("WS10M").mean()), 2) if clean("WS10M").size else 3,
                "solar": round(float(clean("ALLSKY_SFC_SW_DWN").mean()), 2) if clean("ALLSKY_SFC_SW_DWN").size else 5,
            }
        except Exception as e:
            logger.warning(f"Weather fetch failed: {e}")
            return {}
    
    def _fetch_demand(self) -> Dict:
        """Fetch demand statistics from Ember"""
        params = {
            "api_key": CONFIG.EMBER_API_KEY,
            "entity_code": "IND",
            "start_date": "2019-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d"),
        }
        try:
            resp = requests.get(CONFIG.EMBER_BASE_URL, params=params, timeout=45)
            if resp.status_code != 200:
                return {}
            
            data = resp.json().get("data", [])
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            demand = df[df["series"] == "Demand"]["generation_twh"].dropna()
            if demand.empty:
                return {}
            
            demand_mw = (demand * 1_000_000) / 730
            return {
                "demand_base": round(float(demand_mw.mean())),
                "demand_std": round(float(demand_mw.std())),
            }
        except Exception as e:
            logger.warning(f"Demand fetch failed: {e}")
            return {}
    
    def _save_cache(self, data: Dict):
        try:
            with open(self._cache_path, "w") as f:
                json.dump({"built_at": datetime.now().isoformat(), "defaults": data}, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _load_cache(self) -> Optional[Dict]:
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r") as f:
                    cached = json.load(f)
                    return cached.get("defaults", {})
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None
    
    @property
    def data(self) -> Dict:
        return self._data

INDIA_DEFAULTS = IndiaDefaults().data

# ============================================
# SECTION 4: NASA POWER CLIENT (OPTIMIZED)
# ============================================

class NASAPowerClient:
    """Optimized NASA POWER API client with caching"""
    
    PARAMETERS = {
        "T2M": "temperature_2m",
        "RH2M": "relative_humidity",
        "WS10M": "wind_speed_10m",
        "ALLSKY_SFC_SW_DWN": "solar_radiation",
        "PRECTOTCORR": "precipitation",
        "CLOUD_AMT": "cloud_cover",
    }
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ElectricityForecast/1.0"})
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._session.close()
    
    @timed_cache(seconds=3600)  # Cache for 1 hour
    def fetch_forecast(self, lat: float, lon: float, days: int = 30) -> pd.DataFrame:
        """Fetch weather forecast - cached for 1 hour"""
        start = datetime.now()
        end = start + timedelta(days=days)
        return self._fetch_daily_data(lat, lon, start, end)
    
    @retry_on_failure(max_retries=3)
    def _fetch_daily_data(self, lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
        """Internal method with retry logic"""
        params = {
            "community": CONFIG.NASA_COMMUNITY,
            "longitude": lon,
            "latitude": lat,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": CONFIG.NASA_FORMAT,
            "parameters": ",".join(self.PARAMETERS.keys()),
        }
        
        resp = self._session.get(CONFIG.NASA_POWER_URL, params=params, timeout=30)
        resp.raise_for_status()
        return self._parse_response(resp.json())
    
    def _parse_response(self, response: Dict) -> pd.DataFrame:
        """Parse NASA response efficiently"""
        data = response.get("properties", {}).get("parameter", {})
        if not data:
            return pd.DataFrame()
        
        dates = list(next(iter(data.values())).keys())
        if not dates:
            return pd.DataFrame()
        
        records = []
        for d in dates:
            record = {"datetime": pd.to_datetime(d, format="%Y%m%d")}
            for api_key, col_name in self.PARAMETERS.items():
                val = data.get(api_key, {}).get(d, np.nan)
                if val in (-999, -9999):
                    val = np.nan
                record[col_name] = float(val) if val is not None else np.nan
            records.append(record)
        
        return pd.DataFrame(records).set_index("datetime").sort_index()

# ============================================
# SECTION 5: EMBER CLIENT (OPTIMIZED)
# ============================================

class EmberClient:
    """Optimized Ember Energy API client"""
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ElectricityForecast/1.0"})
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self._session.close()
    
    @timed_cache(seconds=86400)  # Cache for 24 hours
    def get_energy_mix(self, iso_code: str = "IND") -> Dict[str, float]:
        """Get latest energy mix - cached for 24 hours"""
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        params = {
            "api_key": CONFIG.EMBER_API_KEY,
            "entity_code": iso_code,
            "start_date": six_months_ago,
            "end_date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        try:
            resp = self._session.get(CONFIG.EMBER_BASE_URL, params=params, timeout=30)
            if resp.status_code != 200:
                return self._fallback_mix()
            
            data = resp.json().get("data", [])
            if not data:
                return self._fallback_mix()
            
            df = pd.DataFrame(data)
            latest = df[df["date"] == df["date"].max()]
            latest = latest[latest.get("is_aggregate_series", True) == False]
            
            result = {}
            for _, row in latest.iterrows():
                series = row.get("series")
                pct = row.get("share_of_generation_pct")
                if series and pct is not None:
                    result[series] = float(pct)
            
            if result:
                result["thermal"] = result.get("Coal", 0) + result.get("Gas", 0)
                result["renewable"] = sum(result.get(s, 0) for s in ["Solar", "Wind", "Hydro", "Bioenergy"])
                result["is_real"] = True
                return result
            
            return self._fallback_mix()
            
        except Exception as e:
            logger.warning(f"Ember API failed: {e}")
            return self._fallback_mix()
    
    def _fallback_mix(self) -> Dict:
        """Fallback values when API fails"""
        return {
            "Coal": 55.0, "Gas": 10.0, "Nuclear": 3.0,
            "Hydro": 12.0, "Solar": 10.0, "Wind": 8.0,
            "thermal": 65.0, "renewable": 32.0,
            "is_real": False
        }

# ============================================
# SECTION 6: FEATURE ENGINEER (OPTIMIZED)
# ============================================

class FeatureEngineer:
    """Optimized feature engineering with vectorized operations"""
    
    def __init__(self):
        self._feature_names = []
    
    def create_features(self, df: pd.DataFrame, demand_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Create all features efficiently using vectorized operations"""
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Temporal features (vectorized)
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_peak_hour"] = ((df["hour"].between(7, 9)) | (df["hour"].between(17, 21))).astype(int)
        
        # Cyclical features (vectorized)
        for col, period in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
        
        # Energy features (vectorized)
        if "temperature_2m" in df.columns:
            temp = df["temperature_2m"]
            df["cooling_degree"] = np.maximum(0, temp - CONFIG.COOLING_BASE)
            df["heating_degree"] = np.maximum(0, CONFIG.HEATING_BASE - temp)
        
        if "wind_speed_10m" in df.columns:
            wind = df["wind_speed_10m"]
            df["wind_power"] = np.where(
                (wind >= CONFIG.WIND_CUT_IN) & (wind <= CONFIG.WIND_CUT_OUT),
                ((wind - CONFIG.WIND_CUT_IN) / (CONFIG.WIND_OPTIMAL - CONFIG.WIND_CUT_IN)) ** 3,
                0
            )
        
        if "solar_radiation" in df.columns:
            df["solar_power"] = (df["solar_radiation"] / CONFIG.SOLAR_MAX) * 100
        
        # Rolling features (efficient using .rolling)
        for col in ["temperature_2m", "relative_humidity"]:
            if col in df.columns:
                for w in CONFIG.ROLLING_WINDOWS:
                    df[f"{col}_ma_{w}"] = df[col].rolling(w, min_periods=1).mean()
        
        # Add demand features if available
        if demand_df is not None and "demand_mw" in demand_df.columns:
            demand_aligned = demand_df["demand_mw"].reindex(df.index)
            for lag in CONFIG.LAG_HOURS:
                df[f"demand_lag_{lag}"] = demand_aligned.shift(lag)
            df["demand_ma_24"] = demand_aligned.rolling(24, min_periods=1).mean()
        
        # Drop NaN rows
        df = df.dropna()
        
        # Store feature names (exclude target and index)
        self._feature_names = [c for c in df.columns if c not in ["datetime", "demand_mw"]]
        
        return df, self._feature_names

# ============================================
# SECTION 7: ENERGY PREDICTOR (OPTIMIZED)
# ============================================

class EnergyPredictor:
    """Predict all energy sources from weather forecast"""
    
    def __init__(self, nasa_client: NASAPowerClient, ember_client: EmberClient):
        self.nasa = nasa_client
        self.ember = ember_client
    
    def predict_for_city(self, city_name: str, lat: float, lon: float, days: int = 30) -> pd.DataFrame:
        """Predict all energy sources for a city"""
        # Get real weather forecast
        weather = self.nasa.fetch_forecast(lat, lon, days)
        
        # Get real energy mix from Ember
        energy_mix = self.ember.get_energy_mix("IND")
        nuclear_pct = energy_mix.get("Nuclear", 3.0)
        
        # Get city profile
        profile = CONFIG.CITY_PROFILES.get(city_name, {"type": "inland", "solar": 0.8, "wind": 0.5})
        
        predictions = []
        for date, row in weather.iterrows():
            # Predict from real weather data
            solar = self._calc_solar(row.get("solar_radiation", 5), row.get("cloud_cover", 30), date.month)
            wind = self._calc_wind(row.get("wind_speed_10m", 5), profile.get("type", "inland"))
            hydro = self._calc_hydro(row.get("precipitation", 0), date.month)
            
            renewable = solar + wind + hydro + nuclear_pct
            thermal = max(0, 100 - renewable)
            
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "solar": round(solar, 1),
                "wind": round(wind, 1),
                "hydro": round(hydro, 1),
                "nuclear": round(nuclear_pct, 1),
                "thermal": round(thermal, 1),
                "renewable_total": round(renewable, 1),
                "best_source": self._best_source({"solar": solar, "wind": wind, "hydro": hydro})
            })
        
        return pd.DataFrame(predictions)
    
    def _calc_solar(self, radiation: float, cloud: float, month: int) -> float:
        """Calculate solar percentage from real NASA data"""
        if radiation <= 0:
            return 0
        solar = (radiation / CONFIG.SOLAR_MAX) * 100
        solar *= (1 - cloud / 100 * 0.7)
        if month in [3, 4, 5]:  # Summer
            solar *= 1.2
        elif month in [11, 12, 1, 2]:  # Winter
            solar *= 0.8
        return min(100, max(0, solar))
    
    def _calc_wind(self, speed: float, location: str) -> float:
        """Calculate wind percentage from real NASA data"""
        if speed < CONFIG.WIND_CUT_IN or speed > CONFIG.WIND_CUT_OUT:
            return 0
        
        if speed <= CONFIG.WIND_OPTIMAL:
            eff = (speed - CONFIG.WIND_CUT_IN) / (CONFIG.WIND_OPTIMAL - CONFIG.WIND_CUT_IN)
        else:
            eff = 1 - (speed - CONFIG.WIND_OPTIMAL) / (CONFIG.WIND_CUT_OUT - CONFIG.WIND_OPTIMAL)
        
        wind = eff * 100
        if location == "coastal":
            wind *= 1.3
        
        return min(100, max(0, wind))
    
    def _calc_hydro(self, rain: float, month: int) -> float:
        """Calculate hydro percentage from real NASA data"""
        hydro = min(100, rain * 5)
        if month in [6, 7, 8, 9]:  # Monsoon
            hydro *= 1.5
        return min(100, max(0, hydro))
    
    def _best_source(self, sources: Dict) -> str:
        """Return best energy source recommendation"""
        best = max(sources, key=sources.get)
        value = sources[best]
        if value > 50:
            return f"{best} ({value:.0f}% available)"
        elif value > 25:
            return f"{best} ({value:.0f}% - moderate)"
        else:
            return f"Thermal (renewable low at {value:.0f}%)"

# ============================================
# SECTION 8: MAIN PIPELINE (OPTIMIZED)
# ============================================

class DataPipeline:
    """Main optimized pipeline orchestrator"""
    
    def __init__(self):
        self.nasa = NASAPowerClient()
        self.ember = EmberClient()
        self.feature_engineer = FeatureEngineer()
        self.energy_predictor = EnergyPredictor(self.nasa, self.ember)
    
    def get_weather_forecast(self, lat: float, lon: float, days: int = 30) -> pd.DataFrame:
        """Get weather forecast for any location"""
        return self.nasa.fetch_forecast(lat, lon, days)
    
    def get_energy_mix(self) -> Dict:
        """Get real energy mix from Ember API"""
        return self.ember.get_energy_mix("IND")
    
    def predict_energy_sources(self, city_name: str, lat: float, lon: float, days: int = 30) -> pd.DataFrame:
        """Predict all energy sources for a city"""
        return self.energy_predictor.predict_for_city(city_name, lat, lon, days)
    
    def prepare_features(self, weather_df: pd.DataFrame, demand_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for model training"""
        return self.feature_engineer.create_features(weather_df, demand_df)

# ============================================
# SECTION 9: EXPORTS
# ============================================

__all__ = [
    "Config", "CONFIG",
    "NASAPowerClient", "EmberClient",
    "FeatureEngineer", "EnergyPredictor", "DataPipeline",
    "INDIA_DEFAULTS",
    "setup_logger", "logger",
]
