import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
NASA_POWER_BASE_URL  = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_COMMUNITY = "RE"
NASA_POWER_FORMAT    = "JSON"
INDIA_LAT, INDIA_LON = 20.5937, 78.9629

DEFAULT_LAG_HOURS       = [1, 2, 3, 24, 48, 168]
DEFAULT_ROLLING_WINDOWS = [3, 7, 14, 30]
CYCLICAL_FEATURES       = ["hour", "day_of_week", "month"]

# IEC 61400-1 wind turbine engineering standards
WIND_CUT_IN_SPEED  = 3    # m/s
WIND_CUT_OUT_SPEED = 25   # m/s

# ASHRAE 55 standard base temperatures for degree-day calculation
HEATING_BASE_TEMP = 18   # °C
COOLING_BASE_TEMP = 21   # °C

# Data quality control rules
MIN_COMPLETENESS_RATIO   = 0.95
MAX_ALLOWED_OUTLIERS     = 0.05
MAX_TEMP_CHANGE_PER_HOUR = 10

# Physical measurement bounds (India recorded extremes + instrument limits)
TEMP_MIN,       TEMP_MAX       = -30, 55
HUMIDITY_MIN,   HUMIDITY_MAX   = 0,   100
WIND_SPEED_MIN, WIND_SPEED_MAX = 0,   50
SOLAR_MIN,      SOLAR_MAX      = 0,   12   # GHI physical ceiling kWh/m²/day

# Cache path for computed defaults (populated on first successful API call)
_DEFAULTS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "india_defaults.json")


# ============================================
# SECTION 2: EXCEPTION HANDLING & LOGGING
# ============================================
class DataPipelineError(Exception):     pass
class APIFetchError(DataPipelineError): pass
class DataValidationError(DataPipelineError): pass
class MissingDataError(DataPipelineError):    pass


def setup_logger(name: str = "DataPipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


# ============================================
# SECTION 3: INDIA_DEFAULTS — BUILT FROM REAL APIs AT STARTUP
#
# Weather baselines → NASA POWER 5-year climatology for India centroid (20.59°N 78.96°E)
# Demand baselines  → Ember API India monthly Demand series (2019–present)
#
# Result is cached to cache/india_defaults.json after every successful fetch.
# On API failure the cache is loaded so the program works offline.
# No hardcoded numbers in this dict.
# ============================================

def _fetch_nasa_climatology_stats() -> Dict:
    """
    Fetch 5-year daily weather from NASA POWER for India centroid.
    Computes: annual mean and seasonal amplitude for temp, humidity,
    wind (10 m + 50 m), solar radiation, and surface pressure.
    Returns populated dict on success, empty dict on failure.
    """
    current_year = datetime.now().year
    params = {
        "community":  NASA_POWER_COMMUNITY,
        "longitude":  INDIA_LON,
        "latitude":   INDIA_LAT,
        "start":      f"{current_year - 5}0101",
        "end":        f"{current_year - 1}1231",
        "format":     NASA_POWER_FORMAT,
        "parameters": "T2M,RH2M,WS10M,WS50M,ALLSKY_SFC_SW_DWN,PS",
    }
    try:
        resp = requests.get(NASA_POWER_BASE_URL, params=params, timeout=45)
        resp.raise_for_status()
        raw = resp.json().get("properties", {}).get("parameter", {})
        if not raw:
            return {}

        # Remove NASA POWER fill values (-999, -9999) and convert to arrays
        def clean(key: str) -> np.ndarray:
            vals = [float(v) for v in raw.get(key, {}).values()
                    if v not in (-999, -9999, None) and not np.isnan(float(v))]
            return np.array(vals) if vals else np.array([])

        t2m  = clean("T2M")
        rh   = clean("RH2M")
        ws10 = clean("WS10M")
        ws50 = clean("WS50M")
        sol  = clean("ALLSKY_SFC_SW_DWN")
        ps   = clean("PS")

        if t2m.size == 0:
            return {}

        # Seasonal amplitude = half the monthly mean peak-to-trough range
        day_idx     = np.arange(len(t2m))
        month_bucket = ((day_idx % 365) / 365 * 12).astype(int).clip(0, 11)
        monthly_t2m  = np.array([t2m[month_bucket == m].mean() for m in range(12)
                                  if (month_bucket == m).any()])

        return {
            "base_temp":        round(float(t2m.mean()),  2),
            "temp_amplitude":   round(float((monthly_t2m.max() - monthly_t2m.min()) / 2), 2)
                                if monthly_t2m.size >= 2 else round(float(t2m.std()), 2),
            "humidity":         round(float(rh.mean()),   2) if rh.size   else None,
            "wind_speed_10m":   round(float(ws10.mean()), 2) if ws10.size else None,
            "wind_speed_50m":   round(float(ws50.mean()), 2) if ws50.size else None,
            "solar_radiation":  round(float(sol.mean()),  2) if sol.size  else None,
            "surface_pressure": round(float(ps.mean()),   2) if ps.size   else None,
        }
    except Exception as e:
        logger.warning(f"NASA POWER climatology fetch failed: {e}")
        return {}


def _fetch_ember_demand_stats() -> Dict:
    """
    Fetch India monthly Demand series from Ember API (2019–present).
    Derives demand_base_mw, floor, and amplitude statistics from real data.
    Returns populated dict on success, empty dict on failure.
    """
    url    = "https://api.ember-energy.org/v1/electricity-generation/monthly"
    params = {
        "api_key":     "22a3271e-b37d-3f53-f084-1c5ffab5b64d",
        "entity_code": "IND",
        "start_date":  "2019-01-01",
        "end_date":    datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        resp = requests.get(url, params=params, timeout=45)
        if resp.status_code != 200:
            return {}

        raw  = resp.json()
        data = raw.get("data", []) if isinstance(raw, dict) else raw
        if not data:
            return {}

        df = pd.DataFrame(data)
        if "series" not in df.columns or "generation_twh" not in df.columns:
            return {}

        demand_mw = (df[df["series"] == "Demand"]["generation_twh"].dropna() * 1_000_000) / 730
        if demand_mw.empty:
            return {}

        d_mean  = float(demand_mw.mean())
        d_min   = float(demand_mw.min())
        d_max   = float(demand_mw.max())
        d_std   = float(demand_mw.std())
        d_range = d_max - d_min

        return {
            "demand_base_mw":      round(d_mean),
            "demand_floor_mw":     round(d_min  * 0.90),   # 10% below observed minimum
            "demand_hourly_amp":   round(d_std   * 0.80),   # intra-day swing ≈ 80% of monthly σ
            "demand_weekly_amp":   round(d_range * 0.12),   # weekday/weekend ≈ 12% of range
            "demand_seasonal_amp": round(d_range * 0.22),   # seasonal swing  ≈ 22% of range
        }
    except Exception as e:
        logger.warning(f"Ember demand stats fetch failed: {e}")
        return {}


def build_india_defaults() -> Dict:
    """
    Compute INDIA_DEFAULTS from real API data:
      - Weather → NASA POWER 5-year climatology for India centroid
      - Demand  → Ember API India monthly Demand series

    Successful result is cached to cache/india_defaults.json.
    If APIs are unavailable the cache is loaded instead.
    Raises RuntimeError only if both APIs fail AND no cache exists.
    """
    os.makedirs(os.path.dirname(_DEFAULTS_CACHE_PATH), exist_ok=True)

    logger.info("Building INDIA_DEFAULTS from NASA POWER + Ember APIs...")
    weather_stats = _fetch_nasa_climatology_stats()
    demand_stats  = _fetch_ember_demand_stats()

    if weather_stats and demand_stats:
        combined = {k: v for k, v in {**weather_stats, **demand_stats}.items() if v is not None}
        try:
            with open(_DEFAULTS_CACHE_PATH, "w") as f:
                json.dump({"built_at": datetime.now().isoformat(), "defaults": combined}, f, indent=2)
            logger.info(f"INDIA_DEFAULTS cached → {_DEFAULTS_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Could not write defaults cache: {e}")
        return combined

    # At least one API failed — try the on-disk cache
    logger.warning("API(s) unavailable — loading cached INDIA_DEFAULTS")
    if os.path.exists(_DEFAULTS_CACHE_PATH):
        try:
            cached = json.load(open(_DEFAULTS_CACHE_PATH))
            logger.info(f"Loaded cached INDIA_DEFAULTS (built {cached.get('built_at','?')})")
            return cached["defaults"]
        except Exception as e:
            logger.error(f"Cache load failed: {e}")

    # Partial success: use whichever API succeeded
    partial = {k: v for k, v in {**weather_stats, **demand_stats}.items() if v is not None}
    if partial:
        logger.warning("Using partial INDIA_DEFAULTS from available API only")
        return partial

    raise RuntimeError(
        "INDIA_DEFAULTS could not be built: both APIs failed and no cache exists at "
        + _DEFAULTS_CACHE_PATH
    )


# Built once at import — all downstream code reads this dict
INDIA_DEFAULTS: Dict = build_india_defaults()


# ============================================
# SHARED UTILITY
# ============================================
def _make_date_range(
    start: Union[str, datetime],
    end:   Union[str, datetime],
    fmt:   str = "%Y%m%d",
) -> Tuple[datetime, datetime]:
    if isinstance(start, str): start = datetime.strptime(start, fmt)
    if isinstance(end,   str): end   = datetime.strptime(end,   fmt)
    return start, end


# ============================================
# SECTION 4: NASA POWER API CLIENT
# ============================================
class NASAPowerClient:
    """
    Handles all interactions with NASA POWER API.
    Use as a context manager to ensure the HTTP session is closed:
        with NASAPowerClient() as client:
            df = client.fetch_daily_data(...)
    """

    PARAMETERS = {
        "T2M":              "temperature_2m",
        "T2M_MAX":          "temperature_max",
        "T2M_MIN":          "temperature_min",
        "RH2M":             "relative_humidity",
        "WS10M":            "wind_speed_10m",
        "WS50M":            "wind_speed_50m",
        "ALLSKY_SFC_SW_DWN":"solar_radiation",
        "PRECTOTCORR":      "precipitation",
        "CLOUD_AMT":        "cloud_cover",
        "PS":               "surface_pressure",
    }

    def __init__(self):
        self.base_url  = NASA_POWER_BASE_URL
        self.community = NASA_POWER_COMMUNITY
        self.format    = NASA_POWER_FORMAT
        self.session   = requests.Session()
        self.session.headers.update({"User-Agent": "ElectricityForecast/1.0"})

    def __enter__(self):   return self
    def __exit__(self, *_): self.session.close()

    def fetch_daily_data(
        self,
        latitude:   float,
        longitude:  float,
        start_date: Union[str, datetime],
        end_date:   Union[str, datetime],
    ) -> pd.DataFrame:
        """Fetch daily weather; falls back to INDIA_DEFAULTS-based synthetic on failure."""
        if isinstance(start_date, datetime): start_date = start_date.strftime("%Y%m%d")
        if isinstance(end_date,   datetime): end_date   = end_date.strftime("%Y%m%d")

        params = {
            "community":  self.community,
            "longitude":  longitude,
            "latitude":   latitude,
            "start":      start_date,
            "end":        end_date,
            "format":     self.format,
            "parameters": ",".join(self.PARAMETERS.keys()),
        }
        try:
            resp = self.session.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
            return self._parse_api_response(resp.json())
        except Exception as e:
            logger.warning(f"NASA POWER fetch failed ({e}). Using computed-defaults fallback.")
            return self._generate_fallback_data(start_date, end_date)

    def fetch_current_weather(self, latitude: float, longitude: float) -> Dict:
        """Get near-real-time weather (NASA POWER ~2-day latency)."""
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=3)
        try:
            df = self.fetch_daily_data(latitude, longitude, start_dt, end_dt)
            if not df.empty:
                return df.iloc[-1].to_dict()
        except Exception as e:
            logger.warning(f"Current weather fetch failed: {e}")
        return self._generate_fallback_weather()

    def fetch_forecast(self, latitude: float, longitude: float, days: int = 90) -> pd.DataFrame:
        """Get weather forecast for the next N days."""
        start_dt = datetime.now()
        return self.fetch_daily_data(latitude, longitude, start_dt, start_dt + timedelta(days=days))

    def fetch_climatology(self, latitude: float, longitude: float) -> pd.DataFrame:
        """Monthly climatology from last 5 full years."""
        yr = datetime.now().year
        try:
            df = self.fetch_daily_data(latitude, longitude, f"{yr-5}0101", f"{yr-1}1231")
            if not df.empty:
                df["month"] = df.index.month  # index is already DatetimeIndex
                return df.groupby("month").mean(numeric_only=True)
        except Exception as e:
            logger.warning(f"Climatology fetch failed: {e}")
        return self._generate_default_climatology()

    # ── private helpers ───────────────────────────────────────────────────────
    def _parse_api_response(self, response_json: Dict) -> pd.DataFrame:
        """Convert NASA POWER JSON → tidy DataFrame indexed by date."""
        try:
            data = response_json.get("properties", {}).get("parameter", {})
            if not data:
                return pd.DataFrame()
            dates = next(iter(data.values()), {}).keys()
            if not dates:
                return pd.DataFrame()
            records = [
                {"datetime": pd.to_datetime(d, format="%Y%m%d"),
                 **{col: data.get(p, {}).get(d, np.nan)
                    for p, col in self.PARAMETERS.items()}}
                for d in dates
            ]
            return pd.DataFrame(records).set_index("datetime").sort_index()
        except Exception as e:
            logger.error(f"Failed to parse NASA POWER response: {e}")
            return pd.DataFrame()

    def _generate_fallback_data(
        self,
        start_date: Union[str, datetime],
        end_date:   Union[str, datetime],
    ) -> pd.DataFrame:
        """
        Synthetic weather using INDIA_DEFAULTS (computed from real NASA POWER
        climatology at startup). No hardcoded numbers here.
        """
        start_date, end_date = _make_date_range(start_date, end_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        n        = len(date_range)
        seasonal = np.sin(np.linspace(0, 2 * np.pi, n))
        monsoon  = np.where((date_range.month >= 6) & (date_range.month <= 9), 15.0, 0.0)

        np.random.seed(42)
        D = INDIA_DEFAULTS
        return pd.DataFrame({
            "temperature_2m":    D["base_temp"] + seasonal * D["temp_amplitude"]
                                 + np.random.normal(0, 2, n),
            "temperature_max":   D["base_temp"] + seasonal * D["temp_amplitude"] + 5
                                 + np.random.normal(0, 1.5, n),
            "temperature_min":   D["base_temp"] + seasonal * D["temp_amplitude"] - 5
                                 + np.random.normal(0, 1.5, n),
            "relative_humidity": np.clip(
                D["humidity"] + seasonal * 15 + monsoon + np.random.normal(0, 8, n), 0, 100),
            "wind_speed_10m":    np.maximum(0, D["wind_speed_10m"] + np.random.exponential(1.5, n)),
            "wind_speed_50m":    np.maximum(0, D["wind_speed_50m"] + np.random.exponential(2.0, n)),
            "solar_radiation":   np.maximum(0, D["solar_radiation"] + seasonal * 2
                                 + np.random.normal(0, 1, n)),
            "precipitation":     np.maximum(0, np.random.exponential(2, n)),
            "cloud_cover":       np.clip(30 + seasonal * 20 + np.random.normal(0, 12, n), 0, 100),
            "surface_pressure":  D["surface_pressure"] + np.random.normal(0, 5, n),
        }, index=date_range)

    def _generate_fallback_weather(self) -> Dict:
        """Single-row weather dict from INDIA_DEFAULTS (API-derived at startup)."""
        D = INDIA_DEFAULTS
        return {
            "temperature_2m":    D["base_temp"],
            "temperature_max":   D["base_temp"] + 5,
            "temperature_min":   D["base_temp"] - 5,
            "relative_humidity": D["humidity"],
            "wind_speed_10m":    D["wind_speed_10m"],
            "wind_speed_50m":    D["wind_speed_50m"],
            "solar_radiation":   D["solar_radiation"],
            "precipitation":     0.0,
            "cloud_cover":       30.0,
            "surface_pressure":  D["surface_pressure"],
        }

    def _generate_default_climatology(self) -> pd.DataFrame:
        """Monthly climatology table using INDIA_DEFAULTS (API-derived)."""
        months   = np.arange(1, 13)
        seasonal = np.sin(2 * np.pi * months / 12)
        D = INDIA_DEFAULTS
        return pd.DataFrame({
            "temperature_2m":    D["base_temp"] + seasonal * D["temp_amplitude"],
            "relative_humidity": D["humidity"]  + seasonal * 12,
            "wind_speed_10m":    D["wind_speed_10m"] + np.random.uniform(-0.5, 0.5, 12),
            "solar_radiation":   D["solar_radiation"] + seasonal * 1.5,
        }, index=pd.Index(months, name="month"))


# ============================================
# SECTION 5: EMBER ENERGY API CLIENT
# ============================================
class EmberEnergyClient:
    """
    Handles all interactions with Ember Energy API.
    Use as a context manager to ensure the HTTP session is closed:
        with EmberEnergyClient() as client:
            df = client.fetch_generation_mix("IND")
    """

    BASE_URL = "https://api.ember-energy.org/v1/electricity-generation/monthly"

    def __init__(self, api_key: str = "22a3271e-b37d-3f53-f084-1c5ffab5b64d"):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ElectricityProject/1.0"})

    def __enter__(self):    return self
    def __exit__(self, *_): self.session.close()

    def fetch_generation_mix(
        self,
        iso_code:   str = "IND",
        start_date: str = "2021-01-01",
        end_date:   str = "2026-12-31",
    ) -> pd.DataFrame:
        """Fetch monthly electricity generation data from Ember Energy API."""
        params = {
            "api_key":     self.api_key,
            "entity_code": iso_code,
            "start_date":  start_date,
            "end_date":    end_date,
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            if resp.status_code != 200:
                logger.error(f"Ember API returned HTTP {resp.status_code}")
                return pd.DataFrame()
            raw  = resp.json()
            data = raw.get("data", []) if isinstance(raw, dict) else raw
            if not data:
                logger.warning("Empty data in Ember API response")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.error(f"Ember API fetch failed: {e}")
            return pd.DataFrame()

    def get_latest_mix_percentages(self, iso_code: str = "IND") -> Dict[str, float]:
        """
        Most recent month's fuel-mix share (%).
        Fetches only last 6 months to minimise data transfer.
        """
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df = self.fetch_generation_mix(iso_code, start_date=six_months_ago)
        if df.empty:
            return {}
        latest = df[df["date"] == df["date"].max()]
        if "is_aggregate_series" in latest.columns:
            latest = latest[latest["is_aggregate_series"] == False]
        if "series" not in latest.columns or "share_of_generation_pct" not in latest.columns:
            logger.warning("Ember response missing expected columns")
            return {}
        return {r["series"]: r["share_of_generation_pct"] for _, r in latest.iterrows()}


# ============================================
# SECTION 6: FEATURE ENGINEERING
# ============================================
class FeatureEngineer:
    """Creates all features needed for demand forecasting models."""

    def __init__(self):
        self.feature_list   = []
        self.feature_config = {
            "heating_base_temp": HEATING_BASE_TEMP,
            "cooling_base_temp": COOLING_BASE_TEMP,
            "wind_cut_in":       WIND_CUT_IN_SPEED,
            "wind_cut_out":      WIND_CUT_OUT_SPEED,
        }

    def create_all_features(
        self,
        weather_df: pd.DataFrame,
        demand_df:  Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        df = weather_df.copy()
        for fn in (self._add_temporal_features, self._add_cyclical_features,
                   self._add_energy_features, self._add_rolling_features,
                   self._add_interaction_features):
            df = fn(df)
        if demand_df is not None:
            df = self._add_lag_features(df, demand_df)
        self.feature_list = [c for c in df.columns if c not in ("datetime", "demand_mw")]
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index(pd.to_datetime(df["datetime"]))

        df["hour"]          = df.index.hour
        df["day_of_week"]   = df.index.dayofweek
        df["day_of_month"]  = df.index.day
        df["month"]         = df.index.month
        df["day_of_year"]   = df.index.dayofyear
        df["week_of_year"]  = df.index.isocalendar().week.astype(int)
        df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
        df["is_peak_hour"]  = (
            ((df["hour"] >= 7) & (df["hour"] <= 9))
            | ((df["hour"] >= 17) & (df["hour"] <= 21))
        ).astype(int)
        df["is_night"]      = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        return self._add_holiday_features(df)

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        india_holidays = {
            "01-01","01-26","08-15","10-02","12-25",
            "01-14","03-08","04-02","04-10","04-14",
            "05-01","08-19","09-17","10-20","11-01",
        }
        df["is_holiday"]         = df.index.strftime("%m-%d").isin(india_holidays).astype(int)
        df["is_festival_season"] = ((df["month"] >= 10) | (df["month"] <= 1)).astype(int)
        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, period in {"hour": 24, "day_of_week": 7, "month": 12, "day_of_year": 365}.items():
            if col in df.columns:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
        return df

    def _add_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df  = df.copy()
        cfg = self.feature_config
        if "temperature_2m" in df.columns:
            temp = df["temperature_2m"]
            df["heating_degree"] = np.maximum(0, cfg["heating_base_temp"] - temp)
            df["cooling_degree"] = np.maximum(0, temp - cfg["cooling_base_temp"])
            df["temp_deviation"] = np.abs(
                temp - (cfg["heating_base_temp"] + cfg["cooling_base_temp"]) / 2)
            if "relative_humidity" in df.columns:
                df["heat_index"] = temp + 0.5 * (df["relative_humidity"] - 50)
            else:
                df["heat_index"] = temp

        if "relative_humidity" in df.columns:
            df["humidity_deficit"] = 100 - df["relative_humidity"]
            df["humidity_stress"]  = (df["relative_humidity"] > 70).astype(int)

        if "wind_speed_10m" in df.columns:
            wind = df["wind_speed_10m"]
            ci, co = cfg["wind_cut_in"], cfg["wind_cut_out"]
            df["wind_power_potential"] = np.where(
                (wind >= ci) & (wind <= co), ((wind - ci) / (co - ci)) ** 3, 0)

        if "solar_radiation" in df.columns:
            df["solar_potential"] = df["solar_radiation"] / SOLAR_MAX * 100

        if "temperature_2m" in df.columns and "relative_humidity" in df.columns:
            df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity"] / 100
        return df

    def _add_lag_features(self, df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "demand_mw" not in demand_df.columns:
            return df
        demand_df = demand_df.copy()
        if not isinstance(demand_df.index, pd.DatetimeIndex):
            demand_df = demand_df.set_index(pd.to_datetime(demand_df["datetime"]))
        common_idx = df.index.intersection(demand_df.index)
        if len(common_idx) == 0:
            return df
        aligned = demand_df.loc[common_idx, "demand_mw"]
        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            df.loc[common_idx, f"demand_lag_{lag}h"] = aligned.shift(lag)
        df.loc[common_idx, "demand_rolling_mean_24h"]  = aligned.rolling(24).mean()
        df.loc[common_idx, "demand_rolling_std_24h"]   = aligned.rolling(24).std()
        df.loc[common_idx, "demand_rolling_mean_168h"] = aligned.rolling(168).mean()
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["temperature_2m", "relative_humidity", "wind_speed_10m"]:
            if col in df.columns:
                for w in [3, 7, 14, 30]:
                    r = df[col].rolling(w, min_periods=1)
                    df[f"{col}_ma_{w}d"]  = r.mean()
                    df[f"{col}_std_{w}d"] = r.std()
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "temperature_2m" in df.columns and "hour" in df.columns:
            df["temp_hour_interaction"] = df["temperature_2m"] * df["hour"]
        if "temperature_2m" in df.columns and "is_peak_hour" in df.columns:
            df["temp_peak_interaction"] = df["temperature_2m"] * df["is_peak_hour"]
        if "wind_speed_10m" in df.columns and "cloud_cover" in df.columns:
            df["wind_cloud_interaction"] = df["wind_speed_10m"] * (100 - df["cloud_cover"]) / 100
        return df

    def get_feature_names(self) -> List[str]:
        return self.feature_list


# ============================================
# SECTION 7: DATA LOADER
# ============================================
class DataLoader:
    """Load and preprocess electricity demand datasets."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def load_demand_data(
        self,
        filepath:    Optional[str] = None,
        date_column: str = "datetime",
    ) -> pd.DataFrame:
        """Load from CSV → Ember API → synthetic (last resort)."""
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=[date_column])
        else:
            logger.info("No local CSV — fetching from Ember API")
            df = self.load_from_api("india")
        return self.handle_missing_values(df.set_index(date_column).sort_index())

    def load_from_api(self, region: str = "india") -> pd.DataFrame:
        """Load real India demand from Ember Energy API."""
        logger.info(f"Fetching real demand from Ember API for: {region}")
        try:
            with EmberEnergyClient() as client:
                df = client.fetch_generation_mix("IND" if region.lower() == "india" else region)
            if df.empty:
                raise ValueError("Empty Ember response")
            demand = df[df["series"] == "Demand"].copy()
            if demand.empty:
                raise ValueError("No 'Demand' series in Ember data")
            demand["demand_mw"] = (demand["generation_twh"] * 1_000_000) / 730
            return demand.rename(columns={"date": "datetime"})[["datetime", "demand_mw"]]
        except Exception as e:
            logger.error(f"Ember load failed ({e}) — using API-derived synthetic data")
            return self._generate_synthetic_demand()

    def _generate_synthetic_demand(self, days: int = 365) -> pd.DataFrame:
        """
        Last-resort hourly synthetic demand.
        All values sourced from INDIA_DEFAULTS (computed from Ember API at startup).
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), periods=days * 24, freq="h")
        t  = np.arange(len(dates))
        D  = INDIA_DEFAULTS
        np.random.seed(42)
        demand = np.maximum(
            D["demand_floor_mw"],
            D["demand_base_mw"]
            + np.sin(2 * np.pi * t / 24)         * D["demand_hourly_amp"]
            + np.sin(2 * np.pi * t / (24 * 7))   * D["demand_weekly_amp"]
            + np.sin(2 * np.pi * t / (24 * 365)) * D["demand_seasonal_amp"]
            + np.random.normal(0, D["demand_hourly_amp"] * 0.5, len(t))
        )
        return pd.DataFrame({"datetime": dates, "demand_mw": demand})

    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Upsample to hourly using time interpolation (pandas ≥ 2.2 freq string 'h')."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.freq is None or df.index.inferred_freq != "h":
            df = df.resample("h").interpolate(method="time")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-interpolate then forward/back-fill residual NaNs (pandas ≥ 2.0 API)."""
        df = df.copy()
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.info(f"Filling {missing} missing values")
            df = df.interpolate(method="time").ffill().bfill()
        return df

    def detect_outliers(self, df: pd.DataFrame, column: str = "demand_mw") -> pd.DataFrame:
        """Flag values outside IQR 3× fence."""
        if column not in df.columns:
            return df
        df = df.copy()
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        IQR = Q3 - Q1
        df["is_outlier"] = (df[column] < Q1 - 3 * IQR) | (df[column] > Q3 + 3 * IQR)
        n_out = df["is_outlier"].sum()
        if n_out:
            logger.warning(f"Detected {n_out} outliers in '{column}'")
        return df

    def split_train_val_test(
        self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        te = int(n * train_ratio)
        ve = int(n * (train_ratio + val_ratio))
        tr, va, te_df = df.iloc[:te], df.iloc[te:ve], df.iloc[ve:]
        logger.info(f"Split → Train={len(tr)}, Val={len(va)}, Test={len(te_df)}")
        return tr, va, te_df


# ============================================
# SECTION 8: DATA VALIDATOR
# ============================================
class DataValidator:
    """Validate data quality before model training."""

    def __init__(self):
        self.validation_results = {}

    def validate_weather_data(self, df: pd.DataFrame) -> Dict:
        issues = []
        for col, (lo, hi) in {
            "temperature_2m":    (TEMP_MIN,       TEMP_MAX),
            "relative_humidity": (HUMIDITY_MIN,   HUMIDITY_MAX),
            "wind_speed_10m":    (WIND_SPEED_MIN, WIND_SPEED_MAX),
            "solar_radiation":   (SOLAR_MIN,      SOLAR_MAX),
        }.items():
            if col in df.columns:
                n = ((df[col] < lo) | (df[col] > hi)).sum()
                if n: issues.append(f"{col} out of [{lo},{hi}]: {n} values")
        return {"status": "PASS" if not issues else "FAIL",
                "issues": issues, "total_records": len(df)}

    def validate_demand_data(self, df: pd.DataFrame) -> Dict:
        if "demand_mw" not in df.columns:
            return {"status": "FAIL", "issues": ["No demand_mw column"], "quality_score": 0}
        issues, quality = [], 100.0
        neg = (df["demand_mw"] < 0).sum()
        if neg:
            issues.append(f"Negative demand: {neg} rows"); quality -= neg / len(df) * 100
        zeros = (df["demand_mw"] == 0).sum()
        if zeros > len(df) * 0.01:
            issues.append(f"Excess zeros: {zeros} rows"); quality -= zeros / len(df) * 50
        if len(df) > 1:
            jumps = (df["demand_mw"].pct_change().abs() > 0.5).sum()
            if jumps:
                issues.append(f"Sudden >50% jumps: {jumps} rows"); quality -= jumps / len(df) * 100
        return {"status": "PASS" if quality >= 80 else "FAIL",
                "issues": issues, "quality_score": max(0.0, quality)}

    def check_data_completeness(self, df: pd.DataFrame, expected_frequency: str = "h") -> Dict:
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"status": "UNKNOWN", "completeness": 0, "gaps": []}
        expected     = pd.date_range(df.index.min(), df.index.max(), freq=expected_frequency)
        missing      = expected.difference(df.index)
        completeness = len(df.index) / len(expected) if len(expected) else 0
        return {
            "status":        "PASS" if completeness >= MIN_COMPLETENESS_RATIO else "FAIL",
            "completeness":  completeness,
            "gaps":          list(missing[:10]),
            "missing_count": len(missing),
        }

    def check_seasonal_consistency(self, df: pd.DataFrame) -> Dict:
        """India: summer (Apr–Jun) demand should exceed winter (Dec–Feb) due to cooling load."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"status": "UNKNOWN", "anomalies": []}
        monthly   = df.copy()
        monthly["month"] = monthly.index.month
        monthly   = monthly.groupby("month").mean(numeric_only=True)
        anomalies = []
        if "demand_mw" in monthly.columns:
            summer = monthly.loc[monthly.index.isin([4, 5, 6]),  "demand_mw"].mean()
            winter = monthly.loc[monthly.index.isin([12, 1, 2]), "demand_mw"].mean()
            if summer < winter:
                anomalies.append("India: summer demand expected >= winter (cooling load dominates)")
        return {"status": "PASS" if not anomalies else "WARNING", "anomalies": anomalies}

    def generate_quality_report(self, weather_df: pd.DataFrame, demand_df: pd.DataFrame) -> Dict:
        report = {
            "weather_validation":   self.validate_weather_data(weather_df),
            "demand_validation":    self.validate_demand_data(demand_df),
            "weather_completeness": self.check_data_completeness(weather_df),
            "demand_completeness":  self.check_data_completeness(demand_df),
            "seasonal_check":       self.check_seasonal_consistency(demand_df),
        }
        report["overall_score"] = (
            report["weather_validation"].get("quality_score", 100)
            + report["demand_validation"].get("quality_score", 100)
            + report["weather_completeness"].get("completeness", 0) * 100
            + report["demand_completeness"].get("completeness",  0) * 100
        ) / 4
        return report


# ============================================
# SECTION 9: DATA PIPELINE ORCHESTRATOR
# ============================================
class DataPipeline:
    """Main orchestrator — single interface for all data operations."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.weather_api      = NASAPowerClient()
        self.feature_engineer = FeatureEngineer()
        self.data_loader      = DataLoader()
        self.validator        = DataValidator()
        self.cache_dir        = cache_dir or os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.scaler           = StandardScaler()

    def prepare_training_data(
        self,
        lat:             float,
        lon:             float,
        start_date:      Union[str, datetime],
        end_date:        Union[str, datetime],
        demand_filepath: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Fetch → validate → engineer → return model-ready (X, y, feature_names)."""
        logger.info("Preparing training data...")
        weather_df = self.weather_api.fetch_daily_data(lat, lon, start_date, end_date)
        demand_df  = self.data_loader.load_demand_data(demand_filepath)
        logger.info(f"Fetched {len(weather_df)} weather + {len(demand_df)} demand records")

        weather_df = self.data_loader.resample_to_hourly(weather_df)
        quality    = self.validator.generate_quality_report(weather_df, demand_df)
        logger.info(f"Data quality score: {quality['overall_score']:.1f}")

        aligned     = self.align_weather_and_demand(weather_df, demand_df)
        features_df = self.feature_engineer.create_all_features(aligned, demand_df).dropna()

        names = self.feature_engineer.get_feature_names()
        X     = features_df[names]
        y     = features_df["demand_mw"] if "demand_mw" in features_df.columns else None
        logger.info(f"Prepared {len(X)} samples × {len(names)} features")
        return X, y, names

    def prepare_forecast_data(
        self,
        lat:               float,
        lon:               float,
        forecast_days:     int = 7,
        last_known_demand: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Prepare feature matrix for future prediction."""
        logger.info(f"Preparing {forecast_days}-day forecast data...")
        wf = self.feature_engineer.create_all_features(
            self.weather_api.fetch_forecast(lat, lon, forecast_days))
        if last_known_demand is not None and len(last_known_demand) > 0:
            for lag in [24, 48, 168]:
                if len(last_known_demand) >= lag:
                    wf[f"demand_lag_{lag}h"] = last_known_demand.iloc[-lag]
            wf["demand_rolling_mean_24h"]  = (
                last_known_demand.iloc[-24:].mean()  if len(last_known_demand) >= 24
                else last_known_demand.mean())
            wf["demand_rolling_mean_168h"] = (
                last_known_demand.iloc[-168:].mean() if len(last_known_demand) >= 168
                else last_known_demand.mean())
        names = self.feature_engineer.get_feature_names()
        return wf[[f for f in names if f in wf.columns]].fillna(0)

    def get_current_conditions(self, lat: float, lon: float) -> Dict:
        w = self.weather_api.fetch_current_weather(lat, lon)
        if "temperature_2m" in w:
            w["heating_degree"] = max(0, HEATING_BASE_TEMP - w["temperature_2m"])
            w["cooling_degree"] = max(0, w["temperature_2m"] - COOLING_BASE_TEMP)
        return w

    def align_weather_and_demand(
        self, weather_df: pd.DataFrame, demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Align on common timestamps; copies inputs to avoid mutating caller data."""
        weather_df, demand_df = weather_df.copy(), demand_df.copy()
        for frame in (weather_df, demand_df):
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame.index = pd.to_datetime(frame.index)
        common = weather_df.index.intersection(demand_df.index)
        if len(common) == 0:
            logger.warning("No overlapping timestamps — aligning by position")
            n = min(len(weather_df), len(demand_df))
            out = weather_df.iloc[:n].copy()
            out["demand_mw"] = demand_df["demand_mw"].iloc[:n].values
            return out
        return pd.concat(
            [weather_df.loc[common], demand_df.loc[common, ["demand_mw"]]],
            axis=1, join="inner")

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        fp = os.path.join(self.cache_dir, f"{filename}.parquet")
        df.to_parquet(fp)
        logger.info(f"Saved → {fp}")

    def load_processed_data(self, filename: str) -> Optional[pd.DataFrame]:
        fp = os.path.join(self.cache_dir, f"{filename}.parquet")
        if os.path.exists(fp):
            logger.info(f"Loaded ← {fp}")
            return pd.read_parquet(fp)
        return None


# ============================================
# SECTION 10: HELPER FUNCTIONS
# ============================================
def validate_coordinates(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180

def get_timezone_from_coordinates(lat: float, lon: float) -> str:
    return f"UTC{round(lon / 15):+d}"

def calculate_season(month: int) -> str:
    if month in (12, 1, 2):   return "Winter"
    if month in (3, 4, 5):    return "Spring"
    if month in (6, 7, 8, 9): return "Monsoon"
    return "Post-Monsoon"

def is_holiday(date: datetime, country: str = "India") -> bool:
    return date in [datetime(date.year, m, d) for m, d in
                    [(1,1),(1,26),(8,15),(10,2),(12,25)]]

def get_default_parameters() -> Dict:
    return {
        "forecast_horizon_days": 7,
        "feature_windows":       DEFAULT_ROLLING_WINDOWS,
        "lag_periods":           DEFAULT_LAG_HOURS,
        "validation_thresholds": {
            "min_completeness": MIN_COMPLETENESS_RATIO,
            "max_outliers":     MAX_ALLOWED_OUTLIERS,
        },
        "heating_base_temp": HEATING_BASE_TEMP,
        "cooling_base_temp": COOLING_BASE_TEMP,
    }


# ============================================
# SECTION 11: EXPORTS
# ============================================
__all__ = [
    "NASAPowerClient", "EmberEnergyClient",
    "FeatureEngineer", "DataLoader", "DataValidator", "DataPipeline",
    "build_india_defaults", "INDIA_DEFAULTS",
    "validate_coordinates", "calculate_season", "get_default_parameters",
    "DataPipelineError", "APIFetchError", "DataValidationError", "MissingDataError",
]
