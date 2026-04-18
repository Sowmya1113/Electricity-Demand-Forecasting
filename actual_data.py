# ============================================
# actual.py
# Fetch REAL hourly data for India with proper disaggregation
# Uses:
#   - Hourly weather from NASA POWER (multi‑city weighted)
#   - Monthly demand from Ember Energy
#   - Realistic daily load shape to disaggregate to hourly
#   - Synthetic fallback with holidays & heatwaves
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline import EmberEnergyClient, NASAPowerClient

# ============================================
# 1. Realistic daily load profile for India (normalized to 1.0)
# Based on POSOCO historical data: peaks at 9am and 7pm, trough at 3am
# ============================================
def get_india_daily_load_profile() -> np.ndarray:
    """Returns array of 24 hourly multipliers (mean = 1.0)"""
    hour = np.arange(24)
    # Morning peak ~9am, Evening peak ~7pm
    morning = np.exp(-((hour - 9) ** 2) / 50)
    evening = np.exp(-((hour - 19) ** 2) / 50)
    profile = 0.5 + 0.3 * (morning + evening)  # baseline + peaks
    # Normalize so mean = 1
    profile = profile / profile.mean()
    return profile

# ============================================
# 2. Disaggregate monthly demand to hourly using the load profile
# ============================================
def disaggregate_monthly_to_hourly(monthly_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Input: monthly_demand with columns ['date', 'demand_twh'] (date is first of month)
    Output: hourly DataFrame with 'datetime' and 'demand_mw'
    """
    profile = get_india_daily_load_profile()
    hourly_records = []
    
    for _, row in monthly_demand.iterrows():
        month_start = row['date']
        # Days in month
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year+1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month+1)
        days_in_month = (next_month - month_start).days
        
        # Total monthly energy in MWh (TWh -> MWh)
        total_mwh = row['demand_twh'] * 1_000_000
        
        # Create all hours in this month
        hours_in_month = days_in_month * 24
        # Repeat the daily profile to fill the month
        hourly_profile = np.tile(profile, days_in_month)
        # Scale so sum of profile equals total_mwh
        scale = total_mwh / hourly_profile.sum()
        hourly_demand_mw = hourly_profile * scale  # MW for each hour
        
        # Generate datetimes
        start_dt = month_start
        for i, mw in enumerate(hourly_demand_mw):
            dt = start_dt + timedelta(hours=i)
            hourly_records.append({'datetime': dt, 'demand_mw': mw})
    
    df_hourly = pd.DataFrame(hourly_records)
    df_hourly = df_hourly.set_index('datetime').sort_index()
    return df_hourly

# ============================================
# 3. Fetch hourly weather (multi‑city weighted)
# ============================================
def fetch_weighted_hourly_weather(
    start_date: datetime, 
    end_date: datetime,
    cities: List[Dict]
) -> pd.DataFrame:
    """
    cities: list of dicts with 'lat', 'lon', 'weight'
    Returns hourly DataFrame with columns: temperature_2m, relative_humidity, wind_speed_10m
    """
    weather_client = NASAPowerClient()
    all_city_dfs = []
    
    for city in cities:
        print(f"  Fetching weather for {city['name']}...")
        df_city = weather_client.fetch_hourly_data(
            city['lat'], city['lon'], start_date, end_date
        )
        if not df_city.empty:
            df_city = df_city.reset_index()
            df_city['weight'] = city['weight']
            all_city_dfs.append(df_city)
    
    if not all_city_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_city_dfs, ignore_index=True)
    # Weighted average by datetime
    weighted = combined.groupby('datetime').apply(
        lambda x: pd.Series({
            'temperature_2m': (x['temperature_2m'] * x['weight']).sum(),
            'relative_humidity': (x['relative_humidity'] * x['weight']).sum(),
            'wind_speed_10m': (x['wind_speed_10m'] * x['weight']).sum(),
        })
    ).reset_index()
    
    weighted = weighted.set_index('datetime').sort_index()
    return weighted

# ============================================
# 4. Synthetic fallback (improved with holidays and heatwaves)
# ============================================
def generate_synthetic_fallback(days: int = 1000) -> pd.DataFrame:
    """Generate realistic synthetic hourly data when APIs fail"""
    print("Generating synthetic data with realistic patterns...")
    end_date = datetime.now() - timedelta(days=2)  # avoid partial day
    start_date = end_date - timedelta(days=days)
    full_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    n = len(full_dates)
    
    # Base demand (MW) - India approx 120 GW average
    base_demand = 120000
    
    # Annual cycle (seasonal)
    annual = np.sin(2 * np.pi * np.arange(n) / (24 * 365))
    seasonal_demand = base_demand * (1 + 0.15 * annual)
    
    # Weekly cycle (weekends lower)
    day_of_week = full_dates.dayofweek
    weekly_pattern = np.where(day_of_week >= 5, 0.85, 1.0)
    
    # Daily cycle (using the same realistic profile)
    profile = get_india_daily_load_profile()
    hour = full_dates.hour
    daily_pattern = profile[hour]
    
    # Combine
    demand = seasonal_demand * weekly_pattern * daily_pattern
    noise = np.random.normal(0, 0.03 * base_demand, n)
    demand += noise
    demand = np.maximum(demand, 50000)
    
    # Weather
    temp_seasonal = 25 + 10 * annual
    temp_daily = 5 * np.sin(2 * np.pi * (hour - 14) / 24)
    temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, n)
    humidity = 80 - 0.5 * (temperature - 20) + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 20, 100)
    wind_speed = 5 + 3 * np.sin(2 * np.pi * (hour - 13) / 24) + np.random.exponential(2, n)
    wind_speed = np.clip(wind_speed, 0, 30)
    
    # Add holidays (Diwali, Holi, etc.)
    for year in range(start_date.year, end_date.year + 1):
        # Diwali (Oct/Nov) - demand drop 25%
        diwali_approx = datetime(year, 10, 20)  # approximate
        for offset in [-1, 0, 1]:
            holiday = diwali_approx + timedelta(days=offset)
            mask = (full_dates.date == holiday.date())
            demand[mask] *= 0.75
        # Holi (March) - demand drop 15%
        holi_approx = datetime(year, 3, 10)
        mask = (full_dates.date == holi_approx.date())
        demand[mask] *= 0.85
    
    # Heatwaves (May-June) -> demand spike 15%
    heatwave_months = (full_dates.month == 5) | (full_dates.month == 6)
    heatwave_temp = temperature > 40
    demand[heatwave_months & heatwave_temp] *= 1.15
    
    df = pd.DataFrame({
        'datetime': full_dates,
        'demand_mw': demand,
        'temperature_2m': temperature,
        'relative_humidity': humidity,
        'wind_speed_10m': wind_speed,
    })
    df = df.set_index('datetime')
    return df

# ============================================
# 5. Main function: fetch actual data with fallback
# ============================================
def fetch_actual_data(days: int = 1000) -> pd.DataFrame:
    """
    Returns hourly DataFrame with columns:
        demand_mw, temperature_2m, relative_humidity, wind_speed_10m
    """
    print("Fetching actual historical demand from Ember Energy API...")
    
    # Date range (stop 3 days early due to NASA latency)
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=days)
    
    # 1. Get monthly demand from Ember
    ember = EmberEnergyClient()
    monthly = ember.get_monthly_demand("IND")
    if monthly.empty:
        raise ValueError("Ember API returned empty demand data")
    
    # Filter to requested date range
    monthly = monthly[(monthly['date'] >= start_date) & (monthly['date'] <= end_date)]
    if monthly.empty:
        raise ValueError("No demand data in date range")
    
    # 2. Disaggregate to hourly
    print("Disaggregating monthly demand to hourly using daily load profile...")
    demand_hourly = disaggregate_monthly_to_hourly(monthly)
    # Clip to exact date range
    demand_hourly = demand_hourly[(demand_hourly.index >= start_date) & (demand_hourly.index <= end_date)]
    
    # 3. Fetch weighted hourly weather
    print("Fetching hourly weather from NASA POWER for major cities...")
    cities = [
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "weight": 0.30},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "weight": 0.20},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "weight": 0.15},
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "weight": 0.15},
        {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "weight": 0.20},
    ]
    weather_hourly = fetch_weighted_hourly_weather(start_date, end_date, cities)
    if weather_hourly.empty:
        raise ValueError("Weather fetch failed for all cities")
    
    # 4. Merge demand and weather on exact datetime
    combined = demand_hourly.join(weather_hourly, how='inner')
    
    # 5. Basic cleaning
    combined = combined.dropna()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    print(f"Success: {len(combined)} hourly records from {combined.index.min()} to {combined.index.max()}")
    return combined

# ============================================
# 6. Run if script executed directly
# ============================================
if __name__ == "__main__":
    try:
        df = fetch_actual_data(days=1000)
    except Exception as e:
        print(f"ERROR fetching real data: {e}")
        print("Falling back to synthetic data generation...")
        df = generate_synthetic_fallback(days=1000)
    
    if df is not None and not df.empty:
        df.to_csv('actual_demand.csv')
        print(f"\n✅ Saved {len(df):,} hourly records to actual_demand.csv")
        print("\n📊 Data Statistics:")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Mean demand: {df['demand_mw'].mean():.0f} MW")
        print(f"   Peak demand: {df['demand_mw'].max():.0f} MW")
        print(f"   Mean temperature: {df['temperature_2m'].mean():.1f}°C")
        
        # Also save a sample
        df.head(1000).to_csv('demand_sample.csv')
        print("📊 Sample (first 1000 rows) saved to demand_sample.csv")
    else:
        print("❌ Failed to generate any data")
