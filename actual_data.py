
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline import EmberEnergyClient, NASAPowerClient

def fetch_actual_data(days=1000):
    """
    Fetch REAL historical demand data for India from Ember API
    instead of generating synthetic sine waves.
    """
    print("Fetching actual historical demand from Ember Energy API...")
    try:
        client = EmberEnergyClient()
        # Fetch records starting from 2021-01-01
        df_real = client.fetch_generation_mix("IND", start_date="2021-01-01")
        
        if df_real.empty:
            print("Error: Could not retrieve data from API.")
            return pd.DataFrame()

        # Filter for Demand series
        demand_data = df_real[df_real["series"] == "Demand"].copy()
        
        if demand_data.empty:
            print("Error: No 'Demand' series found.")
            return pd.DataFrame()

        # Convert Monthly Generation (TWh) to Average Power (MW)
        # Average power in month = (TWh * 1e6 MWh) / 730 hours
        demand_data["demand_mw"] = (demand_data["generation_twh"] * 1000000) / 730
        
        # Sort and clean
        demand_data = demand_data.sort_values("date")
        
        # Resample monthly data to hourly to keep compatibility with models
        # We start from the first month and go to the end
        start_date = demand_data["date"].iloc[0]
        end_date = demand_data["date"].iloc[-1] + pd.DateOffset(months=1)
        
        full_dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Create a template dataframe
        full_df = pd.DataFrame({"datetime": full_dates})
        
        # Create temporary date column for merging
        full_df["merge_date"] = full_df["datetime"].dt.floor("D")
        
        # Merge
        full_df = full_df.merge(
            demand_data[["date", "demand_mw"]], 
            left_on="merge_date", 
            right_on="date", 
            how="left"
        )
        
        # Interpolate the monthly 'steps' into smooth transitions
        full_df["demand_mw"] = full_df["demand_mw"].interpolate(method="linear")

        # Fetch real weather data from NASA POWER API for India (Delhi coordinates as proxy)
        print("Fetching actual weather data from NASA POWER API...")
        weather_client = NASAPowerClient()
        weather_df = weather_client.fetch_daily_data(
            latitude=28.6139, 
            longitude=77.2090, 
            start_date=start_date, 
            end_date=end_date
        )
        
        if not weather_df.empty:
            weather_df = weather_df.reset_index()
            # Merge weather data based on the daily date
            full_df = full_df.merge(
                weather_df[["datetime", "temperature_2m", "relative_humidity", "wind_speed_10m"]],
                left_on="merge_date",
                right_on="datetime",
                how="left",
                suffixes=("", "_weather")
            )
            # Rename columns to match expected format
            full_df = full_df.rename(columns={
                "temperature_2m": "temperature",
                "relative_humidity": "humidity",
                "wind_speed_10m": "wind_speed"
            })
            
            # Forward fill any missing daily weather data across the hours
            for col in ["temperature", "humidity", "wind_speed"]:
                full_df[col] = full_df[col].ffill()
        else:
            print("Error: Could not fetch weather data. Leaving weather columns empty.")
            full_df["temperature"] = np.nan
            full_df["humidity"] = np.nan
            full_df["wind_speed"] = np.nan

        full_df["hour"] = full_df["datetime"].dt.hour
        full_df["day_of_week"] = full_df["datetime"].dt.dayofweek

        # Clean up column names when returning
        return full_df[["datetime", "demand_mw", "temperature", "humidity", "wind_speed", "hour", "day_of_week"]]
    except Exception as e:
        print(f"FAILED to fetch real data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_actual_data()
    if not df.empty:
        df.to_csv('actual_demand.csv', index=False)
        print(f"Successfully saved {len(df)} real data records to actual_demand.csv")
    else:
        print("Data generation failed.")
        # We start from the first month and go to the end
        start_date = demand_data["date"].iloc[0]
        end_date = demand_data["date"].iloc[-1] + pd.DateOffset(months=1)
        
        full_dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Create a template dataframe
        full_df = pd.DataFrame({"datetime": full_dates})
        
        # Create temporary date column for merging
        full_df["merge_date"] = full_df["datetime"].dt.floor("D")
        
        # Merge
        full_df = full_df.merge(
            demand_data[["date", "demand_mw"]], 
            left_on="merge_date", 
            right_on="date", 
            how="left"
        )
        
        # Interpolate the monthly 'steps' into smooth transitions
        full_df["demand_mw"] = full_df["demand_mw"].interpolate(method="linear")
        
        # Add basic hourly variation (real data variation simulation)
        # Peak demand at 9 AM and 7 PM
        hours = full_df["datetime"].dt.hour
        hourly_profile = 1.0 + 0.15 * np.sin(2 * np.pi * (hours - 6) / 24)
        full_df["demand_mw"] *= hourly_profile

        # Add real-world weather placeholders (Project currently expects these columns)
        # In a full pipeline, these would be fetched from NASA POWER
        n = len(full_df)
        full_df["temperature"] = 28 + 5 * np.sin(2 * np.pi * np.arange(n) / (24 * 365))
        full_df["humidity"] = 65 + 10 * np.sin(2 * np.pi * np.arange(n) / (24 * 365 + 50))
        full_df["wind_speed"] = 12.0
        full_df["hour"] = hours
        full_df["day_of_week"] = full_df["datetime"].dt.dayofweek

        return full_df[["datetime", "demand_mw", "temperature", "humidity", "wind_speed", "hour", "day_of_week"]]
    except Exception as e:
        print(f"FAILED to fetch real data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_actual_data()
    if not df.empty:
        df.to_csv('synthetic_demand.csv', index=False)
        print(f"Successfully saved {len(df)} real data records to synthetic_demand.csv")
    else:
        print("Data generation failed.")
