# ============================================
# utils.py
# PURPOSE: Utility functions for electricity demand forecasting system
# - Energy mix calculation (Solar/Wind/Hydro/Thermal percentages)
# - Behavior recommendations based on weather
# - Economic and carbon impact analysis
# - Visualization helpers for Streamlit
# - Report generation utilities
# - File export handlers (CSV, PDF, PNG)
# ============================================

# ============================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================
import os
import io
import json
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")

# ============================================
# SECTION 10: CONFIGURATION CONSTANTS
# ============================================
MAX_SOLAR_PERCENT = 40
MAX_WIND_PERCENT = 35
MAX_HYDRO_PERCENT = 50
MIN_THERMAL_PERCENT = 20

SOLAR_EFFICIENCY = 0.85
WIND_EFFICIENCY = 0.40
HYDRO_EFFICIENCY = 0.90
THERMAL_EFFICIENCY = 0.35

HEAT_WAVE_TEMP = 35
EXTREME_HEAT_TEMP = 40
COLD_WAVE_TEMP = 10
GOOD_WIND_LOW = 3
GOOD_WIND_HIGH = 12
HIGH_WIND_ALERT = 15
CLOUDY_THRESHOLD = 70
VERY_CLOUDY_THRESHOLD = 85

MORNING_PEAK_START = 7
MORNING_PEAK_END = 10
EVENING_PEAK_START = 18
EVENING_PEAK_END = 21

RESIDENTIAL_PRICE_PER_KWH = 8.5
COMMERCIAL_PRICE_PER_KWH = 12.0
INDUSTRIAL_PRICE_PER_KWH = 7.5
PEAK_PRICE_MULTIPLIER = 1.5
OFF_PEAK_PRICE_MULTIPLIER = 0.7

GRID_CARBON_INTENSITY = 0.82
SOLAR_CARBON_INTENSITY = 0.05
WIND_CARBON_INTENSITY = 0.01
HYDRO_CARBON_INTENSITY = 0.02
THERMAL_CARBON_INTENSITY = 0.95

TREES_PER_KG_CO2_PER_YEAR = 1 / 21

COLOR_SOLAR = "#FFD700"
COLOR_WIND = "#87CEEB"
COLOR_HYDRO = "#00008B"
COLOR_THERMAL = "#FF4444"
COLOR_DEMAND = "#1f77b4"
COLOR_FORECAST = "#ff7f0e"

CHART_HEIGHT = 500
CHART_WIDTH = 800


# ============================================
# SECTION 2: ENERGY MIX CALCULATOR
# ============================================
class EnergyMixCalculator:
    """
    Calculate electricity generation mix based on weather and water availability
    PURPOSE: Show what powers the grid (Solar, Wind, Hydro, Thermal)
    """

    def __init__(self):
        self.max_solar_pct = MAX_SOLAR_PERCENT
        self.max_wind_pct = MAX_WIND_PERCENT
        self.max_hydro_pct = MAX_HYDRO_PERCENT
        self.min_thermal_pct = MIN_THERMAL_PERCENT

        self.solar_efficiency = SOLAR_EFFICIENCY
        self.wind_efficiency = WIND_EFFICIENCY
        self.hydro_efficiency = HYDRO_EFFICIENCY

        self.default_mix = {
            "UK": {"solar": 0.08, "wind": 0.25, "hydro": 0.02, "thermal": 0.65},
            "US": {"solar": 0.08, "wind": 0.10, "hydro": 0.02, "thermal": 0.80},
            "EU": {"solar": 0.12, "wind": 0.22, "hydro": 0.10, "thermal": 0.56},
            "India": {"solar": 0.15, "wind": 0.08, "hydro": 0.12, "thermal": 0.65},
        }

    def calculate_mix(
        self, weather_data: Dict, water_availability_percent: float = 75
    ) -> Dict:
        """
        Calculate generation percentages for next period
        """
        cloud_cover = weather_data.get("cloud_cover", 50)
        wind_speed = weather_data.get("wind_speed_10m", 5)
        solar_radiation = weather_data.get("solar_radiation", 5)
        hour = weather_data.get("hour", 12)
        month = weather_data.get("month", 6)

        solar = self._calculate_solar_factor(cloud_cover, solar_radiation, hour, month)
        wind = self._calculate_wind_factor(wind_speed)
        hydro = self._calculate_hydro_factor(water_availability_percent)

        thermal = 100 - (solar + wind + hydro)
        thermal = max(thermal, self.min_thermal_pct)

        total = solar + wind + hydro + thermal
        if total != 100:
            scale = 100 / total
            solar *= scale
            wind *= scale
            hydro *= scale
            thermal *= scale

        return {
            "solar": round(solar, 2),
            "wind": round(wind, 2),
            "hydro": round(hydro, 2),
            "thermal": round(thermal, 2),
            "renewable": round(solar + wind + hydro, 2),
            "explanation": self.explain_mix_calculation(
                weather_data, water_availability_percent
            ),
        }

    def _calculate_solar_factor(
        self, cloud_cover: float, solar_radiation: float, hour: int, month: int
    ) -> float:
        """Calculate solar generation potential"""
        base = 25

        cloud_factor = max(0, 1 - cloud_cover / 100)

        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)

        if 6 <= hour <= 18:
            daylight_factor = 1
        else:
            daylight_factor = 0

        solar = (
            base
            * cloud_factor
            * seasonal_factor
            * daylight_factor
            * self.solar_efficiency
        )

        return min(solar, self.max_solar_pct)

    def _calculate_wind_factor(self, wind_speed: float) -> float:
        """Calculate wind generation potential using power curve"""
        if wind_speed < 3:
            wind = max(0, wind_speed * 5)
        elif wind_speed <= 12:
            wind = min(35, 15 + (wind_speed - 3) * 2)
        else:
            wind = max(0, 35 - (wind_speed - 12) * 3)

        wind = wind * self.wind_efficiency

        return min(wind, self.max_wind_pct)

    def _calculate_hydro_factor(self, water_availability: float) -> float:
        """Calculate hydro potential"""
        hydro = water_availability * 0.5 * self.hydro_efficiency

        return min(hydro, self.max_hydro_pct)

    def _apply_grid_constraints(self, mix_dict: Dict) -> Dict:
        """Apply grid stability limits"""
        mix_dict["solar"] = min(mix_dict["solar"], self.max_solar_pct)
        mix_dict["wind"] = min(mix_dict["wind"], self.max_wind_pct)
        mix_dict["hydro"] = min(mix_dict["hydro"], self.max_hydro_pct)

        total = sum(mix_dict.values())
        if mix_dict.get("thermal", 0) < self.min_thermal_pct:
            mix_dict["thermal"] = self.min_thermal_pct

        return mix_dict

    def calculate_daily_mix(
        self, weather_forecast_df: pd.DataFrame, water_availability: float = 75
    ) -> pd.DataFrame:
        """Calculate mix for each day of forecast period"""
        results = []

        for idx, row in weather_forecast_df.iterrows():
            weather_data = {
                "cloud_cover": row.get("cloud_cover", 50),
                "wind_speed_10m": row.get("wind_speed_10m", 5),
                "solar_radiation": row.get("solar_radiation", 5),
                "hour": row.get("hour", 12),
                "month": row.get("month", 6),
            }

            mix = self.calculate_mix(weather_data, water_availability)
            mix["datetime"] = idx
            results.append(mix)

        return pd.DataFrame(results)

    def get_generation_mw(self, total_demand_mw: float, mix_percentages: Dict) -> Dict:
        """Convert percentages to actual MW generation"""
        return {
            "solar_mw": total_demand_mw * mix_percentages.get("solar", 0) / 100,
            "wind_mw": total_demand_mw * mix_percentages.get("wind", 0) / 100,
            "hydro_mw": total_demand_mw * mix_percentages.get("hydro", 0) / 100,
            "thermal_mw": total_demand_mw * mix_percentages.get("thermal", 0) / 100,
            "total_mw": total_demand_mw,
        }

    def explain_mix_calculation(
        self, weather_data: Dict, water_availability: float
    ) -> str:
        """Generate human-readable explanation"""
        cloud_cover = weather_data.get("cloud_cover", 50)
        wind_speed = weather_data.get("wind_speed_10m", 5)

        explanations = []

        if cloud_cover < 30:
            explanations.append(
                f"Solar at high potential due to clear skies ({cloud_cover}% clouds)."
            )
        elif cloud_cover > 70:
            explanations.append(f"Solar reduced due to {cloud_cover}% cloud cover.")

        if 3 <= wind_speed <= 12:
            explanations.append(f"Wind at optimal range ({wind_speed} m/s).")
        elif wind_speed < 3:
            explanations.append(
                f"Wind low ({wind_speed} m/s) - below optimal for turbines."
            )

        if water_availability < 40:
            explanations.append(
                f"Low water availability ({water_availability}%) limiting hydro generation."
            )

        return (
            " ".join(explanations)
            if explanations
            else "Normal grid conditions expected."
        )

    def get_source_color(self, source_name: str) -> str:
        """Return color code for each energy source"""
        colors = {
            "solar": COLOR_SOLAR,
            "wind": COLOR_WIND,
            "hydro": COLOR_HYDRO,
            "thermal": COLOR_THERMAL,
            "nuclear": "#FFA500",
            "imports": "#808080",
        }
        return colors.get(source_name.lower(), "#000000")

    def get_source_icon(self, source_name: str) -> str:
        """Return emoji/icon for each source"""
        icons = {
            "solar": "☀️",
            "wind": "💨",
            "hydro": "💧",
            "thermal": "🔥",
            "nuclear": "☢️",
            "renewable": "🌱",
        }
        return icons.get(source_name.lower(), "⚡")

    # Backwards compatibility aliases
    def get_energy_mix(self, region: str = "India") -> Dict:
        """Alias for backwards compatibility"""
        weather_data = {
            "cloud_cover": 50,
            "wind_speed_10m": 5,
            "solar_radiation": 5,
            "hour": 12,
            "month": 6,
        }
        return self.calculate_mix(weather_data)

    def calculate_carbon_intensity(self, region: str = "India") -> float:
        """Calculate carbon intensity"""
        mix = self.get_energy_mix(region)
        return (
            mix.get("solar", 0) * SOLAR_CARBON_INTENSITY
            + mix.get("wind", 0) * WIND_CARBON_INTENSITY
            + mix.get("hydro", 0) * HYDRO_CARBON_INTENSITY
            + mix.get("thermal", 0) * THERMAL_CARBON_INTENSITY
        )

    def calculate_renewable_share(self, region: str = "India") -> float:
        """Calculate renewable percentage"""
        mix = self.get_energy_mix(region)
        return mix.get("solar", 0) + mix.get("wind", 0) + mix.get("hydro", 0)

    def get_mix_breakdown(self, region: str = "India") -> pd.DataFrame:
        """Get mix as DataFrame"""
        mix = self.get_energy_mix(region)
        return pd.DataFrame(
            {"Source": list(mix.keys()), "Percentage": list(mix.values())}
        )

    def forecast_mix_2030(self, region: str = "India") -> Dict:
        """Forecast mix for 2030"""
        current = self.get_energy_mix(region)
        return {
            "solar": min(current.get("solar", 15) * 2, 30),
            "wind": min(current.get("wind", 8) * 1.5, 25),
            "hydro": current.get("hydro", 12),
            "thermal": max(
                100
                - current.get("solar", 15) * 2
                - current.get("wind", 8) * 1.5
                - current.get("hydro", 12),
                25,
            ),
        }


# ============================================
# SECTION 3: RECOMMENDATION ENGINE
# ============================================
class RecommendationEngine:
    """
    Generate actionable recommendations for electricity consumers
    """

    def __init__(self):
        self.heat_wave_threshold = HEAT_WAVE_TEMP
        self.cold_wave_threshold = COLD_WAVE_TEMP
        self.good_wind_low = GOOD_WIND_LOW
        self.good_wind_high = GOOD_WIND_HIGH
        self.cloudy_threshold = CLOUDY_THRESHOLD
        self.morning_peak_start = MORNING_PEAK_START
        self.evening_peak_start = EVENING_PEAK_START

    def get_recommendations(
        self,
        weather_data: Dict,
        demand_peak_time: Optional[datetime] = None,
        energy_mix: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate list of recommendations based on conditions"""
        recommendations = []

        temperature = weather_data.get("temperature_2m", 20)
        humidity = weather_data.get("relative_humidity", 50)
        wind_speed = weather_data.get("wind_speed_10m", 5)
        cloud_cover = weather_data.get("cloud_cover", 50)
        hour = weather_data.get("hour", 12)

        if temperature > self.heat_wave_threshold:
            recommendations.append(
                {
                    "title": "Heat Wave Alert",
                    "description": f"Temperature {temperature}°C exceeds heat wave threshold. Pre-cool buildings before peak hours (6-9 PM).",
                    "savings_estimate": "15-20% cooling costs",
                    "priority": "High",
                    "category": "Conservation",
                    "icon": "🌡️",
                    "action_time": "Now",
                }
            )

        if wind_speed >= self.good_wind_low and wind_speed <= self.good_wind_high:
            recommendations.append(
                {
                    "title": "High Wind - Clean Energy Available",
                    "description": f"Wind speed {wind_speed} m/s is optimal for wind generation. Charge EVs/batteries during this period.",
                    "savings_estimate": "Reduce carbon footprint by 30%",
                    "priority": "Medium",
                    "category": "Renewable",
                    "icon": "💨",
                    "action_time": "Today",
                }
            )

        if cloud_cover > self.cloudy_threshold:
            recommendations.append(
                {
                    "title": "Low Solar Expected",
                    "description": f"{cloud_cover}% cloud cover will reduce solar generation. Consider reducing non-essential loads.",
                    "savings_estimate": "5-10% energy savings",
                    "priority": "Medium",
                    "category": "Conservation",
                    "icon": "☁️",
                    "action_time": "Today",
                }
            )

        if hour >= self.evening_peak_start and hour <= 21:
            recommendations.append(
                {
                    "title": "Peak Demand Period",
                    "description": "Current time is peak demand period. Avoid running high-power appliances.",
                    "savings_estimate": "Avoid peak charges",
                    "priority": "High",
                    "category": "Load Shifting",
                    "icon": "⚡",
                    "action_time": "Now",
                }
            )

        if humidity > 80 and temperature > 30:
            recommendations.append(
                {
                    "title": "High Humidity Warning",
                    "description": f"{humidity}% humidity combined with {temperature}°C feels hotter. Use fans instead of AC when possible.",
                    "savings_estimate": "10-15% cooling costs",
                    "priority": "Medium",
                    "category": "Conservation",
                    "icon": "💦",
                    "action_time": "Now",
                }
            )

        if wind_speed < self.good_wind_low:
            recommendations.append(
                {
                    "title": "Low Wind - High Thermal Expected",
                    "description": f"Wind {wind_speed} m/s below optimal. Grid will rely more on thermal generation.",
                    "savings_estimate": "Expect higher rates",
                    "priority": "Low",
                    "category": "Alert",
                    "icon": "🔔",
                    "action_time": "Today",
                }
            )

        if temperature < self.cold_wave_threshold:
            recommendations.append(
                {
                    "title": "Cold Weather - Heating Required",
                    "description": f"Temperature {temperature}°C triggers heating load. Use efficient heating methods.",
                    "savings_estimate": "10-15% heating costs",
                    "priority": "Medium",
                    "category": "Conservation",
                    "icon": "❄️",
                    "action_time": "Now",
                }
            )

        if hour >= 22 or hour <= 5:
            recommendations.append(
                {
                    "title": "Off-Peak Hours",
                    "description": "Current time is off-peak. Good time for high-consumption tasks.",
                    "savings_estimate": "30-50% cheaper rates",
                    "priority": "Low",
                    "category": "Cost",
                    "icon": "🌙",
                    "action_time": "Now",
                }
            )

        return sorted(
            recommendations,
            key=lambda x: {"High": 0, "Medium": 1, "Low": 2}[x["priority"]],
        )

    def get_peak_alert(
        self, current_time: datetime, peak_time: datetime, peak_demand_mw: float
    ) -> Dict:
        """Generate specific alert for upcoming peak demand"""
        time_until_peak = (peak_time - current_time).total_seconds() / 3600

        return {
            "time_until_peak_hours": round(time_until_peak, 1),
            "peak_demand_mw": peak_demand_mw,
            "alert_level": "Critical"
            if time_until_peak < 2
            else "Warning"
            if time_until_peak < 6
            else "Info",
            "actions": [
                "Delay high-power appliance use",
                "Pre-cool or pre-heat if needed",
                "Check for renewable-rich periods",
            ],
        }

    def get_seasonal_tips(self, season: str, weather_forecast: Dict) -> List[Dict]:
        """Generate season-specific recommendations"""
        tips = []

        if season.lower() == "summer":
            tips.append(
                {
                    "title": "Summer Efficiency",
                    "description": "Set AC to 24-26°C. Use ceiling fans to circulate air.",
                    "impact": "15-20% savings",
                }
            )
        elif season.lower() == "winter":
            tips.append(
                {
                    "title": "Winter Heating",
                    "description": "Use sunlight for passive heating during day. Seal drafts.",
                    "impact": "10-15% savings",
                }
            )
        elif season.lower() == "monsoon":
            tips.append(
                {
                    "title": "Monsoon Precautions",
                    "description": "Expect reduced solar generation. Check for moisture-related issues.",
                    "impact": "Plan for lower solar",
                }
            )

        return tips

    def get_efficiency_tips(self, appliance_type: str) -> Dict:
        """Get appliance-specific efficiency recommendations"""
        tips = {
            "ac": {
                "temperature": "24-26°C",
                "maintenance": "Clean filters monthly",
                "savings": "10-15% per degree increase",
            },
            "refrigerator": {
                "temperature": "3-4°C for fridge, -18°C for freezer",
                "maintenance": "Clean coils annually",
                "savings": "5-10% with proper settings",
            },
            "water_heater": {
                "temperature": "50-55°C",
                "maintenance": "Insulate tank",
                "savings": "20-30% with timer",
            },
            "ev": {
                "charging": "Charge during off-peak (10PM-6AM)",
                "savings": "50%+ vs peak rates",
            },
        }
        return tips.get(appliance_type.lower(), {})

    # Backwards compatibility
    def get_consumption_recommendations(
        self, current_kwh: float, temperature: float, region: str = "UK"
    ) -> List[Dict]:
        """Get consumption recommendations"""
        weather_data = {
            "temperature_2m": temperature,
            "relative_humidity": 50,
            "wind_speed_10m": 5,
            "cloud_cover": 50,
            "hour": 12,
        }
        return self.get_recommendations(weather_data)

    def get_tariff_recommendations(
        self, consumption_kwh: float, region: str = "UK", usage_pattern: str = "mixed"
    ) -> List[Dict]:
        """Get tariff recommendations"""
        return [
            {
                "type": "tariff",
                "title": "Fixed Rate",
                "description": "Consider fixed rate for stable bills",
                "savings": consumption_kwh * 365 * 0.05,
            }
        ]

    def get_investment_recommendations(
        self, annual_spend: float, region: str = "UK"
    ) -> pd.DataFrame:
        """Get investment recommendations"""
        return pd.DataFrame(
            {
                "Investment": ["LED Lighting", "Smart Thermostat", "Solar Panels"],
                "Cost": ["£200", "£250", "£5000"],
                "Annual Savings": ["£40", "£150", "£400"],
                "ROI Years": [5.0, 1.7, 12.5],
                "Priority": ["High", "High", "Medium"],
            }
        )


# ============================================
# SECTION 4: ECONOMIC ANALYZER
# ============================================
class EconomicAnalyzer:
    """
    Calculate economic and environmental impact of forecasts
    """

    def __init__(self):
        self.residential_price = RESIDENTIAL_PRICE_PER_KWH
        self.commercial_price = COMMERCIAL_PRICE_PER_KWH
        self.industrial_price = INDUSTRIAL_PRICE_PER_KWH
        self.peak_multiplier = PEAK_PRICE_MULTIPLIER
        self.off_peak_multiplier = OFF_PEAK_PRICE_MULTIPLIER

        self.grid_carbon = GRID_CARBON_INTENSITY
        self.solar_carbon = SOLAR_CARBON_INTENSITY
        self.wind_carbon = WIND_CARBON_INTENSITY
        self.hydro_carbon = HYDRO_CARBON_INTENSITY
        self.thermal_carbon = THERMAL_CARBON_INTENSITY

    def calculate_savings(
        self,
        predicted_demand: float,
        actual_demand: Optional[float] = None,
        customer_type: str = "residential",
    ) -> Dict:
        """Calculate cost savings from accurate forecasting"""
        price = self._get_price(customer_type)

        if actual_demand is not None:
            forecast_error = abs(predicted_demand - actual_demand)
            cost_saved = forecast_error * price * 0.15
        else:
            cost_saved = predicted_demand * price * 0.15

        return {
            "predicted_cost": predicted_demand * price,
            "estimated_savings": cost_saved,
            "improvement_percent": 15,
            "customer_type": customer_type,
        }

    def calculate_carbon_impact(
        self, demand_mwh: float, energy_mix: Optional[Dict] = None
    ) -> Dict:
        """Calculate carbon emissions based on generation mix"""
        if energy_mix is not None:
            emissions = (
                demand_mwh * 1000 * energy_mix.get("solar", 0) / 100 * self.solar_carbon
                + demand_mwh * 1000 * energy_mix.get("wind", 0) / 100 * self.wind_carbon
                + demand_mwh
                * 1000
                * energy_mix.get("hydro", 0)
                / 100
                * self.hydro_carbon
                + demand_mwh
                * 1000
                * energy_mix.get("thermal", 0)
                / 100
                * self.thermal_carbon
            ) / 1000
        else:
            emissions = demand_mwh * self.grid_carbon

        trees_needed = emissions * TREES_PER_KG_CO2_PER_YEAR

        return {
            "total_emissions_kg": emissions,
            "emissions_per_kwh": emissions / (demand_mwh * 1000) * 1000,
            "equivalent_trees": trees_needed,
            "equivalent_cars_km": emissions * 3700,
        }

    def calculate_peak_shaving_value(
        self, original_peak_mw: float, new_peak_mw: float, duration_hours: float
    ) -> Dict:
        """Calculate value of reducing peak demand"""
        peak_reduction_mw = original_peak_mw - new_peak_mw

        infrastructure_savings = peak_reduction_mw * 5000
        operational_savings = peak_reduction_mw * duration_hours * self.commercial_price

        return {
            "peak_reduction_mw": peak_reduction_mw,
            "infrastructure_savings": infrastructure_savings,
            "operational_savings": operational_savings,
            "total_value": infrastructure_savings + operational_savings,
        }

    def calculate_renewable_benefit(
        self, energy_mix: Dict, total_demand_mwh: float
    ) -> Dict:
        """Calculate benefit of renewable generation"""
        renewable_pct = (
            energy_mix.get("solar", 0)
            + energy_mix.get("wind", 0)
            + energy_mix.get("hydro", 0)
        )

        renewable_mwh = total_demand_mwh * renewable_pct / 100

        fossil_avoided_kg = (
            renewable_mwh * 1000 * (self.thermal_carbon - self.wind_carbon)
        )

        cost_saved = renewable_mwh * self.industrial_price

        return {
            "renewable_percentage": renewable_pct,
            "fossil_fuel_avoided_kg": fossil_avoided_kg,
            "cost_saved": cost_saved,
        }

    def get_price_by_hour(self, hour: int, customer_type: str = "residential") -> float:
        """Get electricity price for specific hour"""
        base_price = self._get_price(customer_type)

        if 7 <= hour <= 10 or 18 <= hour <= 21:
            return base_price * self.peak_multiplier
        elif 22 <= hour <= 6:
            return base_price * self.off_peak_multiplier
        else:
            return base_price

    def _get_price(self, customer_type: str) -> float:
        """Get base price by customer type"""
        prices = {
            "residential": self.residential_price,
            "commercial": self.commercial_price,
            "industrial": self.industrial_price,
        }
        return prices.get(customer_type.lower(), self.residential_price)

    def format_currency(self, amount: float) -> str:
        """Format currency with Indian numbering"""
        return f"₹{amount:,.2f}"

    def format_carbon(self, kg_co2: float) -> str:
        """Format carbon with appropriate units"""
        if kg_co2 >= 1000:
            return f"{kg_co2 / 1000:.2f} tonnes"
        return f"{kg_co2:.2f} kg"


# ============================================
# SECTION 5: VISUALIZATION HELPERS
# ============================================
class VisualizationHelper:
    """Create charts and graphs for Streamlit dashboard"""

    def __init__(self):
        self.height = CHART_HEIGHT
        self.width = CHART_WIDTH

        self.energy_colors = {
            "solar": COLOR_SOLAR,
            "wind": COLOR_WIND,
            "hydro": COLOR_HYDRO,
            "thermal": COLOR_THERMAL,
            "nuclear": "#FFA500",
        }

    def create_demand_forecast_chart(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        confidence_intervals: Optional[Dict] = None,
    ) -> go.Figure:
        """Create main demand forecast chart"""
        fig = go.Figure()

        if "demand_mw" in historical_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=historical_df.index,
                    y=historical_df["demand_mw"],
                    mode="lines",
                    name="Historical",
                    line=dict(color=self.energy_colors.get("thermal"), width=2),
                )
            )

        if "demand_mw" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["demand_mw"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color=COLOR_FORECAST, width=2, dash="dash"),
                )
            )

            if confidence_intervals:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=confidence_intervals.get("upper", forecast_df["demand_mw"]),
                        mode="lines",
                        name="Upper Bound",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=confidence_intervals.get("lower", forecast_df["demand_mw"]),
                        mode="lines",
                        name="Confidence Interval",
                        fill="tonexty",
                        fillcolor="rgba(255, 127, 14, 0.2)",
                        line=dict(width=0),
                    )
                )

        fig.update_layout(
            title="Electricity Demand Forecast",
            xaxis_title="Date/Time",
            yaxis_title="Demand (MW)",
            template="plotly_white",
            height=self.height,
            hovermode="x unified",
        )

        return fig

    def create_energy_mix_chart(
        self, mix_data: Union[pd.DataFrame, Dict], chart_type: str = "pie"
    ) -> go.Figure:
        """Create energy mix visualization"""
        if chart_type == "pie":
            if isinstance(mix_data, dict):
                labels = list(mix_data.keys())
                values = list(mix_data.values())
            else:
                labels = ["solar", "wind", "hydro", "thermal"]
                values = [mix_data.get(k, 0) for k in labels]

            fig = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(
                        colors=[self.energy_colors.get(k, "#000") for k in labels]
                    ),
                    hole=0.4,
                )
            )
            fig.update_layout(title="Energy Mix")

        elif chart_type == "bar":
            if isinstance(mix_data, pd.DataFrame):
                fig = px.bar(
                    mix_data,
                    x="datetime" if "datetime" in mix_data.columns else mix_data.index,
                    y=["solar", "wind", "hydro", "thermal"],
                    color_discrete_map=self.energy_colors,
                )
            else:
                fig = go.Figure(
                    go.Bar(
                        x=list(mix_data.keys()),
                        y=list(mix_data.values()),
                        marker_color=[
                            self.energy_colors.get(k, "#000") for k in mix_data.keys()
                        ],
                    )
                )
            fig.update_layout(title="Energy Mix by Source", height=self.height)

        else:
            fig = go.Figure()

        return fig

    def create_gauge_chart(
        self,
        value: float,
        title: str,
        min_val: float = 0,
        max_val: float = 100,
        unit: str = "%",
    ) -> go.Figure:
        """Create single gauge chart"""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title},
                gauge={
                    "axis": {"range": [min_val, max_val]},
                    "bar": {"color": "#2ca02c"},
                    "steps": [
                        {"range": [min_val, max_val * 0.33], "color": "#ff9896"},
                        {"range": [max_val * 0.33, max_val * 0.66], "color": "#ffeaa7"},
                        {"range": [max_val * 0.66, max_val], "color": "#55efc4"},
                    ],
                },
            )
        )
        fig.update_layout(height=250)
        return fig

    def create_peak_demand_heatmap(self, demand_data: pd.DataFrame) -> go.Figure:
        """Create hourly heatmap showing peak periods"""
        if "demand_mw" not in demand_data.columns:
            demand_data["demand_mw"] = np.random.uniform(1000, 5000, len(demand_data))

        demand_data = demand_data.copy()
        demand_data["hour"] = demand_data.index.hour
        demand_data["dayofweek"] = demand_data.index.dayofweek

        heatmap_data = demand_data.pivot_table(
            values="demand_mw", index="dayofweek", columns="hour", aggfunc="mean"
        )

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Demand (MW)"),
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(title="Peak Demand Heatmap")

        return fig

    def create_accuracy_comparison_chart(self, model_metrics: Dict) -> go.Figure:
        """Compare accuracy across models"""
        models = list(model_metrics.keys())
        metrics = list(model_metrics[models[0]].keys()) if models else []

        fig = go.Figure()

        for metric in metrics:
            values = [model_metrics[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(name=metric, x=models, y=values))

        fig.update_layout(
            barmode="group", title="Model Accuracy Comparison", yaxis_title="Value"
        )

        return fig

    def create_weather_impact_chart(
        self, demand_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> go.Figure:
        """Show correlation between weather and demand"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=demand_data.index,
                y=demand_data.get("demand_mw", demand_data.iloc[:, 0]),
                name="Demand",
                line=dict(color=COLOR_DEMAND),
            ),
            secondary_y=False,
        )

        if "temperature_2m" in weather_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=weather_data.index,
                    y=weather_data["temperature_2m"],
                    name="Temperature",
                    line=dict(color=COLOR_FORECAST),
                ),
                secondary_y=True,
            )

        fig.update_layout(title="Weather Impact on Demand")
        fig.update_yaxes(title_text="Demand (MW)", secondary_y=False)
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

        return fig


# ============================================
# SECTION 6: REPORT GENERATOR
# ============================================
class ReportGenerator:
    """Generate professional reports"""

    def __init__(self):
        self.company_name = "Electricity Demand Forecasting System"

    def generate_summary_report(self, forecast_data: Dict, metrics: Dict) -> str:
        """Generate concise text summary"""
        report = f"""
{self.company_name}
{"=" * 50}

FORECAST SUMMARY
---------------
Forecast Period: {forecast_data.get("start_date", "N/A")} to {forecast_data.get("end_date", "N/A")}
Peak Demand: {forecast_data.get("peak_demand_mw", 0):.0f} MW
Average Demand: {forecast_data.get("avg_demand_mw", 0):.0f} MW

MODEL PERFORMANCE
-----------------
MAPE: {metrics.get("mape", 0):.2f}%
RMSE: {metrics.get("rmse", 0):.2f} MW
Accuracy: {metrics.get("accuracy", 0):.1f}%

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
        return report

    def generate_full_report(
        self, forecast_data: Dict, metrics: Dict, recommendations: List[Dict]
    ) -> str:
        """Generate complete report"""
        report = self.generate_summary_report(forecast_data, metrics)

        report += "\n\nRECOMMENDATIONS\n"
        report += "---------------\n"

        for i, rec in enumerate(recommendations[:5], 1):
            report += f"\n{i}. {rec.get('title', 'N/A')}\n"
            report += f"   {rec.get('description', '')}\n"
            report += f"   Priority: {rec.get('priority', 'N/A')}\n"

        return report


# ============================================
# SECTION 7: FILE EXPORT HANDLERS
# ============================================
class ExportHandler:
    """Handle downloading of results in various formats"""

    def export_to_csv(self, data: Union[pd.DataFrame, Dict], filename: str) -> bytes:
        """Export data to CSV"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        return data.to_csv(index=True).encode("utf-8")

    def export_to_json(self, data: Dict, filename: str) -> bytes:
        """Export data to JSON"""
        return json.dumps(data, indent=2, default=str).encode("utf-8")

    def create_download_button(self, file_bytes: bytes, filename: str, mime_type: str):
        """Create Streamlit download button"""
        import streamlit as st

        return st.download_button(
            label=f"Download {filename}",
            data=file_bytes,
            file_name=filename,
            mime=mime_type,
        )


# ============================================
# SECTION 8: GENERAL UTILITIES
# ============================================
class GeneralUtilities:
    """Miscellaneous helper functions"""

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """Check if latitude/longitude are valid"""
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def get_season(month: int, country: str = "India") -> str:
        """Determine season based on month"""
        if country == "India":
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Summer"
            elif month in [6, 7, 8, 9]:
                return "Monsoon"
            else:
                return "Post-Monsoon"
        else:
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

    @staticmethod
    def calculate_heat_index(temp_celsius: float, humidity_percent: float) -> float:
        """Calculate apparent temperature"""
        if temp_celsius < 27:
            return temp_celsius

        hi = (
            -8.78469475556
            + 1.61139411 * temp_celsius
            + 2.33854883889 * humidity_percent
        )
        hi += -0.14611605 * temp_celsius * humidity_percent
        hi += -0.012308094 * temp_celsius**2
        hi += -0.0164248277778 * humidity_percent**2

        return hi

    @staticmethod
    def calculate_wind_chill(temp_celsius: float, wind_speed_ms: float) -> float:
        """Calculate wind chill temperature"""
        if temp_celsius > 10 or wind_speed_ms < 1.34:
            return temp_celsius

        wind_kmh = wind_speed_ms * 3.6
        wc = (
            13.12
            + 0.6215 * temp_celsius
            - 11.37 * (wind_kmh**0.16)
            + 0.3965 * temp_celsius * (wind_kmh**0.16)
        )

        return wc

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
        """Divide with protection against division by zero"""
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def round_to_significant(value: float, digits: int = 3) -> float:
        """Round number to significant digits"""
        if value == 0:
            return 0
        return round(value, digits - int(np.floor(np.log10(abs(value)))) - 1)

    @staticmethod
    def format_timestamp(dt: datetime, format_type: str = "datetime") -> str:
        """Format timestamp for display"""
        formats = {
            "datetime": "%b %d, %Y %I:%M %p",
            "date": "%b %d, %Y",
            "time": "%I:%M %p",
            "filename": "%Y-%m-%d_%H-%M-%S",
        }
        return dt.strftime(formats.get(format_type, "%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def generate_cache_key(lat: float, lon: float, forecast_days: int) -> str:
        """Generate unique key for caching"""
        key_string = f"{lat}_{lon}_{forecast_days}"
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change"""
        if old_value == 0:
            return 0
        return ((new_value - old_value) / old_value) * 100

    @staticmethod
    def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average"""
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# ============================================
# SECTION 9: CACHING HELPERS
# ============================================
class CacheManager:
    """Manage caching of API responses and predictions"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}

    def get_cached(self, key: str) -> Optional[Any]:
        """Retrieve cached data if available"""
        return self.cache.get(key)

    def set_cached(self, key: str, data: Any, ttl_hours: int = 24):
        """Store data in cache with TTL"""
        self.cache[key] = {
            "data": data,
            "expires": datetime.now() + timedelta(hours=ttl_hours),
        }

    def clear_expired(self):
        """Remove expired cache entries"""
        now = datetime.now()
        self.cache = {k: v for k, v in self.cache.items() if v["expires"] > now}

    def get_cache_stats(self) -> Dict:
        """Return cache statistics"""
        return {
            "total_entries": len(self.cache),
            "expired_entries": sum(
                1 for v in self.cache.values() if v["expires"] <= datetime.now()
            ),
        }


# ============================================
# SECTION 11: EXPORTS
# ============================================
__all__ = [
    "EnergyMixCalculator",
    "RecommendationEngine",
    "EconomicAnalyzer",
    "VisualizationHelper",
    "ReportGenerator",
    "ExportHandler",
    "GeneralUtilities",
    "CacheManager",
    "COLOR_SOLAR",
    "COLOR_WIND",
    "COLOR_HYDRO",
    "COLOR_THERMAL",
    "MAX_SOLAR_PERCENT",
    "HEAT_WAVE_TEMP",
]
