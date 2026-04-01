# ⚡ Electricity Demand Forecaster

A comprehensive electricity demand forecasting application powered by deep learning models (NHITS & iTransformer) with real-time weather integration, energy mix analysis, and economic insights.

## Features

- **Demand Forecasting**: Predict electricity demand using state-of-the-art neural network models
- **Multi-Horizon Predictions**: Forecast from 6 hours to 1 week ahead
- **Weather Integration**: Real-time weather data from OpenWeatherMap API
- **Energy Mix Analysis**: View regional energy composition and renewable shares
- **Carbon Footprint**: Calculate and track carbon emissions
- **Economic Insights**: Tariff comparison, cost analysis, and investment recommendations

## Project Structure

```
electricity_forecast_project/
├── app.py              # Main Streamlit dashboard
├── model_trainer.py    # NHITS & iTransformer model training
├── data_pipeline.py    # Weather API, preprocessing, feature engineering
├── utils.py           # Energy mix, recommendations, economics
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Installation

1. Clone or download this repository

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Dashboard Sections

1. **Dashboard**: Overview of current demand, forecasts, and quick metrics
2. **Forecasting**: Generate demand predictions with configurable horizons
3. **Models**: Train and evaluate NHITS and iTransformer models
4. **Energy Mix**: Analyze regional energy composition
5. **Economics**: Cost analysis, tariffs, and investment recommendations
6. **Settings**: API configuration and preferences

## Configuration

### Weather API (Optional)

For real-time weather data, get a free API key from [OpenWeatherMap](https://openweathermap.org/api):

```bash
export OPENWEATHERMAP_API_KEY=your_api_key
```

Or enter it in the Settings page of the application.

### Model Parameters

Edit `model_trainer.py` to adjust:
- Training epochs
- Learning rates
- Forecast horizons
- Model architectures

## Requirements

- Python 3.9+
- Streamlit
- PyTorch
- NeuralForecast
- Pandas
- NumPy
- Plotly

See `requirements.txt` for complete list.

## Models

### NHITS (Neural Hierarchical Interpolation for Time Series)
- Multi-scale hierarchical time series forecasting
- Efficient for capturing long-term patterns

### iTransformer
- Transformer-based architecture for time series
- Excellent for complex temporal dependencies

## Energy Mix Data

The application includes pre-configured energy mix data for:
- United Kingdom (UK)
- United States (US)
- European Union (EU)

Data is sourced from publicly available energy statistics and updates periodically.

## Economic Calculations

The economics module provides:
- Bill estimation for various tariff types
- Annual cost projections
- Investment ROI calculations
- Savings recommendations

Tariff rates can be customized in `utils.py`.

## Development

### Adding Custom Models

1. Add model class to `model_trainer.py`
2. Register in the model selection
3. Implement training and prediction interfaces

### Adding Regions

1. Update `data_pipeline.py` with new city coordinates
2. Add energy mix data in `utils.py`
3. Add tariff rates in `utils.py`

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
