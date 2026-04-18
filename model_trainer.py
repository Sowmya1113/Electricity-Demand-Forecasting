# ============================================
# model_trainer.py
# PURPOSE: Train and manage NHiTS & iTransformer models for electricity demand forecasting
# - Define NHiTS architecture (multi-scale forecasting)
# - Define iTransformer architecture (multivariate time series)
# - Ensemble predictions from both models
# - Handle training, validation, and evaluation
# - Save/load trained models
# ============================================

# ============================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================
import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

import warnings

warnings.filterwarnings("ignore")

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, iTransformer

    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    warnings.warn("NeuralForecast not available, using fallback models")

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "input_length": 720,
    "output_length": 720,
    "hidden_dim": 512,
    "learning_rate": 0.0005,
    "batch_size": 64,
    "max_epochs": 150,
    "patience": 20,
    "d_model": 256,
    "n_heads": 16,
    "n_layers": 4,
    "dropout": 0.05,
    "weight_decay": 0.01,
}


def setup_logger(name: str = "ModelTrainer") -> logging.Logger:
    """Configure logging for the model trainer"""
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
# SECTION 7: DATA LOADING FOR MODELS
# ============================================
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data
    PURPOSE: Prepare batches for training
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_length: int = 168,
        output_length: int = 24,
    ):
        self.features = features
        self.targets = targets
        self.input_length = input_length
        self.output_length = output_length

        self.valid_indices = list(
            range(len(features) - input_length - output_length + 1)
        )

    def __len__(self) -> int:
        if len(self.features.shape) == 3:
            return len(self.features)
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.features.shape) == 3:
            return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.targets[idx])

        start_idx = self.valid_indices[idx]

        X = self.features[start_idx : start_idx + self.input_length]
        y = self.targets[
            start_idx + self.input_length : start_idx
            + self.input_length
            + self.output_length
        ]

        return torch.FloatTensor(X), torch.FloatTensor(y)


class DataPreprocessor:
    """
    Prepare data for model input
    PURPOSE: Scaling, sequence creation, batch generation
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.feature_scaler = None
        self.target_scaler = None

    def fit_scalers(self, train_data: pd.DataFrame, target_col: str = "demand_mw"):
        """Fit scalers on training data only"""
        feature_cols = [c for c in train_data.columns if c != target_col]

        if self.scaler_type == "standard":
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values.reshape(-1, 1)

        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)

        logger.info(f"Fitted scalers on {len(train_data)} samples")

    def transform(
        self, data: pd.DataFrame, target_col: str = "demand_mw"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply scaling to data"""
        feature_cols = [c for c in data.columns if c != target_col]

        X = self.feature_scaler.transform(data[feature_cols].values)

        if target_col in data.columns:
            y = self.target_scaler.transform(
                data[target_col].values.reshape(-1, 1)
            ).flatten()
        else:
            y = None

        return X, y

    def inverse_transform_target(self, scaled_data: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale"""
        if len(scaled_data.shape) == 1:
            scaled_data = scaled_data.reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled_data).flatten()

    def create_sequences(
        self,
        data: pd.DataFrame,
        input_length: int = 168,
        output_length: int = 24,
        target_col: str = "demand_mw",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting"""
        feature_cols = [c for c in data.columns if c != target_col]

        X, y = self.transform(data, target_col)

        X_sequences = []
        y_sequences = []

        for i in range(len(data) - input_length - output_length + 1):
            X_sequences.append(X[i : i + input_length])
            y_sequences.append(y[i + input_length : i + input_length + output_length])

        return np.array(X_sequences), np.array(y_sequences)

    def create_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        input_length: int = 168,
        output_length: int = 24,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders"""
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        train_dataset = TimeSeriesDataset(X_train, y_train, input_length, output_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, input_length, output_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, input_length, output_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(
            f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
        )

        return train_loader, val_loader, test_loader


# ============================================
# SECTION 2: NHiTS MODEL ARCHITECTURE
# ============================================
class NHiTSBlock(nn.Module):
    """
    Single NHiTS block - handles one temporal scale
    PURPOSE: Learn patterns at specific frequency (hourly, daily, weekly)
    """

    def __init__(
        self,
        input_length: int,
        output_length: int,
        pool_size: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.output_length = output_length

        self.pool = (
            nn.AvgPool1d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        )

        reduced_length = input_length // pool_size

        self.fc = nn.Sequential(
            nn.Linear(reduced_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x.unsqueeze(1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class NHiTSModel(nn.Module):
    """
    Complete NHiTS Model - Neural Hierarchical Interpolation for Time Series
    PURPOSE: Capture multiple seasonal patterns simultaneously
    """

    def __init__(
        self,
        input_length: int = 168,
        output_length: int = 24,
        hidden_dim: int = 256,
        pool_sizes: List[int] = None,
    ):
        super().__init__()

        if pool_sizes is None:
            pool_sizes = [1, 2, 4, 8, 24]

        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(input_length, output_length, ps, hidden_dim // (i + 1))
                for i, ps in enumerate(pool_sizes)
            ]
        )

        self.output_length = output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x[:, :, 0]

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        outputs = [block(x) for block in self.blocks]

        return torch.stack(outputs).mean(dim=0)

    def get_scale_contributions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get contribution from each temporal scale"""
        if len(x.shape) == 3:
            x = x[:, :, 0]

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        return [block(x) for block in self.blocks]


# ============================================
# SECTION 3: iTRANSFORMER MODEL ARCHITECTURE
# ============================================
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class iTransformerBlock(nn.Module):
    """Single transformer block for iTransformer"""

    def __init__(
        self, variate_num: int, d_model: int, n_heads: int, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class iTransformer(nn.Module):
    """
    Inverted Transformer for Multivariate Time Series Forecasting
    PURPOSE: Model relationships between different variables
    """

    def __init__(
        self,
        variate_num: int,
        input_len: int = 168,
        pred_len: int = 24,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.variate_num = variate_num
        self.input_len = input_len
        self.pred_len = pred_len

        self.variate_embedding = nn.Linear(input_len, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=variate_num)

        self.encoder_layers = nn.ModuleList(
            [
                iTransformerBlock(variate_num, d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

        self.projection = nn.Linear(d_model, pred_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        batch_size = x.size(0)

        x = x.permute(0, 2, 1)
        x = self.variate_embedding(x)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)

        output = self.projection(x)

        return output


# ============================================
# SECTION 4: iTRANSFORMER FOR ELECTRICITY DEMAND
# ============================================
class ElectricityTransformer(nn.Module):
    """
    iTransformer specifically for electricity demand forecasting
    PURPOSE: Model multivariate relationships between demand and weather covariates

    WHY iTRANSFORMER FOR DEMAND:
    - Electricity demand is influenced by multiple variables (temperature, humidity, time)
    - iTransformer treats each variable as a token, capturing cross-variable dependencies
    - Excellent for understanding how weather affects demand

    INPUT VARIABLES:
    - demand_mw (primary target)
    - temperature_2m
    - relative_humidity
    - hour_sin, hour_cos (temporal)
    - is_peak_hour
    - heating_degree, cooling_degree
    """

    def __init__(
        self,
        variate_num: int,
        input_len: int = 168,
        pred_len: int = 24,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.variate_num = variate_num
        self.input_len = input_len
        self.pred_len = pred_len

        self.variate_embedding = nn.Linear(input_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=variate_num)

        self.encoder_layers = nn.ModuleList(
            [
                iTransformerBlock(variate_num, d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

        self.demand_projection = nn.Linear(d_model, pred_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1)
        x = self.variate_embedding(x)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)
        output = self.demand_projection(x)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights to see which variables drive demand"""
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1)
        x = self.variate_embedding(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoding(x)

        attn_layer = self.encoder_layers[0].attention
        attn_out, _ = attn_layer(x, x, x)

        return attn_out.mean(dim=0)


# ============================================
# SECTION 4B: NHiTS FOR WEATHER FORECASTING
# ============================================
class WeatherNHiTS(nn.Module):
    """
    NHiTS specifically for weather forecasting
    PURPOSE: Capture multi-scale seasonal patterns in weather data

    WHY NHiTS FOR WEATHER:
    - Weather has clear hierarchical patterns: hourly, daily, seasonal
    - NHiTS explicitly models different temporal frequencies
    - Temperature follows daily cycles AND seasonal trends
    - Solar radiation has strong diurnal patterns

    TARGET VARIABLES:
    - temperature_2m
    - solar_radiation
    - wind_speed
    - relative_humidity
    """

    def __init__(
        self,
        input_length: int = 168,
        output_length: int = 24,
        hidden_dim: int = 256,
        pool_sizes: List[int] = None,
        n_outputs: int = 1,
    ):
        super().__init__()

        if pool_sizes is None:
            pool_sizes = [1, 2, 4, 8, 24, 168]

        self.n_outputs = n_outputs

        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(
                    input_length, output_length * n_outputs, ps, hidden_dim // (i + 1)
                )
                for i, ps in enumerate(pool_sizes)
            ]
        )

        self.output_length = output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x[:, :, 0]

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        outputs = [block(x) for block in self.blocks]

        combined = torch.stack(outputs).mean(dim=0)

        if self.n_outputs > 1:
            combined = combined.view(x.size(0), self.output_length, self.n_outputs)

        return combined

    def get_scale_contributions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get forecast contribution from each temporal scale"""
        if len(x.shape) == 3:
            x = x[:, :, 0]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        scale_names = ["hourly", "2h", "4h", "8h", "daily", "weekly"]
        return {
            name: block(x)
            for name, block in zip(scale_names[: len(self.blocks)], self.blocks)
        }


# ============================================
# SECTION 4C: HYBRID ENSEMBLE
# ============================================
class HybridEnsemble(nn.Module):
    """
    Hybrid ensemble combining:
    - iTransformer for electricity demand (with weather covariates)
    - NHiTS for weather forecasting

    PURPOSE: Leverage each model's strengths
    - iTransformer captures demand-weather relationships
    - NHiTS provides accurate weather forecasts
    - Combined: Better demand predictions using weather forecasts
    """

    def __init__(
        self,
        input_length: int = 168,
        output_length: int = 24,
        n_demand_features: int = 5,
        n_weather_features: int = 4,
        demand_config: Dict = None,
        weather_config: Dict = None,
    ):
        super().__init__()

        demand_config = demand_config or {}
        weather_config = weather_config or {}

        self.output_length = output_length

        self.demand_model = ElectricityTransformer(
            variate_num=n_demand_features,
            input_len=input_length,
            pred_len=output_length,
            d_model=demand_config.get("d_model", 128),
            n_heads=demand_config.get("n_heads", 8),
            n_layers=demand_config.get("n_layers", 3),
        )

        self.weather_model = WeatherNHiTS(
            input_length=input_length,
            output_length=output_length,
            hidden_dim=weather_config.get("hidden_dim", 256),
            n_outputs=n_weather_features,
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_length + n_weather_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_length),
        )

        self.demand_weight = nn.Parameter(torch.tensor(0.7))
        self.weather_influence = nn.Parameter(torch.tensor(0.3))

    def forward(
        self, demand_history: torch.Tensor, weather_history: torch.Tensor
    ) -> torch.Tensor:
        demand_pred = self.demand_model(demand_history)

        weather_pred = self.weather_model(weather_history)

        if len(weather_pred.shape) == 3:
            weather_pred = weather_pred.mean(dim=-1)

        combined = torch.cat([demand_pred, weather_pred], dim=-1)

        fused_pred = self.fusion(combined)

        weights = torch.softmax(
            torch.stack([self.demand_weight, self.weather_influence]), dim=0
        )

        final_pred = weights[0] * demand_pred + weights[1] * fused_pred

        return final_pred

    def get_demand_forecast(self, demand_history: torch.Tensor) -> torch.Tensor:
        """Get demand forecast only (iTransformer)"""
        return self.demand_model(demand_history)

    def get_weather_forecast(self, weather_history: torch.Tensor) -> torch.Tensor:
        """Get weather forecast only (NHiTS)"""
        return self.weather_model(weather_history)

    def get_model_weights(self) -> Dict[str, float]:
        """Return current ensemble weights"""
        weights = (
            torch.softmax(
                torch.stack([self.demand_weight, self.weather_influence]), dim=0
            )
            .detach()
            .numpy()
        )
        return {
            "demand_transformer": float(weights[0]),
            "weather_nhits": float(weights[1]),
        }


# ============================================
# LEGACY: Original Ensemble for backwards compatibility
# ============================================
class EnsembleForecaster(nn.Module):
    """
    Legacy ensemble for backwards compatibility
    """

    def __init__(
        self,
        input_length: int = 168,
        output_length: int = 24,
        n_features: int = 1,
        nhits_config: Dict = None,
        itransformer_config: Dict = None,
    ):
        super().__init__()

        nhits_config = nhits_config or {}
        itransformer_config = itransformer_config or {}

        self.nhits = NHiTSModel(
            input_length=input_length,
            output_length=output_length,
            hidden_dim=nhits_config.get("hidden_dim", 256),
        )

        self.itransformer = iTransformer(
            variate_num=n_features,
            input_len=input_length,
            pred_len=output_length,
            d_model=itransformer_config.get("d_model", 128),
            n_heads=itransformer_config.get("n_heads", 8),
            n_layers=itransformer_config.get("n_layers", 3),
        )

        self.nhits_weight = nn.Parameter(torch.tensor(0.5))
        self.itransformer_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        demand_history: torch.Tensor,
        weather_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        nhits_out = self.nhits(demand_history)

        if weather_history is not None and weather_history.shape[-1] > 1:
            combined = torch.cat([demand_history, weather_history], dim=-1)
            itransformer_out = self.itransformer(combined)
        else:
            itransformer_out = self.itransformer(demand_history)

        weights = torch.softmax(
            torch.stack([self.nhits_weight, self.itransformer_weight]), dim=0
        )

        output = weights[0] * nhits_out + weights[1] * itransformer_out

        return output

    def get_model_weights(self) -> Dict[str, float]:
        """Return current ensemble weights"""
        weights = (
            torch.softmax(
                torch.stack([self.nhits_weight, self.itransformer_weight]), dim=0
            )
            .detach()
            .numpy()
        )
        return {"nhits": float(weights[0]), "itransformer": float(weights[1])}


# ============================================
# SECTION 5: TRAINING MANAGER
# ============================================
class ModelTrainer:
    def __init__(self, model, learning_rate, target_scaler=None):
        # ... other init code ...
        self.target_scaler = target_scaler   # Store scaler for evaluation

    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate and return metrics in original MW scale."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(X_batch)
                loss = nn.functional.mse_loss(output, y_batch)
                total_loss += loss.item()

                all_preds.append(output.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Use ModelEvaluator with scaler to get correct metrics
        evaluator = ModelEvaluator(target_scaler=self.target_scaler)
        metrics = evaluator.calculate_metrics(all_targets, all_preds)
        metrics["loss"] = avg_loss

        return metrics

# ============================================
# SECTION 6: LOSS FUNCTIONS
# ============================================
class DemandForecastingLoss(nn.Module):
    """
    Custom loss functions for demand forecasting
    """

    def __init__(self, peak_weight: float = 2.0):
        super().__init__()
        self.peak_weight = peak_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        hours: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate weighted loss"""
        base_loss = nn.functional.lse_loss(predictions, targets)

        return base_loss


class QuantileLoss(nn.Module):
    """Pinball loss for quantile forecasting"""

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate pinball loss for each quantile"""
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())

        return torch.stack(losses).mean()


# ============================================
# SECTION 8: EVALUATION METRICS
# ============================================
class ModelEvaluator:
    """
    Comprehensive model evaluation on original MW scale
    PURPOSE: Ensure metrics reflect real-world accuracy
    """

    def __init__(self, target_scaler: Optional[StandardScaler] = None):
        """
        Args:
            target_scaler: Fitted scaler used to inverse-transform predictions.
                          If None, assumes data is already in original scale.
        """
        self.target_scaler = target_scaler
        self.metrics_history = []

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate all accuracy metrics on ORIGINAL MW scale.
        If target_scaler is provided, y_true and y_pred should be scaled,
        and will be inverse-transformed before metric calculation.
        """
        # Inverse transform if scaler is available
        if self.target_scaler is not None:
            y_true = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Avoid division by zero in MAPE
        mask = y_true != 0
        if mask.sum() == 0:
            mape = 0.0
        else:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # MAE and RMSE
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Symmetric MAPE (SMAPE)
        smape = np.mean(200 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

        # Accuracy derived from MAPE
        simple_accuracy = max(0.0, 100 - mape)

        # Combined accuracy (weighted with R²)
        r2_based_accuracy = max(0.0, min(100.0, r2 * 100))
        combined_accuracy = (simple_accuracy * 0.6) + (r2_based_accuracy * 0.4)

        metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "smape": round(smape, 2),
            "r2": round(r2, 4),
            "accuracy": round(combined_accuracy, 2),
            "simple_accuracy": round(simple_accuracy, 2)
        }

        self.metrics_history.append(metrics)
        return metrics

    def check_accuracy_threshold(self, metrics: Dict, threshold: float = 85.0) -> Tuple[bool, str]:
        """Verify if model meets the required accuracy threshold."""
        acc = metrics.get("accuracy", 0)
        if acc >= threshold:
            return True, f"✅ PASS: {acc:.2f}% >= {threshold}%"
        return False, f"❌ FAIL: {acc:.2f}% < {threshold}%"


# ============================================
# SECTION 9: MODEL PERSISTENCE
# ============================================
class ModelCheckpoint:
    """
    Save and load model checkpoints
    PURPOSE: Resume training, save best models
    """

    def __init__(self, save_dir: str = "models/"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self, model: nn.Module, filename: str, metadata: Optional[Dict] = None
    ):
        """Save trained model"""
        filepath = self.save_dir / filename

        model_state = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }

        torch.save(model_state, filepath)
        logger.info(f"Saved model to {filepath}")

        return str(filepath)

    def load_model(self, model: nn.Module, filename: str) -> Dict:
        """Load trained model"""
        filepath = self.save_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Loaded model from {filepath}")

        return checkpoint.get("metadata", {})


# ============================================
# SECTION 10: HYPERPARAMETER TUNING
# ============================================
class HyperparameterOptimizer:
    """
    Automated hyperparameter search
    PURPOSE: Find optimal configuration for >85% accuracy
    """

    def __init__(self, model_class, train_loader: DataLoader, val_loader: DataLoader):
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_default_params(self) -> Dict:
        """Return proven good hyperparameters for demand forecasting"""
        return DEFAULT_CONFIG.copy()

    def random_search(
        self, param_grid: Dict, n_iterations: int = 10
    ) -> Tuple[Dict, float]:
        """Random search over hyperparameters"""
        best_params = None
        best_score = float("inf")

        for _ in range(n_iterations):
            params = {}
            for key, values in param_grid.items():
                params[key] = np.random.choice(values)

            model = self.model_class(**params)
            trainer = ModelTrainer(
                model, learning_rate=params.get("learning_rate", 0.001)
            )

            try:
                result = trainer.train(
                    self.train_loader, self.val_loader, epochs=10, patience=5
                )

                score = result["best_val_loss"]

                if score < best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")

        return best_params, best_score


# ============================================
# SECTION 11: MAIN TRAINING SCRIPT
# ============================================
def train_all_models(data, target_col="demand_mw", ...):
    preprocessor = DataPreprocessor()
    preprocessor.fit_scalers(train_data, target_col)

    # ... create dataloaders ...

    # Trainer with target scaler for correct metric reporting
    trainer = ModelTrainer(
        model,
        learning_rate=config["learning_rate"],
        target_scaler=preprocessor.target_scaler
    )

def main():
    """Entry point for training from command line"""
    import argparse

    parser = argparse.ArgumentParser(description="Train electricity forecasting models")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data CSV"
    )
    parser.add_argument(
        "--target", type=str, default="demand_mw", help="Target column name"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models/", help="Model save directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    data = pd.read_csv(args.data)

    config = DEFAULT_CONFIG.copy()
    config["max_epochs"] = args.epochs

    results = train_all_models(
        data=data, target_col=args.target, save_dir=args.save_dir, config=config
    )

    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)

    for model_name, result in results.items():
        if isinstance(result, dict) and "best_val_loss" in result:
            print(f"\n{model_name.upper()}:")
            print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
            print(f"  Epochs: {result['epochs_trained']}")


# ============================================
# SECTION 12: EXPORTS
# ============================================
__all__ = [
    "NHiTSModel",
    "NHiTSBlock",
    "iTransformer",
    "ElectricityTransformer",
    "WeatherNHiTS",
    "HybridEnsemble",
    "EnsembleForecaster",
    "ModelTrainer",
    "ModelEvaluator",
    "DataPreprocessor",
    "TimeSeriesDataset",
    "DemandForecastingLoss",
    "QuantileLoss",
    "ModelCheckpoint",
    "HyperparameterOptimizer",
    "train_all_models",
    "main",
    "DEFAULT_CONFIG",
    "DEVICE",
]
