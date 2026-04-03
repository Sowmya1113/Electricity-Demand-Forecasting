# ============================================
# train_high_accuracy.py
# PURPOSE: Train and evaluate electricity demand forecasting models
# Replaces both train_high_accuracy.py and evaluate_models.py
# Usage:
#   python train_high_accuracy.py            → full training + evaluation
#   python train_high_accuracy.py --eval-only → evaluation only (no training)
# ============================================

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("TrainHighAccuracy")

from model_trainer import (
    NHiTSModel,
    iTransformer,
    EnsembleForecaster,
    ModelTrainer,
    ModelEvaluator,
    DataPreprocessor,
    ModelCheckpoint,
    DEFAULT_CONFIG,
)


# ============================================
# SECTION 1: DATA LOADING
# ============================================
def load_or_fetch_data() -> pd.DataFrame:
    """
    Load actual_demand.csv if it exists, otherwise fetch from Ember + NASA APIs.
    FIX (Problem 17 & 18): import from actual_data (not 'actual')
    """
    if os.path.exists("actual_demand.csv"):
        logger.info("Loading existing actual_demand.csv...")
        df = pd.read_csv("actual_demand.csv")
        logger.info(f"Loaded {len(df)} rows from actual_demand.csv")
        return df

    logger.info("actual_demand.csv not found — fetching real data from APIs...")
    # FIX: correct module name is actual_data, not actual
    from actual_data import fetch_actual_data

    df = fetch_actual_data()
    if df.empty:
        logger.error("fetch_actual_data() returned empty DataFrame. Cannot proceed.")
        sys.exit(1)

    df.to_csv("actual_demand.csv", index=False)
    logger.info(f"Saved {len(df)} rows to actual_demand.csv")
    return df


# ============================================
# SECTION 2: PREPROCESSING
# ============================================
def preprocess(df: pd.DataFrame, target_col: str = "demand_mw"):
    """
    Prepare DataFrame → sequences → DataLoaders.
    Returns preprocessor, loaders, n_features.
    """
    # Ensure datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    if target_col not in df.columns:
        logger.error(f"Column '{target_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)
    logger.info(f"Features ({n_features}): {feature_cols}")

    preprocessor = DataPreprocessor()
    preprocessor.fit_scalers(df, target_col)

    X, y = preprocessor.create_sequences(
        df,
        input_length=DEFAULT_CONFIG["input_length"],
        output_length=DEFAULT_CONFIG["output_length"],
        target_col=target_col,
    )
    logger.info(f"Created sequences: X={X.shape}, y={y.shape}")

    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        X,
        y,
        batch_size=DEFAULT_CONFIG["batch_size"],
        input_length=DEFAULT_CONFIG["input_length"],
        output_length=DEFAULT_CONFIG["output_length"],
    )

    return preprocessor, train_loader, val_loader, test_loader, n_features


# ============================================
# SECTION 3: EVALUATION HELPER
# ============================================
def run_evaluation(model, test_loader, preprocessor, evaluator, label="Ensemble"):
    """
    Run model over test_loader and return metrics dict.
    Inverse-transforms predictions back to MW scale before computing metrics.
    """
    device = next(model.parameters()).device
    all_preds, all_targets = [], []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Inverse-transform to original MW scale for meaningful metrics
    y_true_mw = preprocessor.inverse_transform_target(all_targets.flatten())
    y_pred_mw = preprocessor.inverse_transform_target(all_preds.flatten())

    metrics = evaluator.calculate_metrics(y_true_mw, y_pred_mw)

    logger.info(
        f"[{label}] MAPE={metrics['mape']:.2f}%  RMSE={metrics['rmse']:.2f}  "
        f"R²={metrics['r2']:.4f}  Accuracy={metrics['accuracy']:.2f}%"
    )
    return metrics


# ============================================
# SECTION 4: PRINT REPORT
# ============================================
def print_report(metrics: dict, label: str = "ACCURACY REPORT"):
    print("\n" + "=" * 55)
    print(label)
    print("-" * 55)
    print(f"  MAPE              : {metrics['mape']:.2f}%")
    print(f"  RMSE              : {metrics['rmse']:.2f} MW")
    print(f"  MAE               : {metrics['mae']:.2f} MW")
    print(f"  R² Score          : {metrics['r2']:.4f}")
    print(f"  Combined Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  Simple Accuracy   : {metrics['simple_accuracy']:.2f}%")

    evaluator = ModelEvaluator()
    passed, msg = evaluator.check_accuracy_threshold(metrics, threshold=85.0)
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  Status            : {status}")
    print(f"  Detail            : {msg}")
    print("=" * 55)


# ============================================
# SECTION 5: SAVE METRICS JSON
# ============================================
def save_metrics_json(metrics: dict, path: str = "models/metrics.json"):
    """
    Save metrics to JSON so app.py can load them without PyTorch.
    FIX: app.py checks models/metrics.json first — this ensures it is always written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")


# ============================================
# SECTION 6: MAIN — FULL TRAINING
# ============================================
def train_for_accuracy():
    """Full training pipeline (up to 100 epochs, early stopping)."""

    df = load_or_fetch_data()
    preprocessor, train_loader, val_loader, test_loader, n_features = preprocess(df)

    # Build ensemble model
    model = EnsembleForecaster(
        input_length=DEFAULT_CONFIG["input_length"],
        output_length=DEFAULT_CONFIG["output_length"],
        n_features=n_features,
    )

    trainer = ModelTrainer(model, learning_rate=DEFAULT_CONFIG["learning_rate"])
    evaluator = ModelEvaluator()

    logger.info("Starting high-accuracy training (up to 100 epochs, patience=15)...")
    result = trainer.train(
        train_loader,
        val_loader,
        epochs=100,
        patience=15,
        save_path="models/ensemble_best.pt",
    )
    logger.info(
        f"Training finished in {result['epochs_trained']} epochs. "
        f"Best val loss: {result['best_val_loss']:.6f}"
    )

    # ── Evaluate individual sub-models ──────────────────────────────────────
    device = trainer.device

    # NHiTS sub-model
    nhits_preds, itrans_preds, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            nhits_preds.append(model.nhits(X_batch[:, :, 0].unsqueeze(-1)).cpu().numpy())
            itrans_preds.append(model.itransformer(X_batch).cpu().numpy())
            targets.append(y_batch.numpy())

    nhits_preds = np.concatenate(nhits_preds)
    itrans_preds = np.concatenate(itrans_preds)
    targets = np.concatenate(targets)

    y_true_mw = preprocessor.inverse_transform_target(targets.flatten())

    nhits_metrics = evaluator.calculate_metrics(
        y_true_mw,
        preprocessor.inverse_transform_target(nhits_preds.flatten()),
    )
    itrans_metrics = evaluator.calculate_metrics(
        y_true_mw,
        preprocessor.inverse_transform_target(itrans_preds.flatten()),
    )

    print("\n" + "=" * 55)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("-" * 55)
    print(f"  NHiTS accuracy       : {nhits_metrics['accuracy']:.2f}%")
    print(f"  iTransformer accuracy: {itrans_metrics['accuracy']:.2f}%")

    # ── Evaluate full ensemble ───────────────────────────────────────────────
    ensemble_metrics = run_evaluation(
        model, test_loader, preprocessor, evaluator, label="Ensemble"
    )
    print_report(ensemble_metrics, "ENSEMBLE — ACCURACY REPORT (HIGH ACCURACY TRAINING)")

    # ── Save models + metrics ────────────────────────────────────────────────
    ckpt = ModelCheckpoint(save_dir="models/")
    ckpt.save_model(model, "ensemble_model.pt", {"metrics": ensemble_metrics})
    ckpt.save_model(model.nhits, "nhits_model.pt", {"type": "nhits"})
    ckpt.save_model(model.itransformer, "itransformer_model.pt", {"type": "itransformer"})

    # FIX: also write metrics.json so app.py can read accuracy without PyTorch
    save_metrics_json(ensemble_metrics)

    logger.info("All models and metrics saved successfully.")


# ============================================
# SECTION 7: EVAL-ONLY MODE (replaces evaluate_models.py)
# ============================================
def evaluate_only(epochs: int = 5):
    """
    Quick sanity-check: train for a small number of epochs then evaluate.
    Replaces the old evaluate_models.py.
    """
    df = load_or_fetch_data()
    preprocessor, train_loader, val_loader, test_loader, n_features = preprocess(df)

    model = EnsembleForecaster(
        input_length=DEFAULT_CONFIG["input_length"],
        output_length=DEFAULT_CONFIG["output_length"],
        n_features=n_features,
    )

    trainer = ModelTrainer(model, learning_rate=DEFAULT_CONFIG["learning_rate"])
    evaluator = ModelEvaluator()

    logger.info(f"Quick evaluation: training for {epochs} epochs...")
    trainer.train(train_loader, val_loader, epochs=epochs, patience=3)

    metrics = run_evaluation(model, test_loader, preprocessor, evaluator, label="Quick Eval")
    print_report(metrics, "ACCURACY REPORT (QUICK EVAL)")

    # Save metrics so app.py can display them even after an eval-only run
    save_metrics_json(metrics)


# ============================================
# SECTION 8: ENTRY POINT
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate electricity demand forecasting models."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip full training; run a quick 5-epoch evaluation instead.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=5,
        help="Number of epochs to use in --eval-only mode (default: 5).",
    )
    args = parser.parse_args()

    if args.eval_only:
        evaluate_only(epochs=args.eval_epochs)
    else:
        train_for_accuracy()
