
import os
import pandas as pd
import numpy as np
import torch
import logging
from model_trainer import (
    NHiTSModel, 
    iTransformer, 
    EnsembleForecaster, 
    ModelTrainer, 
    ModelEvaluator, 
    DataPreprocessor,
    ModelCheckpoint,
    DEFAULT_CONFIG
)

# Setup simple logging to stdout
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("TrainHighAccuracy")

def train_for_accuracy():
    # 1. Load data
    # 1. Load data
    if not os.path.exists('actual_demand.csv'):
        from actual import fetch_actual_data
        logger.info("Fetching real historical demand data...")
        df = fetch_actual_data()
        if not df.empty:
            df.to_csv('actual_demand.csv', index=False)
        else:
            logger.error("Failed to fetch real data.")
            return
    else:
        logger.info("Loading existing actual_demand.csv...")
        df = pd.read_csv('actual_demand.csv')
    
    logger.info(f"Loaded {len(df)} rows of data")
    
    # 2. Preprocess
    # Ensure datetime is index and only numeric columns are features
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Only keep numeric columns
    df = df.select_dtypes(include=[np.number])
    
    preprocessor = DataPreprocessor()
    target_col = 'demand_mw'
    preprocessor.fit_scalers(df, target_col)
    
    X, y = preprocessor.create_sequences(
        df, 
        input_length=DEFAULT_CONFIG['input_length'], 
        output_length=DEFAULT_CONFIG['output_length'],
        target_col=target_col
    )
    
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        X, y, 
        batch_size=DEFAULT_CONFIG['batch_size'],
        input_length=DEFAULT_CONFIG['input_length'],
        output_length=DEFAULT_CONFIG['output_length']
    )
    
    # 3. Model Initialization (Ensemble)
    # Filter features (matching how model_trainer does it)
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)
    
    logger.info(f"Features: {feature_cols}")
    
    model = EnsembleForecaster(
        input_length=DEFAULT_CONFIG['input_length'],
        output_length=DEFAULT_CONFIG['output_length'],
        n_features=n_features
    )
    
    # 4. Training (50 epochs)
    logger.info("Starting high-accuracy training (up to 100 epochs)...")
    trainer = ModelTrainer(model, learning_rate=DEFAULT_CONFIG['learning_rate'])
    evaluator = ModelEvaluator()
    
    result = trainer.train(
        train_loader, 
        val_loader, 
        epochs=100, 
        patience=15,
        save_path="models/ensemble_best.pt"
    )
    
    logger.info(f"Training complete in {result['epochs_trained']} epochs")
    
    # 5. Final Evaluation on Test Set
    logger.info("Final evaluation on unseen test set...")
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(trainer.device)
            output = model(X_batch)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Inverse transform to get original scale for metrics
    y_true_orig = preprocessor.inverse_transform_target(all_targets.flatten())
    y_pred_orig = preprocessor.inverse_transform_target(all_preds.flatten())
    
    metrics = evaluator.calculate_metrics(y_true_orig, y_pred_orig)
    
    # Evaluate individual models
    model.eval()
    with torch.no_grad():
        nhits_preds = []
        itrans_preds = []
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(trainer.device)
            # NHITS only uses first feature
            nhits_preds.append(model.nhits(X_batch[:,:,0].unsqueeze(-1)).cpu().numpy())
            itrans_preds.append(model.itransformer(X_batch).cpu().numpy())
            
    nhits_preds = np.concatenate(nhits_preds)
    itrans_preds = np.concatenate(itrans_preds)
    
    nhits_metrics = evaluator.calculate_metrics(y_true_orig, preprocessor.inverse_transform_target(nhits_preds.flatten()))
    itrans_metrics = evaluator.calculate_metrics(y_true_orig, preprocessor.inverse_transform_target(itrans_preds.flatten()))

    print("\n" + "="*50)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("-" * 50)
    print(f"NHITS accuracy: {nhits_metrics['accuracy']:.2f}%")
    print(f"iTransformer accuracy: {itrans_metrics['accuracy']:.2f}%")
    print("="*50)

    print("\n" + "="*50)
    print("ACCURACY REPORT (HIGH ACCURACY TRAINING)")
    print("="*50)
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"Combined Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Simple Accuracy: {metrics['simple_accuracy']:.2f}%")
    
    pass_status, msg = evaluator.check_accuracy_threshold(metrics, threshold=85.0)
    print(f"Status: {'CORRECT' if pass_status else 'INCORRECT'}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Message: {msg}")
    print("="*50)
    
    # Save the final model properly for app.py
    checkpoint = ModelCheckpoint(save_dir="models/")
    checkpoint.save_model(model, "ensemble_model.pt", {"metrics": metrics})
    
    # Also save individual models if needed
    checkpoint.save_model(model.nhits, "nhits_model.pt", {"type": "nhits"})
    checkpoint.save_model(model.itransformer, "itransformer_model.pt", {"type": "itransformer"})

if __name__ == "__main__":
    train_for_accuracy()
