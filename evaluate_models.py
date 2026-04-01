
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
    DEFAULT_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Evaluator")

def evaluate():
    # 1. Load data
    if not os.path.exists('synthetic_demand.csv'):
        from generate_data import generate_synthetic_data
        df = generate_synthetic_data(days=365)
        df.to_csv('synthetic_demand.csv', index=False)
    else:
        df = pd.read_csv('synthetic_demand.csv')
    
    logger.info(f"Loaded {len(df)} rows of data")
    
    # 2. Preprocess
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
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)
    
    model = EnsembleForecaster(
        input_length=DEFAULT_CONFIG['input_length'],
        output_length=DEFAULT_CONFIG['output_length'],
        n_features=n_features
    )
    
    trainer = ModelTrainer(model, learning_rate=DEFAULT_CONFIG['learning_rate'])
    
    # 4. Training (Short to verify accuracy)
    logger.info("Training model for 5 epochs to check initial accuracy...")
    trainer.train(train_loader, val_loader, epochs=5, patience=3)
    
    # 5. Evaluation
    logger.info("Evaluating on test set...")
    # Get test predictions
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
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(all_targets, all_preds)
    
    print("\n" + "="*50)
    print("ACCURACY REPORT")
    print("="*50)
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"Combined Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Simple Accuracy: {metrics['simple_accuracy']:.2f}%")
    
    pass_status, msg = evaluator.check_accuracy_threshold(metrics, threshold=85.0)
    print(f"Status: {'✅ PASSED' if pass_status else '❌ FAILED'}")
    print(f"Message: {msg}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
