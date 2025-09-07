"""Model training pipeline.

This module trains machine learning models and stores artifacts in
``settings.artifacts_dir``. The following files are created for each
training run:

* ``model_{market}_{timestamp}.pkl`` - trained model
* ``metadata_{market}_{timestamp}.pkl`` - training metadata
* ``X_test_{market}_{timestamp}.pkl`` - holdout features
* ``y_test_{market}_{timestamp}.pkl`` - holdout labels
"""

import pickle
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb

from ..config import settings
from ..logging import get_logger
from ..utils.betting import american_to_prob, remove_vig

logger = get_logger(__name__)


class ModelTrainer:
    """Trainer for ML models."""

    def __init__(self):
        """Initialize model trainer."""
        self.artifacts_dir = settings.artifacts_dir
        self.artifacts_dir.mkdir(exist_ok=True)
        self.random_seed = settings.random_seed

    def prepare_training_data(
        self, features_df: pd.DataFrame, labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data by joining features and labels."""
        logger.info("Preparing training data")

        # Join features and labels
        training_data = features_df.merge(labels_df, on="game_id", how="inner")

        # Separate features and target
        feature_cols = [
            col
            for col in training_data.columns
            if col
            not in ["game_id", "result_cover", "result_over", "home_team", "away_team"]
        ]

        X = training_data[feature_cols]

        # Determine target column based on labels
        if "result_cover" in labels_df.columns:
            y = training_data["result_cover"]
        elif "result_over" in labels_df.columns:
            y = training_data["result_over"]
        else:
            raise ValueError("No valid target column found in labels")

        # Handle missing values
        X = X.fillna(0)

        logger.info(
            f"Prepared training data: {len(X)} samples, {len(feature_cols)} features"
        )
        return X, y

    def train_xgboost_model(
        self, X: pd.DataFrame, y: pd.Series, market: str, test_size: float = 0.2
    ) -> Dict:
        """Train XGBoost model with time-based split."""
        logger.info(f"Training XGBoost model for {market}")

        # Time-based split (assuming data is ordered by date)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.random_seed,
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 50,
        }

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info(f"Model training completed. ROC-AUC: {metrics['roc_auc']:.4f}")

        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred,
        }

    def save_model(self, model_results: Dict, market: str) -> None:
        """Save trained model, metadata, and test data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_filename = f"model_{market}_{timestamp}.pkl"
        model_path = self.artifacts_dir / model_filename

        with open(model_path, "wb") as f:
            pickle.dump(model_results["model"], f)

        # Save metadata
        metadata = {
            "market": market,
            "timestamp": timestamp,
            "metrics": model_results["metrics"],
            "feature_importance": model_results["feature_importance"].to_dict(),
            "model_path": str(model_path),
        }

        metadata_filename = f"metadata_{market}_{timestamp}.pkl"
        metadata_path = self.artifacts_dir / metadata_filename

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # Save test data
        X_test_filename = f"X_test_{market}_{timestamp}.pkl"
        X_test_path = self.artifacts_dir / X_test_filename
        with open(X_test_path, "wb") as f:
            pickle.dump(model_results["X_test"], f)

        y_test_filename = f"y_test_{market}_{timestamp}.pkl"
        y_test_path = self.artifacts_dir / y_test_filename
        with open(y_test_path, "wb") as f:
            pickle.dump(model_results["y_test"], f)

        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Saved test features to {X_test_path}")
        logger.info(f"Saved test labels to {y_test_path}")

    def train_market_model(self, market: str) -> Dict:
        """Train model for a specific market."""
        logger.info(f"Training model for market: {market}")

        # Load feature data
        feature_files = list(settings.data_dir.glob("gold/features_*.parquet"))
        if not feature_files:
            raise ValueError("No feature data found")

        latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
        features_df = pd.read_parquet(latest_features)

        # Load labels for the market
        labels_file = settings.data_dir / f"gold/labels_{market}.parquet"
        if not labels_file.exists():
            raise ValueError(f"No labels found for market: {market}")

        labels_df = pd.read_parquet(labels_file)

        # Prepare training data
        X, y = self.prepare_training_data(features_df, labels_df)

        # Train model
        model_results = self.train_xgboost_model(X, y, market)

        # Save model
        self.save_model(model_results, market)

        logger.info(f"Model training completed for {market}")
        return model_results
