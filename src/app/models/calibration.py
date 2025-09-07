"""Model calibration pipeline.

Calibration artifacts and required test data are loaded from
``settings.artifacts_dir``. For each market and timestamp the following
files are used or created:

* ``model_{market}_{timestamp}.pkl`` - trained model
* ``X_test_{market}_{timestamp}.pkl`` - holdout features
* ``y_test_{market}_{timestamp}.pkl`` - holdout labels
* ``calibrator_{market}_{timestamp}.pkl`` - calibrated model
* ``calibration_curve_{market}_{timestamp}.png`` - calibration plot
"""

import pickle
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class ModelCalibrator:
    """Calibrator for ML model probabilities."""

    def __init__(self):
        """Initialize model calibrator."""
        self.artifacts_dir = settings.artifacts_dir
        self.artifacts_dir.mkdir(exist_ok=True)
        self.calibration_method = settings.calibration_method

    def calibrate_probabilities(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, method: str = "isotonic"
    ) -> Dict:
        """Calibrate model probabilities."""
        logger.info(f"Calibrating probabilities using {method} method")

        # Get uncalibrated probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Apply calibration
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_pred_proba, y_test)
            y_calibrated = calibrator.predict(y_pred_proba)
        elif method == "platt":
            calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrator.fit(X_test, y_test)
            y_calibrated = calibrator.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Calculate calibration metrics
        brier_score = brier_score_loss(y_test, y_calibrated)
        log_loss_score = log_loss(y_test, y_calibrated)

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_calibrated, n_bins=10
        )

        metrics = {
            "brier_score": brier_score,
            "log_loss": log_loss_score,
            "calibration_curve": {
                "fraction_of_positives": fraction_of_positives,
                "mean_predicted_value": mean_predicted_value,
            },
        }

        logger.info(f"Calibration completed. Brier score: {brier_score:.4f}")

        return {
            "calibrator": calibrator,
            "y_calibrated": y_calibrated,
            "metrics": metrics,
        }

    def plot_calibration_curve(
        self,
        y_test: pd.Series,
        y_pred_proba: np.ndarray,
        y_calibrated: np.ndarray,
        market: str,
    ) -> None:
        """Plot calibration curve."""
        logger.info(f"Creating calibration plot for {market}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Uncalibrated probabilities
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )

        ax1.plot(
            mean_predicted_value, fraction_of_positives, "s-", label="Uncalibrated"
        )
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_title(f"Calibration Curve - {market} (Uncalibrated)")
        ax1.legend()
        ax1.grid(True)

        # Calibrated probabilities
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
            y_test, y_calibrated, n_bins=10
        )

        ax2.plot(
            mean_predicted_value_cal,
            fraction_of_positives_cal,
            "s-",
            label="Calibrated",
        )
        ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax2.set_xlabel("Mean predicted probability")
        ax2.set_ylabel("Fraction of positives")
        ax2.set_title(f"Calibration Curve - {market} (Calibrated)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"calibration_curve_{market}_{timestamp}.png"
        plot_path = self.artifacts_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved calibration plot to {plot_path}")

    def save_calibrator(self, calibrator, market: str) -> None:
        """Save calibrated model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        calibrator_filename = f"calibrator_{market}_{timestamp}.pkl"
        calibrator_path = self.artifacts_dir / calibrator_filename

        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)

        logger.info(f"Saved calibrator to {calibrator_path}")

    def calibrate_market_model(self, market: str) -> Dict:
        """Calibrate model for a specific market."""
        logger.info(f"Calibrating model for market: {market}")

        # Load trained model
        model_files = list(self.artifacts_dir.glob(f"model_{market}_*.pkl"))
        if not model_files:
            raise ValueError(f"No trained model found for market: {market}")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        with open(latest_model, "rb") as f:
            model = pickle.load(f)

        # Load corresponding test data
        timestamp = latest_model.stem.split(f"model_{market}_")[-1]
        X_test_path = self.artifacts_dir / f"X_test_{market}_{timestamp}.pkl"
        y_test_path = self.artifacts_dir / f"y_test_{market}_{timestamp}.pkl"

        if not X_test_path.exists() or not y_test_path.exists():
            raise ValueError("Test data artifacts not found for calibration")

        with open(X_test_path, "rb") as f:
            X_test = pickle.load(f)

        with open(y_test_path, "rb") as f:
            y_test = pickle.load(f)

        # Calibrate probabilities
        calibration_results = self.calibrate_probabilities(
            model, X_test, y_test, self.calibration_method
        )

        # Create calibration plot
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        self.plot_calibration_curve(
            y_test, y_pred_proba, calibration_results["y_calibrated"], market
        )

        # Save calibrator
        self.save_calibrator(calibration_results["calibrator"], market)

        logger.info(f"Model calibration completed for {market}")
        return calibration_results
