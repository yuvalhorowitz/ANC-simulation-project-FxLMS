"""
3-Class Adaptive FxNLMS Wrapper

Wraps FxNLMS with 3-class step size selection (Low/Medium/High μ).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.fxlms import FxNLMS
from src.ml.phase1_step_size.feature_extractor import extract_features
from src.ml.phase1_step_size.step_size_selector_3class import ThreeClassStepSizeSelector


class ThreeClassAdaptiveFxNLMS:
    """
    FxNLMS with 3-class step size selection (Low/Medium/High μ).

    Maps 3 classes to step sizes:
    - Class 0 (Low μ): 0.004 - for CITY/HIGHWAY
    - Class 1 (Medium μ): 0.0085 - for transitional scenarios
    - Class 2 (High μ): 0.015 - for IDLE
    """

    # Class to step size mapping
    CLASS_TO_STEP_SIZE = {
        0: 0.004,   # Low μ (CITY/HIGHWAY)
        1: 0.0085,  # Medium μ (transitional)
        2: 0.015    # High μ (IDLE)
    }

    def __init__(
        self,
        filter_length: int,
        secondary_path_estimate: np.ndarray,
        model_path: Optional[Union[str, Path]] = None,
        fallback_step_size: float = 0.005,
        regularization: float = 1e-6,
        fs: int = 16000
    ):
        """
        Initialize 3-class adaptive FxNLMS.

        Args:
            filter_length: Number of FIR filter taps
            secondary_path_estimate: Estimated secondary path impulse response
            model_path: Path to trained 3-class step size selector model
            fallback_step_size: Step size to use if model is not loaded
            regularization: Regularization constant for NLMS
            fs: Sample rate in Hz (for feature extraction)
        """
        self.filter_length = filter_length
        self.secondary_path_estimate = secondary_path_estimate
        self.fallback_step_size = fallback_step_size
        self.regularization = regularization
        self.fs = fs

        # Load ML model if path provided
        self.selector: Optional[ThreeClassStepSizeSelector] = None
        if model_path is not None:
            model_path = Path(model_path)
            if model_path.exists():
                self.selector = ThreeClassStepSizeSelector.load(model_path)
                print(f"Loaded 3-class step size selector from {model_path}")
            else:
                print(f"Warning: Model path {model_path} not found, using fallback")

        # FxNLMS instance (created during initialization)
        self.fxnlms: Optional[FxNLMS] = None
        self.selected_step_size = fallback_step_size

    def initialize(self, reference_signal: np.ndarray) -> float:
        """
        Extract features and select step size.

        Args:
            reference_signal: Initial reference signal window (e.g., first 1 second)

        Returns:
            Selected step size μ
        """
        if self.selector is not None:
            # Extract features
            features = extract_features(reference_signal, self.fs)

            # Predict class
            class_idx = self.selector.predict(features)

            # Map to step size
            self.selected_step_size = self.CLASS_TO_STEP_SIZE[class_idx]
        else:
            self.selected_step_size = self.fallback_step_size

        # Create FxNLMS with selected step size
        self.fxnlms = FxNLMS(
            filter_length=self.filter_length,
            step_size=self.selected_step_size,
            secondary_path_estimate=self.secondary_path_estimate,
            regularization=self.regularization
        )

        return self.selected_step_size

    def generate_antinoise(self, x: float) -> float:
        """Generate anti-noise sample."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing samples")
        return self.fxnlms.generate_antinoise(x)

    def filter_reference(self, x: float) -> float:
        """Filter reference through secondary path estimate."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing samples")
        return self.fxnlms.filter_reference(x)

    def update_weights(self, e: float):
        """Update filter weights using error signal."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing samples")
        self.fxnlms.update_weights(e)

    def get_weights(self) -> np.ndarray:
        """Get current filter weights."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before accessing weights")
        return self.fxnlms.get_weights()

    def get_selected_step_size(self) -> float:
        """Get the step size that was selected."""
        return self.selected_step_size

    def get_mse_history(self):
        """Get MSE history for convergence analysis."""
        if self.fxnlms is None:
            return []
        return self.fxnlms.mse_history

    @property
    def mse_history(self):
        """Property for direct access to MSE history."""
        return self.get_mse_history()
