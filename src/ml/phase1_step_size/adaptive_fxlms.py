"""
Adaptive FxNLMS Wrapper

Wraps the original FxNLMS with ML-based step size selection.
The original FxNLMS implementation is used unchanged.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.fxlms import FxNLMS
from src.ml.phase1_step_size.feature_extractor import extract_features
from src.ml.phase1_step_size.step_size_selector import StepSizeSelector


class AdaptiveFxNLMS:
    """
    FxNLMS with ML-based adaptive step size selection.

    This wrapper uses a trained neural network to select the optimal
    step size based on signal features, then delegates all processing
    to the original FxNLMS implementation.

    The original src/core/fxlms.py FxNLMS is used unchanged.

    Usage:
        # Create adaptive filter
        adaptive_anc = AdaptiveFxNLMS(
            filter_length=256,
            secondary_path_estimate=S_hat,
            model_path='output/models/phase1/step_selector.pt'
        )

        # Initialize with first second of reference signal
        mu = adaptive_anc.initialize(reference_signal[:16000])

        # Use like normal FxNLMS
        for x, d in samples:
            y = adaptive_anc.generate_antinoise(x)
            adaptive_anc.filter_reference(x)
            adaptive_anc.update_weights(e)
    """

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
        Initialize adaptive FxNLMS.

        Args:
            filter_length: Number of FIR filter taps
            secondary_path_estimate: Estimated secondary path impulse response
            model_path: Path to trained step size selector model
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
        self.selector: Optional[StepSizeSelector] = None
        if model_path is not None:
            model_path = Path(model_path)
            if model_path.exists():
                self.selector = StepSizeSelector.load(model_path)
                print(f"Loaded step size selector from {model_path}")
            else:
                print(f"Warning: Model not found at {model_path}, using fallback")

        # FxNLMS will be created when initialize() is called
        self.fxnlms: Optional[FxNLMS] = None
        self.selected_step_size: Optional[float] = None
        self.selected_features: Optional[np.ndarray] = None

    def initialize(self, reference_signal: np.ndarray) -> float:
        """
        Initialize the filter by selecting step size from signal features.

        Should be called with an initial segment of the reference signal
        (e.g., first second) before processing begins.

        Args:
            reference_signal: Initial reference signal segment for feature extraction

        Returns:
            Selected step size
        """
        # Extract features
        self.selected_features = extract_features(reference_signal, self.fs)

        # Select step size
        if self.selector is not None:
            self.selected_step_size = self.selector.predict(self.selected_features)
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
        """
        Generate anti-noise signal.

        Args:
            x: Current reference sample

        Returns:
            Anti-noise output y(n)
        """
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing")
        return self.fxnlms.generate_antinoise(x)

    def filter_reference(self, x: float) -> float:
        """
        Filter reference signal through secondary path estimate.

        Args:
            x: Current reference sample

        Returns:
            Filtered reference sample
        """
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing")
        return self.fxnlms.filter_reference(x)

    def update_weights(self, e: float) -> None:
        """
        Update adaptive filter weights.

        Args:
            e: Error signal sample
        """
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing")
        self.fxnlms.update_weights(e)

    def get_weights(self) -> np.ndarray:
        """Get current adaptive filter weights."""
        if self.fxnlms is None:
            return np.zeros(self.filter_length)
        return self.fxnlms.get_weights()

    def get_mse(self, window: int = 100) -> float:
        """Get recent mean squared error."""
        if self.fxnlms is None:
            return float('inf')
        return self.fxnlms.get_mse(window)

    def reset(self) -> None:
        """Reset filter state."""
        if self.fxnlms is not None:
            self.fxnlms.reset()
        self.selected_step_size = None
        self.selected_features = None

    @property
    def mse_history(self) -> List[float]:
        """Get MSE history."""
        if self.fxnlms is None:
            return []
        return self.fxnlms.mse_history

    def get_info(self) -> dict:
        """
        Get information about the current configuration.

        Returns:
            Dictionary with configuration details
        """
        return {
            'filter_length': self.filter_length,
            'selected_step_size': self.selected_step_size,
            'fallback_step_size': self.fallback_step_size,
            'model_loaded': self.selector is not None,
            'features': self.selected_features.tolist() if self.selected_features is not None else None,
            'initialized': self.fxnlms is not None,
        }


def create_adaptive_fxnlms(
    filter_length: int = 256,
    secondary_path_estimate: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> AdaptiveFxNLMS:
    """
    Factory function to create AdaptiveFxNLMS.

    Args:
        filter_length: Number of filter taps
        secondary_path_estimate: Secondary path impulse response
        model_path: Path to trained model
        **kwargs: Additional arguments passed to AdaptiveFxNLMS

    Returns:
        Configured AdaptiveFxNLMS instance
    """
    if secondary_path_estimate is None:
        # Create default secondary path (simple delay + attenuation)
        secondary_path_estimate = np.zeros(256)
        secondary_path_estimate[10] = 0.8  # 10 sample delay, 0.8 gain

    return AdaptiveFxNLMS(
        filter_length=filter_length,
        secondary_path_estimate=secondary_path_estimate,
        model_path=model_path,
        **kwargs
    )
