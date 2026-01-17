"""
Classified FxNLMS Wrapper

Wraps FxNLMS with noise classification for automatic parameter selection.
The original FxNLMS implementation is used unchanged.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.fxlms import FxNLMS
from src.ml.phase2_classifier.spectrogram import extract_mel_spectrogram
from src.ml.phase2_classifier.noise_classifier import NoiseClassifier
from src.ml.phase2_classifier.parameter_lookup import get_params, FxNLMSParams


class ClassifiedFxNLMS:
    """
    FxNLMS with automatic noise classification and parameter selection.

    This wrapper:
    1. Classifies the noise type from the reference signal
    2. Looks up optimal parameters (step_size, filter_length)
    3. Creates FxNLMS with those parameters
    4. Delegates all processing to original FxNLMS

    Usage:
        anc = ClassifiedFxNLMS(
            secondary_path_estimate=S_hat,
            classifier_path='output/models/phase2/noise_classifier.pt'
        )

        # Initialize with 1 second of reference signal
        noise_class, params = anc.initialize(reference_signal[:16000])

        # Use like normal FxNLMS
        for x, d in samples:
            y = anc.generate_antinoise(x)
            anc.filter_reference(x)
            anc.update_weights(e)
    """

    def __init__(
        self,
        secondary_path_estimate: np.ndarray,
        classifier_path: Optional[Union[str, Path]] = None,
        fallback_class: str = 'highway',
        regularization: float = 1e-6,
        fs: int = 16000
    ):
        """
        Initialize classified FxNLMS.

        Args:
            secondary_path_estimate: Estimated secondary path impulse response
            classifier_path: Path to trained classifier model
            fallback_class: Class to use if classifier not loaded
            regularization: Regularization constant
            fs: Sample rate in Hz
        """
        self.secondary_path_estimate = secondary_path_estimate
        self.fallback_class = fallback_class
        self.regularization = regularization
        self.fs = fs

        # Load classifier if path provided
        self.classifier: Optional[NoiseClassifier] = None
        if classifier_path is not None:
            classifier_path = Path(classifier_path)
            if classifier_path.exists():
                self.classifier = NoiseClassifier.load(classifier_path)
                print(f"Loaded noise classifier from {classifier_path}")
            else:
                print(f"Warning: Classifier not found at {classifier_path}, using fallback")

        # FxNLMS will be created when initialize() is called
        self.fxnlms: Optional[FxNLMS] = None
        self.noise_class: Optional[str] = None
        self.selected_params: Optional[FxNLMSParams] = None
        self.mel_spectrogram: Optional[np.ndarray] = None

    def initialize(
        self,
        reference_signal: np.ndarray
    ) -> Tuple[str, FxNLMSParams]:
        """
        Initialize by classifying noise and selecting parameters.

        Should be called with ~1 second of reference signal.

        Args:
            reference_signal: Initial reference signal for classification

        Returns:
            Tuple of (noise_class, selected_params)
        """
        # Extract mel spectrogram
        self.mel_spectrogram = extract_mel_spectrogram(
            reference_signal,
            fs=self.fs,
            target_shape=(64, 32)
        )

        # Classify noise type
        if self.classifier is not None:
            self.noise_class = self.classifier.predict(self.mel_spectrogram)
        else:
            self.noise_class = self.fallback_class

        # Get optimal parameters
        self.selected_params = get_params(self.noise_class)

        # Resize secondary path estimate if needed
        s_hat = self.secondary_path_estimate
        if len(s_hat) < self.selected_params.filter_length:
            # Pad with zeros
            s_hat = np.pad(s_hat, (0, self.selected_params.filter_length - len(s_hat)))
        elif len(s_hat) > self.selected_params.filter_length:
            # Truncate
            s_hat = s_hat[:self.selected_params.filter_length]

        # Create FxNLMS with selected parameters
        self.fxnlms = FxNLMS(
            filter_length=self.selected_params.filter_length,
            step_size=self.selected_params.step_size,
            secondary_path_estimate=s_hat,
            regularization=self.regularization
        )

        return self.noise_class, self.selected_params

    def generate_antinoise(self, x: float) -> float:
        """Generate anti-noise signal."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing")
        return self.fxnlms.generate_antinoise(x)

    def filter_reference(self, x: float) -> float:
        """Filter reference signal through secondary path estimate."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing")
        return self.fxnlms.filter_reference(x)

    def update_weights(self, e: float) -> None:
        """Update adaptive filter weights."""
        if self.fxnlms is None:
            raise RuntimeError("Must call initialize() before processing")
        self.fxnlms.update_weights(e)

    def get_weights(self) -> np.ndarray:
        """Get current adaptive filter weights."""
        if self.fxnlms is None:
            return np.zeros(256)
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
        self.noise_class = None
        self.selected_params = None
        self.mel_spectrogram = None

    @property
    def mse_history(self) -> List[float]:
        """Get MSE history."""
        if self.fxnlms is None:
            return []
        return self.fxnlms.mse_history

    def get_info(self) -> dict:
        """Get information about the current configuration."""
        return {
            'noise_class': self.noise_class,
            'step_size': self.selected_params.step_size if self.selected_params else None,
            'filter_length': self.selected_params.filter_length if self.selected_params else None,
            'classifier_loaded': self.classifier is not None,
            'initialized': self.fxnlms is not None,
        }
