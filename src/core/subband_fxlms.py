"""
Subband FxLMS Adaptive Filter

Runs K independent full-capacity FxNLMS filters, one per frequency band.
Each band gets:
  - Its own analysis bandpass (splits reference into the band)
  - A band-limited secondary path estimate
  - Full filter_length taps (not divided)
  - Optionally, per-band error filtering
Anti-noise outputs are summed across bands at the speaker.
"""

import numpy as np

from .fxlms import FxNLMS
from .filters import BandpassPreFilter


class SubbandFxLMS:
    """
    Subband FxLMS with independent full-capacity filters per frequency band.
    """

    def __init__(
        self,
        num_bands: int,
        freq_range: tuple,
        filter_length: int,
        step_size: float,
        secondary_path_estimate: np.ndarray,
        sample_rate: float,
        regularization: float = 1e-4,
        filter_error: bool = False
    ):
        self.num_bands = num_bands
        self.fs = sample_rate
        self.low_freq, self.high_freq = freq_range
        self.filter_length = filter_length
        self.filter_error = filter_error

        band_width = (self.high_freq - self.low_freq) / num_bands

        self.analysis_filters = []
        self.error_filters = []
        self.band_fxlms = []
        self.band_edges = []

        for k in range(num_bands):
            low = self.low_freq + k * band_width
            high = self.low_freq + (k + 1) * band_width
            self.band_edges.append((low, high))

            self.analysis_filters.append(
                BandpassPreFilter(low, high, sample_rate, order=3)
            )

            if filter_error:
                self.error_filters.append(
                    BandpassPreFilter(low, high, sample_rate, order=3)
                )

            bp_sec = BandpassPreFilter(low, high, sample_rate, order=3)
            s_hat_band = np.array([
                bp_sec.filter_sample(s) for s in secondary_path_estimate
            ])

            self.band_fxlms.append(FxNLMS(
                filter_length=filter_length,
                step_size=step_size,
                secondary_path_estimate=s_hat_band,
                regularization=regularization
            ))

        self.mse_history = []
        self._band_x = [0.0] * num_bands

    def generate_antinoise(self, x: float) -> float:
        y_total = 0.0
        for k in range(self.num_bands):
            x_band = self.analysis_filters[k].filter_sample(x)
            self._band_x[k] = x_band
            y_total += self.band_fxlms[k].generate_antinoise(x_band)
        return y_total

    def filter_reference(self, x: float) -> float:
        for k in range(self.num_bands):
            self.band_fxlms[k].filter_reference(self._band_x[k])
        return 0.0

    def update_weights(self, e: float) -> None:
        for k in range(self.num_bands):
            if self.filter_error:
                e_k = self.error_filters[k].filter_sample(e)
            else:
                e_k = e
            self.band_fxlms[k].update_weights(e_k)
        self.mse_history.append(e ** 2)

    def reset(self):
        for k in range(self.num_bands):
            self.analysis_filters[k].reset()
            if self.filter_error:
                self.error_filters[k].reset()
            self.band_fxlms[k].reset()
        self.mse_history = []
        self._band_x = [0.0] * self.num_bands

    @property
    def weights(self):
        return np.concatenate([f.weights for f in self.band_fxlms])
