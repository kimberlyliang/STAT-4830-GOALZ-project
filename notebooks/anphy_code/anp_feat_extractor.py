import numpy as np
from scipy.signal import butter, filtfilt, welch

class ANPFeatExtractor:
    def __init__(self, fs, lowcut=0.5, highcut=40, order=4, nperseg=1024):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.nperseg = nperseg
        # Frequency bands in Hz
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30)
        }

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data):
        b, a = self.butter_bandpass()
        return filtfilt(b, a, data)

    def compute_psd(self, data):
        f, Pxx = welch(data, self.fs, nperseg=self.nperseg)
        return f, Pxx

    def compute_relative_band_power(self, data):
        # data: 1D array for one channel
        filtered = self.bandpass_filter(data)
        f, Pxx = self.compute_psd(filtered)
        total_idx = (f >= self.lowcut) & (f <= self.highcut)
        total_power = np.trapz(Pxx[total_idx], f[total_idx])
        rel_power = {}
        for band, (low, high) in self.bands.items():
            idx = (f >= low) & (f < high)
            band_power = np.trapz(Pxx[idx], f[idx])
            rel_power[band] = band_power / total_power if total_power > 0 else 0
        return rel_power

    def extract_features(self, epoch_data):
        # epoch_data shape: (n_channels, n_samples)
        features = []
        for ch in epoch_data:
            rel_power = self.compute_relative_band_power(ch)
            # Order: delta, theta, alpha, beta
            features.append([rel_power['delta'], rel_power['theta'], rel_power['alpha'], rel_power['beta']])
        return np.array(features)  # shape: (n_channels, 4)
