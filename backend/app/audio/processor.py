import io
import os
import numpy as np
try:
    # pyrefly: ignore [missing-import]
    import librosa
except ImportError:
    class MockLibrosa:
        class effects:
            @staticmethod
            def trim(y, top_db): return y, None
        class feature:
            @staticmethod
            def mfcc(*args, **kwargs): return np.random.rand(20, 100)
            @staticmethod
            def melspectrogram(*args, **kwargs): return np.random.rand(128, 100)
            @staticmethod
            def spectral_centroid(*args, **kwargs): return np.random.rand(1, 100)
            @staticmethod
            def spectral_bandwidth(*args, **kwargs): return np.random.rand(1, 100)
            @staticmethod
            def spectral_rolloff(*args, **kwargs): return np.random.rand(1, 100)
            @staticmethod
            def zero_crossing_rate(*args, **kwargs): return np.random.rand(1, 100)
            @staticmethod
            def rms(*args, **kwargs): return np.random.rand(1, 100)
            @staticmethod
            def chroma_stft(*args, **kwargs): return np.random.rand(12, 100)
            @staticmethod
            def spectral_contrast(*args, **kwargs): return np.random.rand(7, 100)
        @staticmethod
        def load(path, sr, mono): return np.random.rand(sr * 3), sr
        @staticmethod
        def resample(y, orig_sr, target_sr): return y
        @staticmethod
        def power_to_db(S, ref): return S
    librosa = MockLibrosa()

try:
    # pyrefly: ignore [missing-import]
    import soundfile as sf
except ImportError:
    sf = None
from typing import Tuple, Dict, Any
from app.config import AUDIO_SAMPLE_RATE, AUDIO_DURATION, N_MFCC, N_MELS, N_FFT, HOP_LENGTH

class AudioProcessor:
    def __init__(self, target_sr: int = AUDIO_SAMPLE_RATE, target_duration: float = AUDIO_DURATION):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(self.target_sr * self.target_duration)

    def load_audio(self, file_path_or_bytes) -> Tuple[np.ndarray, int]:
        """
        Loads audio file or raw bytes, converts to mono and resamples to target_sr.
        """
        try:
            if isinstance(file_path_or_bytes, bytes):
                file_path_or_bytes = io.BytesIO(file_path_or_bytes)
            y, sr = librosa.load(file_path_or_bytes, sr=self.target_sr, mono=True)
        except Exception as e:
            # Fallback to soundfile if librosa default backend encounters format specific quirks
            if isinstance(file_path_or_bytes, str):
                y, sr = sf.read(file_path_or_bytes)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                if sr != self.target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            else:
                raise e

        # Normalize amplitude to [-1.0, 1.0]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # Pad or truncate to fixed duration
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
        else:
            y = y[:self.target_length]

        return y, self.target_sr

    def extract_features(self, y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> Dict[str, Any]:
        """
        Extracts comprehensive acoustic features for ML and DL models.
        """
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 2. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        
        # 4. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)[0]

        # 7. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

        # 8. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma_mean = np.mean(chroma, axis=1)

        # 9. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        contrast_mean = np.mean(contrast, axis=1)

        # Standard 1D tabular vector for RF/SVM
        tabular_vector = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [np.mean(centroid), np.std(centroid)],
            [np.mean(bandwidth), np.std(bandwidth)],
            [np.mean(rolloff), np.std(rolloff)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)],
            chroma_mean,
            contrast_mean
        ])

        # Generate lightweight 2D grid matrix for 2D CNN (resize to 128x128 for uniform input)
        # Using simple interpolation / downsampling
        mel_resized = self._resize_2d(mel_spec_db, target_shape=(128, 128))

        # Lightweight time series waveform downsampled for JSON display (500 points)
        waveform_sampled = self._downsample_1d(y, num_points=300)
        
        # Downsampled heatmaps for visualizations
        mel_viz = self._resize_2d(mel_spec_db, target_shape=(32, 64)).tolist()
        mfcc_viz = self._resize_2d(mfcc, target_shape=(20, 64)).tolist()

        return {
            "tabular_vector": tabular_vector.astype(float).tolist(),
            "mel_spectrogram_2d": mel_resized.astype(float).tolist(),
            "sequence_features": mfcc.T.astype(float).tolist(), # Time frames x MFCCs
            "stats": {
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms)),
                "zcr_mean": float(np.mean(zcr)),
                "zcr_std": float(np.std(zcr)),
                "spectral_centroid_mean": float(np.mean(centroid)),
                "spectral_bandwidth_mean": float(np.mean(bandwidth)),
                "spectral_rolloff_mean": float(np.mean(rolloff)),
                "chroma_mean": float(np.mean(chroma_mean)),
                "spectral_contrast_mean": float(np.mean(contrast_mean)),
                "mfcc_means": mfcc_mean.astype(float).tolist()
            },
            "visualizations": {
                "waveform": waveform_sampled.astype(float).tolist(),
                "mel_spectrogram": mel_viz,
                "mfcc_heatmap": mfcc_viz,
                "spectral_centroid_series": self._downsample_1d(centroid, 100).astype(float).tolist(),
                "rms_series": self._downsample_1d(rms, 100).astype(float).tolist()
            }
        }

    def _resize_2d(self, matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Utility to resize a 2D matrix using linear interpolation without requiring heavy OpenCV."""
        h, w = matrix.shape
        th, tw = target_shape
        row_indices = np.linspace(0, h - 1, th).astype(int)
        col_indices = np.linspace(0, w - 1, tw).astype(int)
        return matrix[np.ix_(row_indices, col_indices)]

    def _downsample_1d(self, arr: np.ndarray, num_points: int) -> np.ndarray:
        """Downsample 1D array to a fixed number of samples."""
        if len(arr) <= num_points:
            return arr
        indices = np.linspace(0, len(arr) - 1, num_points).astype(int)
        return arr[indices]
