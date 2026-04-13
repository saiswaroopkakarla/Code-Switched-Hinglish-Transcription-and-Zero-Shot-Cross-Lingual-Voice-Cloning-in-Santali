"""
part1/denoiser.py — Spectral Subtraction for classroom noise/reverb removal.

Method:
  1. Estimate noise PSD from a silent/noise-only segment (first 0.5s).
  2. Subtract scaled noise spectrum from signal spectrum.
  3. Apply half-wave rectification (floor at 0) and Wiener-post-filter.
  
Reference: Boll (1979) "Suppression of acoustic noise in speech using spectral subtraction"
"""

import numpy as np
import librosa
import soundfile as sf
from utils.audio_utils import SAMPLE_RATE


def spectral_subtraction(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_estimate_frames: int = 10,   # frames used for noise PSD estimate
    over_subtraction: float = 1.5,     # α — controls aggressiveness
    spectral_floor: float = 0.002,     # β — prevents musical noise
    wiener_smoothing: bool = True,
) -> np.ndarray:
    """
    Spectral subtraction denoiser.

    Args:
        wav:                    Input waveform (float32, mono).
        sr:                     Sample rate.
        n_fft:                  FFT size.
        hop_length:             Hop size.
        noise_estimate_frames:  Number of initial frames for noise PSD estimate.
        over_subtraction:       Over-subtraction factor α (1.0–2.0).
        spectral_floor:         Spectral floor β to prevent musical noise.
        wiener_smoothing:       Apply Wiener post-filter.

    Returns:
        clean_wav (np.ndarray): Denoised waveform.
    """
    # STFT
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)          # (F, T)
    phase = np.angle(D)            # (F, T)

    # ── Step 1: Noise PSD estimation from initial silence ──────────────────
    # Use first `noise_estimate_frames` as noise reference
    noise_frames = magnitude[:, :noise_estimate_frames]
    noise_psd = np.mean(noise_frames ** 2, axis=1, keepdims=True)   # (F, 1)

    # ── Step 2: Spectral subtraction ────────────────────────────────────────
    signal_psd = magnitude ** 2    # (F, T)

    # P_clean = P_signal - α * P_noise
    clean_psd = signal_psd - over_subtraction * noise_psd
    # Half-wave rectify with spectral floor
    clean_psd = np.maximum(clean_psd, spectral_floor * signal_psd)

    clean_magnitude = np.sqrt(clean_psd)

    # ── Step 3: Wiener post-filter ──────────────────────────────────────────
    if wiener_smoothing:
        gain = clean_magnitude / (magnitude + 1e-8)
        # Smooth gain across time (3-frame moving average)
        for i in range(1, gain.shape[1] - 1):
            gain[:, i] = (gain[:, i - 1] + gain[:, i] + gain[:, i + 1]) / 3.0
        clean_magnitude = magnitude * gain

    # ── Step 4: Reconstruct waveform ────────────────────────────────────────
    D_clean = clean_magnitude * np.exp(1j * phase)
    clean_wav = librosa.istft(D_clean, hop_length=hop_length, length=len(wav))

    return clean_wav.astype(np.float32)


def adaptive_noise_estimation(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = 512,
    hop_length: int = 128,
    frame_update_rate: float = 0.1,
) -> np.ndarray:
    """
    Adaptive variant: continuously updates noise estimate using minimum
    statistics (Martin, 2001). Better for non-stationary classroom noise.
    """
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    n_frames = magnitude.shape[1]

    noise_est = magnitude[:, 0:1].copy()   # Init from first frame
    clean_mag = np.zeros_like(magnitude)

    for t in range(n_frames):
        sig = magnitude[:, t:t+1]
        # Minimum statistics: noise estimate = running min of smoothed PSD
        noise_est = np.minimum(noise_est, sig) * (1 - frame_update_rate) + \
                    sig * frame_update_rate
        sub = sig ** 2 - 1.5 * noise_est ** 2
        sub = np.maximum(sub, 0.002 * sig ** 2)
        clean_mag[:, t:t+1] = np.sqrt(sub)

    D_clean = clean_mag * np.exp(1j * phase)
    clean_wav = librosa.istft(D_clean, hop_length=hop_length, length=len(wav))
    return clean_wav.astype(np.float32)


def denoise_file(input_path: str, output_path: str, adaptive: bool = False):
    """Convenience wrapper: load → denoise → save."""
    import soundfile as sf
    wav, sr = sf.read(input_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)          # Stereo → mono
    if adaptive:
        clean = adaptive_noise_estimation(wav, sr)
    else:
        clean = spectral_subtraction(wav, sr)
    sf.write(output_path, clean, sr)
    print(f"[Denoiser] {input_path} → {output_path}")
    return clean, sr
