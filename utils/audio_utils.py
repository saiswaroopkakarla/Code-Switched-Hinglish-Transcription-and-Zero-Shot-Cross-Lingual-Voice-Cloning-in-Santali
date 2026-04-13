"""
audio_utils.py — Common audio I/O and processing helpers.
"""

import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import librosa
import subprocess


SAMPLE_RATE = 16000          # Standard SR for ASR
TTS_SAMPLE_RATE = 22050      # Required by assignment (≥22.05 kHz)


# ─────────────────────────────── I/O ────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load any audio/video file as mono float32, resampled to `sr`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # ffmpeg handles mp4/mkv/webm/mp3/wav/flac, etc.
    tmp_wav = path + "_tmp.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(sr), tmp_wav],
        check=True, capture_output=True
    )
    wav, file_sr = sf.read(tmp_wav)
    os.remove(tmp_wav)
    return wav.astype(np.float32), file_sr


def save_audio(path: str, wav: np.ndarray, sr: int = TTS_SAMPLE_RATE):
    """Save float32 audio array to wav."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    sf.write(path, wav, sr)
    print(f"[Audio] Saved → {path}  ({len(wav)/sr:.1f}s @ {sr}Hz)")


def extract_segment(wav: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Slice a segment from a waveform."""
    return wav[int(start_sec * sr): int(end_sec * sr)]


def to_tensor(wav: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(wav).unsqueeze(0)          # (1, T)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().detach().cpu().numpy()


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)


# ─────────────────────────────── Features ───────────────────────────────────

def compute_stft(wav: np.ndarray, sr: int = SAMPLE_RATE,
                 n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Return power spectrogram (F, T)."""
    S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop)) ** 2
    return S


def compute_mel(wav: np.ndarray, sr: int = SAMPLE_RATE,
                n_mels: int = 80, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Return log-mel spectrogram (n_mels, T)."""
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels,
                                          n_fft=n_fft, hop_length=hop)
    return librosa.power_to_db(mel)


def compute_mfcc(wav: np.ndarray, sr: int = SAMPLE_RATE,
                 n_mfcc: int = 40) -> np.ndarray:
    """Return MFCC (n_mfcc, T)."""
    return librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)


def compute_lfcc(wav: np.ndarray, sr: int = SAMPLE_RATE,
                 n_filter: int = 70, n_lfcc: int = 60) -> np.ndarray:
    """
    Linear Frequency Cepstral Coefficients (LFCC).
    Uses a linearly-spaced filter bank instead of mel.
    """
    n_fft = 512
    hop = 128
    stft = compute_stft(wav, sr, n_fft=n_fft, hop=hop)    # (F, T)

    # Linear filter bank
    freqs = np.linspace(0, sr / 2, stft.shape[0])
    filter_freqs = np.linspace(0, sr / 2, n_filter + 2)
    fb = np.zeros((n_filter, stft.shape[0]))
    for m in range(1, n_filter + 1):
        for k in range(stft.shape[0]):
            f = freqs[k]
            if filter_freqs[m - 1] <= f < filter_freqs[m]:
                fb[m - 1, k] = (f - filter_freqs[m - 1]) / (filter_freqs[m] - filter_freqs[m - 1])
            elif filter_freqs[m] <= f <= filter_freqs[m + 1]:
                fb[m - 1, k] = (filter_freqs[m + 1] - f) / (filter_freqs[m + 1] - filter_freqs[m])

    lin_spec = fb @ stft                                   # (n_filter, T)
    lin_spec = np.log(lin_spec + 1e-8)
    lfcc = scipy_dct(lin_spec, axis=0, norm='ortho')[:n_lfcc]
    return lfcc


def scipy_dct(x, axis=0, norm='ortho'):
    from scipy.fftpack import dct
    return dct(x, axis=axis, norm=norm)
