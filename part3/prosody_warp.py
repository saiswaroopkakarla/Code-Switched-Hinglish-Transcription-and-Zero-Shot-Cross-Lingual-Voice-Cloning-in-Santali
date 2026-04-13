"""
part3/prosody_warp.py — Prosody Feature Extraction + DTW Warping

Goal:
    Extract F0 (fundamental frequency) and Energy contours from the professor's
    lecture, then warp synthesised LRL speech to match the "teaching style".

Method:
    1. Extract F0 using WORLD vocoder (pyworld) or CREPE
    2. Extract energy as RMS per frame
    3. Apply DTW to align synthesised prosody to professor prosody
    4. Apply PSOLA-style pitch modification to synthesised waveform

Mathematical basis:
    DTW cost:   C(i,j) = |f_ref(i) - f_syn(j)|
    Warping path W* = argmin_{W} sum_{(i,j)∈W} C(i,j)
    Modified F0: f_out(t) = f_ref(W*(t))
"""

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from utils.audio_utils import SAMPLE_RATE, TTS_SAMPLE_RATE


# ─────────────────────────── F0 Extraction ──────────────────────────────────

def extract_f0_pyworld(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_period: float = 5.0,     # ms
    f0_floor: float = 65.0,        # Hz (lower bound for male voice)
    f0_ceil: float = 1100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract F0, spectral envelope, aperiodicity using WORLD vocoder.
    Returns: (f0, sp, ap)
    """
    try:
        import pyworld as pw
        wav_d = wav.astype(np.float64)
        f0, sp, ap = pw.wav2world(
            wav_d, sr,
            frame_period=frame_period,
        )
        return f0, sp, ap
    except ImportError:
        print("[Prosody] pyworld not available; using librosa pyin")
        return extract_f0_librosa(wav, sr)


def extract_f0_librosa(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = 256,
    fmin: float = 65.0,
    fmax: float = 1100.0,
) -> tuple[np.ndarray, None, None]:
    """Fallback F0 extraction using librosa pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        wav,
        fmin=fmin, fmax=fmax,
        sr=sr, hop_length=hop_length,
    )
    f0 = np.where(np.isnan(f0), 0.0, f0)
    return f0, None, None


def extract_energy(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_len: int = 512,
    hop_len: int = 256,
) -> np.ndarray:
    """RMS energy per frame."""
    frames = librosa.util.frame(wav, frame_length=frame_len, hop_length=hop_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    return rms


def extract_prosody(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> dict:
    """Extract full prosodic features from a waveform."""
    f0, sp, ap = extract_f0_pyworld(wav, sr)
    energy = extract_energy(wav, sr)

    # Smooth F0 (replace zeros with interpolated values for DTW)
    f0_voiced = f0.copy()
    voiced_mask = f0 > 0
    if voiced_mask.sum() > 2:
        x_voiced = np.where(voiced_mask)[0]
        y_voiced = f0[voiced_mask]
        interp = interp1d(x_voiced, y_voiced, bounds_error=False, fill_value=(y_voiced[0], y_voiced[-1]))
        f0_interp = interp(np.arange(len(f0)))
    else:
        f0_interp = f0_voiced

    # Smooth with median filter
    f0_smooth = medfilt(f0_interp, kernel_size=5)

    return {
        "f0":       f0,
        "f0_smooth": f0_smooth,
        "f0_interp": f0_interp,
        "sp":       sp,
        "ap":       ap,
        "energy":   energy,
        "voiced_mask": voiced_mask,
    }


# ─────────────────────────── DTW Warping ────────────────────────────────────

def dtw_align(
    ref_contour: np.ndarray,
    syn_contour: np.ndarray,
) -> tuple[np.ndarray, list]:
    """
    DTW alignment between reference and synthesised prosody contours.

    Args:
        ref_contour: (T_ref,) reference F0/energy
        syn_contour: (T_syn,) synthesised F0/energy

    Returns:
        warped_contour: (T_syn,) syn contour warped to match ref
        path:           list of (i, j) index pairs
    """
    try:
        from dtaidistance import dtw_ndim
        # Reshape to 2D for dtaidistance
        ref_2d = ref_contour.reshape(-1, 1).astype(np.float64)
        syn_2d = syn_contour.reshape(-1, 1).astype(np.float64)
        path = dtw_ndim.warping_path(ref_2d, syn_2d)
    except ImportError:
        # Manual DTW implementation
        path = _manual_dtw(ref_contour, syn_contour)

    # Build mapping: for each syn index j, find corresponding ref index i
    syn_to_ref = {}
    for i, j in path:
        if j not in syn_to_ref:
            syn_to_ref[j] = []
        syn_to_ref[j].append(i)

    # Warped: for each syn frame, use mean of mapped ref frames
    warped = np.zeros(len(syn_contour))
    for j in range(len(syn_contour)):
        if j in syn_to_ref:
            warped[j] = np.mean(ref_contour[syn_to_ref[j]])
        else:
            warped[j] = syn_contour[j]

    return warped, path


def _manual_dtw(x: np.ndarray, y: np.ndarray) -> list:
    """Pure Python DTW path computation."""
    N, M = len(x), len(y)
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c = abs(float(x[i - 1]) - float(y[j - 1]))
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack
    path = []
    i, j = N, M
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
        choice = np.argmin(choices)
        if choice == 0:
            i -= 1; j -= 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


# ─────────────────────────── Apply Warped Prosody ───────────────────────────

def apply_prosody_warping(
    syn_wav: np.ndarray,
    ref_wav: np.ndarray,
    sr: int = TTS_SAMPLE_RATE,
    frame_period: float = 5.0,
) -> np.ndarray:
    """
    Warp synthesised speech to match reference (professor's) prosody.

    Steps:
        1. Extract F0 + SP + AP from synthesised speech
        2. Extract F0 from reference speech
        3. DTW-align reference F0 to synthesised F0
        4. Re-synthesise with WORLD using warped F0

    Args:
        syn_wav:     Synthesised LRL waveform.
        ref_wav:     Reference professor lecture waveform.
        sr:          Sample rate of both waveforms.
        frame_period: WORLD analysis frame period (ms).

    Returns:
        warped_wav (np.ndarray): Waveform with professor's prosodic style.
    """
    try:
        import pyworld as pw
    except ImportError:
        print("[Prosody] pyworld not available; returning unmodified synthesis.")
        return syn_wav

    # Extract prosody from both
    ref_prosody = extract_prosody(ref_wav, sr)
    syn_prosody = extract_prosody(syn_wav, sr)

    ref_f0 = ref_prosody["f0_smooth"]
    syn_f0 = syn_prosody["f0"]
    syn_sp = syn_prosody["sp"]
    syn_ap = syn_prosody["ap"]

    if syn_sp is None:
        print("[Prosody] WORLD not available; returning unmodified synthesis.")
        return syn_wav

    # DTW-align ref F0 to syn F0 length
    warped_f0, path = dtw_align(ref_f0, syn_f0)

    # Preserve voicing mask from synthesised speech
    voiced_mask = syn_f0 > 0
    warped_f0_final = np.where(voiced_mask, warped_f0, 0.0)
    warped_f0_final = np.maximum(warped_f0_final, 0.0)

    # Re-synthesise with WORLD
    warped_wav = pw.synthesize(
        warped_f0_final.astype(np.float64),
        syn_sp.astype(np.float64),
        syn_ap.astype(np.float64),
        sr,
        frame_period=frame_period,
    )

    # Also warp energy
    ref_energy = ref_prosody["energy"]
    syn_energy = syn_prosody["energy"]
    warped_energy, _ = dtw_align(ref_energy, syn_energy)
    # Energy scaling per-frame
    hop = 256
    for i in range(min(len(warped_energy), len(syn_energy))):
        start = i * hop
        end = min(start + hop * 2, len(warped_wav))
        if end > start and syn_energy[i] > 1e-6:
            scale = warped_energy[i] / (syn_energy[i] + 1e-8)
            scale = np.clip(scale, 0.3, 3.0)
            warped_wav[start:end] *= scale

    print(f"[Prosody] Warping complete. Output: {len(warped_wav)/sr:.1f}s")
    return warped_wav.astype(np.float32)


def ablation_flat_synthesis(syn_wav: np.ndarray) -> np.ndarray:
    """
    Flat synthesis baseline (no prosody warping) for ablation study.
    Simply returns the unmodified synthesis output.
    """
    return syn_wav
