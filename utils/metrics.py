"""
metrics.py — WER, MCD, EER evaluation helpers.
"""

import numpy as np
from jiwer import wer as _wer
import librosa


# ─────────────────────────── WER ────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate using jiwer."""
    return _wer(reference.lower(), hypothesis.lower())


# ─────────────────────────── MCD ────────────────────────────────────────────

def compute_mcd(ref_wav: np.ndarray, syn_wav: np.ndarray,
                sr: int = 22050, n_mfcc: int = 24) -> float:
    """
    Mel-Cepstral Distortion between reference and synthesized waveforms.
    MCD = (10 / ln(10)) * sqrt(2 * sum((c_ref - c_syn)^2))
    Lower is better; passing criterion: MCD < 8.0
    """
    def get_mcep(wav):
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc + 1)
        return mfcc[1:]         # Drop C0 (energy term)

    ref_mcep = get_mcep(ref_wav).T    # (T_ref, n_mfcc)
    syn_mcep = get_mcep(syn_wav).T    # (T_syn, n_mfcc)

    # DTW-align lengths for fair comparison
    from dtaidistance import dtw_ndim
    path = dtw_ndim.warping_path(ref_mcep, syn_mcep)
    diffs = []
    for i, j in path:
        diff = ref_mcep[i] - syn_mcep[j]
        diffs.append(np.sum(diff ** 2))

    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diffs))
    return mcd


# ─────────────────────────── EER ────────────────────────────────────────────

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """
    Equal Error Rate for anti-spoofing.
    bonafide_scores: higher score = more bonafide
    spoof_scores:    lower score = more spoof
    Returns (eer_rate, eer_threshold)
    """
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(bonafide_scores)),
                              np.zeros(len(spoof_scores))])

    thresholds = np.sort(all_scores)
    fars, frrs = [], []

    for t in thresholds:
        pred = (all_scores >= t).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tn = np.sum((pred == 0) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))

        far = fp / (fp + tn + 1e-9)    # False Acceptance Rate
        frr = fn / (fn + tp + 1e-9)    # False Rejection Rate
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    # EER: point where FAR ≈ FRR
    diff = np.abs(fars - frrs)
    idx = np.argmin(diff)
    eer = (fars[idx] + frrs[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


# ─────────────────────────── LID Metrics ────────────────────────────────────

def compute_f1(y_true: list, y_pred: list, pos_label: int = 1) -> float:
    """Binary F1 score."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(f1)


def switching_timestamp_accuracy(true_switches: list[float],
                                  pred_switches: list[float],
                                  tolerance_ms: float = 200.0) -> float:
    """
    % of predicted switch timestamps within `tolerance_ms` of a true switch.
    Assignment requirement: within 200ms.
    """
    if not true_switches:
        return 1.0
    correct = 0
    for ps in pred_switches:
        if any(abs(ps - ts) * 1000 <= tolerance_ms for ts in true_switches):
            correct += 1
    return correct / len(true_switches)
