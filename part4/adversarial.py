"""
part4/adversarial.py — FGSM Adversarial Attack on LID System

Goal:
    Find minimum perturbation ε such that:
      - The perturbation is inaudible (SNR > 40 dB)
      - The LID model misclassifies Hindi as English

Method: Fast Gradient Sign Method (FGSM)
    δ = ε * sign(∇_x L(f(x), y_true))
    x_adv = x + δ

    Search ε from small (1e-5) to large (1e-2) until misclassification occurs.
    Report minimum ε where SNR ≥ 40 dB.

Mathematical formulation:
    SNR (dB) = 10 * log10(E[x²] / E[δ²])
    Constraint: SNR > 40 dB  ⟹  E[δ²] < E[x²] / 10^4
"""

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from utils.audio_utils import SAMPLE_RATE


def compute_snr_db(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio between original and perturbed signal.
    SNR (dB) = 10 * log10(signal_power / noise_power)
    """
    noise = perturbed - original
    signal_power = np.mean(original ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * np.log10(signal_power / (noise_power + 1e-12))


def fgsm_attack(
    wav: np.ndarray,
    lid_model,
    processor,
    target_class: int = 0,     # 0 = English (we want to flip Hindi->English)
    epsilon: float = 1e-3,
    device: str = None,
) -> tuple[np.ndarray, float]:
    """
    Single-step FGSM: generate adversarial perturbation.

    Args:
        wav:          Input audio segment (5 seconds of Hindi speech).
        lid_model:    MultiHeadLID model.
        processor:    Wav2Vec2Processor.
        target_class: Class to misclassify towards (0=EN).
        epsilon:      Perturbation magnitude.

    Returns:
        adv_wav:  Perturbed waveform.
        snr:      SNR of the perturbation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lid_model.eval()

    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    wav_tensor = inputs.input_values.to(device)
    wav_tensor.requires_grad_(True)

    logits = lid_model(wav_tensor)                    # (1, T_f, 2)
    T_f = logits.shape[1]
    target = torch.full((1, T_f), target_class, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
    lid_model.zero_grad()
    loss.backward()

    # FGSM update
    grad_sign = wav_tensor.grad.sign()
    perturbation = (epsilon * grad_sign).detach().cpu().numpy().squeeze()
    adv_wav = wav + perturbation[:len(wav)]

    snr = compute_snr_db(wav, adv_wav)
    return adv_wav.astype(np.float32), snr


def find_minimum_epsilon(
    wav: np.ndarray,
    lid_model,
    processor,
    sr: int = SAMPLE_RATE,
    snr_threshold: float = 40.0,
    eps_range: tuple = (1e-5, 1e-1),
    n_steps: int = 30,
    device: str = None,
) -> dict:
    """
    Binary search for minimum ε that:
      (1) Causes LID to misclassify Hindi as English
      (2) Maintains SNR ≥ snr_threshold dB

    Returns:
        {
          'min_epsilon':    float
          'snr_at_epsilon': float
          'adv_wav':        np.ndarray
          'original_pred':  str
          'adv_pred':       str
          'success':        bool
        }
    """
    from transformers import Wav2Vec2Processor
    from part1.lid import infer_lid, LANG_ENGLISH, LANG_HINDI

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lid_model.to(device)

    # Get original prediction
    with torch.no_grad():
        orig_result = infer_lid(wav, lid_model, sr)
    orig_dominant = "hi" if np.mean(orig_result["frame_labels"]) > 0.5 else "en"
    print(f"[FGSM] Original dominant language: {orig_dominant}")

    eps_values = np.logspace(
        np.log10(eps_range[0]),
        np.log10(eps_range[1]),
        n_steps
    )

    min_eps = None
    best_adv_wav = None
    best_snr = None

    for eps in eps_values:
        adv_wav, snr = fgsm_attack(wav, lid_model, processor, epsilon=float(eps), device=device)

        if snr < snr_threshold:
            print(f"  ε={eps:.2e}  SNR={snr:.1f}dB  [SNR too low, stopping]")
            break

        # Check if LID is fooled
        with torch.no_grad():
            adv_result = infer_lid(adv_wav, lid_model, sr)
        adv_dominant = "hi" if np.mean(adv_result["frame_labels"]) > 0.5 else "en"

        fooled = (orig_dominant == "hi") and (adv_dominant == "en")
        print(f"  ε={eps:.2e}  SNR={snr:.1f}dB  pred={adv_dominant}  fooled={fooled}")

        if fooled and min_eps is None:
            min_eps = eps
            best_adv_wav = adv_wav
            best_snr = snr
            # Keep going to find minimum, but if SNR would drop, stop here

    if min_eps is None:
        print("[FGSM] No successful attack found within SNR constraint.")
        return {
            "min_epsilon": None, "snr_at_epsilon": None,
            "adv_wav": None, "original_pred": orig_dominant,
            "adv_pred": orig_dominant, "success": False,
        }

    print(f"\n[FGSM] Minimum ε = {min_eps:.2e}  SNR = {best_snr:.1f} dB")
    print(f"       Hindi misclassified as English  ✓")

    return {
        "min_epsilon":    float(min_eps),
        "snr_at_epsilon": float(best_snr),
        "adv_wav":        best_adv_wav,
        "original_pred":  orig_dominant,
        "adv_pred":       "en",
        "success":        True,
    }


def pgd_attack(
    wav: np.ndarray,
    lid_model,
    processor,
    epsilon: float = 5e-3,
    alpha: float = 5e-4,
    n_iter: int = 40,
    target_class: int = 0,
    device: str = None,
) -> np.ndarray:
    """
    Projected Gradient Descent (PGD) — stronger iterative attack.
    More reliable than single-step FGSM.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    wav_orig = inputs.input_values.to(device)

    adv_wav = wav_orig.clone().detach()
    adv_wav.requires_grad_(True)

    for step in range(n_iter):
        adv_wav = adv_wav.detach().requires_grad_(True)
        logits = lid_model(adv_wav)
        T_f = logits.shape[1]
        target = torch.full((1, T_f), target_class, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
        loss.backward()

        with torch.no_grad():
            adv_wav = adv_wav + alpha * adv_wav.grad.sign()
            delta = adv_wav - wav_orig
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_wav = wav_orig + delta

    adv_numpy = adv_wav.detach().cpu().numpy().squeeze()[:len(wav)]
    snr = compute_snr_db(wav, adv_numpy)
    print(f"[PGD] SNR = {snr:.1f} dB  (ε={epsilon})")
    return adv_numpy.astype(np.float32)


def save_adversarial_report(results: dict, path: str = "outputs/adversarial_report.txt"):
    """Save adversarial robustness report."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Adversarial Robustness Report ===\n\n")
        f.write(f"Attack: FGSM (Fast Gradient Sign Method)\n")
        f.write(f"Target: Misclassify Hindi -> English\n")
        f.write(f"SNR Constraint: > 40 dB (inaudible)\n\n")
        f.write(f"Original prediction: {results.get('original_pred', 'N/A')}\n")
        f.write(f"Adversarial prediction: {results.get('adv_pred', 'N/A')}\n")
        f.write(f"Attack success: {results.get('success', False)}\n")
        f.write(f"Minimum ε: {results.get('min_epsilon', 'N/A')}\n")
        f.write(f"SNR at ε: {results.get('snr_at_epsilon', 'N/A')} dB\n")
    print(f"[FGSM] Report saved -> {path}")


import os
