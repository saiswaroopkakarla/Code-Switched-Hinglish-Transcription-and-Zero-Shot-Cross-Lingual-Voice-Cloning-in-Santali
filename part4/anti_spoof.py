"""
part4/anti_spoof.py — Anti-Spoofing Countermeasure (CM) System

Implements a Countermeasure (CM) classifier based on LFCC/CQCC features
to distinguish between:
    - Bona Fide:  Real human voice (student reference)
    - Spoof:      Synthesised output from Part III (TTS cloned voice)

Architecture:
    LFCC features (60-dim) → LSTM Encoder → Dense → Sigmoid score
    Evaluated with Equal Error Rate (EER)

Reference: ASVspoof 2019 baseline CM system
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from utils.audio_utils import SAMPLE_RATE, compute_lfcc
from utils.metrics import compute_eer


# ─────────────────────────── CQCC Extraction ────────────────────────────────

def compute_cqcc(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_bins: int = 84,
    fmin: float = 32.7,
    n_cqcc: int = 60,
) -> np.ndarray:
    """
    Constant-Q Cepstral Coefficients (CQCC).
    Uses CQT → log → DCT pipeline.
    """
    cqt = np.abs(librosa.cqt(wav, sr=sr, n_bins=n_bins, fmin=fmin))
    log_cqt = np.log(cqt + 1e-8)
    from scipy.fftpack import dct
    cqcc = dct(log_cqt, axis=0, norm='ortho')[:n_cqcc]
    return cqcc   # (n_cqcc, T)


# ─────────────────────────── Model ──────────────────────────────────────────

class AntiSpoofCM(nn.Module):
    """
    LFCC/CQCC based anti-spoofing countermeasure.
    Input: feature sequence (B, T, n_feat)
    Output: spoof score (B,) — higher = more bona fide
    """

    def __init__(
        self,
        input_dim: int = 60,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns: (B,) bona fide score
        """
        lstm_out, _ = self.lstm(x)     # (B, T, H*2)
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)   # (B, T, 1)
        pooled = (lstm_out * attn_weights).sum(dim=1)                   # (B, H*2)
        score = self.classifier(pooled).squeeze(-1)                      # (B,)
        return score


# ─────────────────────────── Feature Extraction ─────────────────────────────

def extract_features(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    feature_type: str = "lfcc",
    n_feats: int = 60,
    max_frames: int = 300,
) -> np.ndarray:
    """
    Extract LFCC or CQCC features from a waveform.
    Returns: (T, n_feats) array, padded/truncated to max_frames
    """
    if feature_type == "lfcc":
        feat = compute_lfcc(wav, sr, n_lfcc=n_feats)           # (n_feats, T)
    elif feature_type == "cqcc":
        feat = compute_cqcc(wav, sr, n_cqcc=n_feats)           # (n_feats, T)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    feat = feat.T     # (T, n_feats)

    # Normalise (mean subtraction, variance normalisation)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)

    # Pad or truncate
    if feat.shape[0] < max_frames:
        pad = np.zeros((max_frames - feat.shape[0], feat.shape[1]))
        feat = np.vstack([feat, pad])
    else:
        feat = feat[:max_frames]

    return feat.astype(np.float32)


# ─────────────────────────── Training ───────────────────────────────────────

def train_anti_spoof(
    bonafide_wavs: list,    # list of wav file paths (real)
    spoof_wavs: list,       # list of wav file paths (synthesised)
    save_path: str = "ngram_lm/anti_spoof_weights.pt",
    feature_type: str = "lfcc",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 8,
) -> AntiSpoofCM:
    """
    Train the anti-spoofing CM on bona fide vs spoof audio.

    For Assignment 2, minimum dataset:
      - Bona fide: student's reference recording (60s) + augmented versions
      - Spoof:     TTS output from Part III + augmented versions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AntiSpoofCM(input_dim=60).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def load_features(paths, label):
        data = []
        for p in paths:
            wav, sr_read = sf.read(p)
            if wav.ndim > 1:
                wav = wav.mean(1)
            feat = extract_features(wav, sr_read, feature_type)
            data.append((feat, label))
        return data

    print("[CM] Loading features...")
    pos_data = load_features(bonafide_wavs, 1.0)
    neg_data = load_features(spoof_wavs, 0.0)
    all_data = pos_data + neg_data
    np.random.shuffle(all_data)

    feats = torch.FloatTensor(np.array([d[0] for d in all_data]))
    labels = torch.FloatTensor([d[1] for d in all_data])

    dataset = torch.utils.data.TensorDataset(feats, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for feat_batch, label_batch in loader:
            feat_batch, label_batch = feat_batch.to(device), label_batch.to(device)
            scores = model(feat_batch)
            loss = criterion(scores, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"[CM] Training complete. Best loss: {best_loss:.4f}")
    return model


# ─────────────────────────── Evaluation ─────────────────────────────────────

def evaluate_eer(
    model: AntiSpoofCM,
    bonafide_wavs: list,
    spoof_wavs: list,
    feature_type: str = "lfcc",
) -> tuple[float, float]:
    """
    Evaluate EER on test set.
    Returns (eer, threshold). Passing criterion: EER < 10%.
    """
    device = next(model.parameters()).device
    model.eval()

    def score_wavs(paths):
        scores = []
        for p in paths:
            wav, sr_read = sf.read(p)
            if wav.ndim > 1:
                wav = wav.mean(1)
            feat = extract_features(wav, sr_read, feature_type)
            feat_t = torch.FloatTensor(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                s = torch.sigmoid(model(feat_t)).item()
            scores.append(s)
        return np.array(scores)

    bonafide_scores = score_wavs(bonafide_wavs)
    spoof_scores    = score_wavs(spoof_wavs)

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    print(f"[CM] EER = {eer*100:.2f}%  (threshold={threshold:.4f})")
    print(f"     {'✓ PASS' if eer < 0.10 else '✗ FAIL'}  (criterion: EER < 10%)")
    return eer, threshold


def load_anti_spoof_model(
    weights_path: str = "ngram_lm/anti_spoof_weights.pt",
) -> AntiSpoofCM:
    model = AntiSpoofCM()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"[CM] Loaded weights from {weights_path}")
    model.eval()
    return model


# ─────────────────────────── Quick Test (no training data) ──────────────────

def demo_eer_with_noise(
    real_wav_path: str,
    synth_wav_path: str,
    n_augments: int = 20,
) -> tuple[float, float]:
    """
    Demo: create augmented versions of real and synth audio for EER test.
    Used when only one bona fide and one spoof file are available.
    """
    import tempfile

    real_wav, sr = sf.read(real_wav_path)
    syn_wav, _  = sf.read(synth_wav_path)
    if real_wav.ndim > 1: real_wav = real_wav.mean(1)
    if syn_wav.ndim > 1:  syn_wav  = syn_wav.mean(1)

    real_paths, spoof_paths = [], []
    tmp_dir = tempfile.mkdtemp()

    for i in range(n_augments):
        # Bona fide augmentation (add small Gaussian noise, pitch shift slightly)
        noise = np.random.normal(0, 0.002, real_wav.shape)
        aug_real = real_wav + noise
        p = os.path.join(tmp_dir, f"real_{i}.wav")
        sf.write(p, aug_real, sr)
        real_paths.append(p)

        # Spoof augmentation
        aug_spoof = syn_wav + np.random.normal(0, 0.001, syn_wav.shape)
        p2 = os.path.join(tmp_dir, f"spoof_{i}.wav")
        sf.write(p2, aug_spoof, sr)
        spoof_paths.append(p2)

    model = AntiSpoofCM()
    # Quick train
    model = train_anti_spoof(
        real_paths[:int(0.8 * n_augments)],
        spoof_paths[:int(0.8 * n_augments)],
        epochs=15,
    )
    eer, thr = evaluate_eer(
        model,
        real_paths[int(0.8 * n_augments):],
        spoof_paths[int(0.8 * n_augments):],
    )
    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)
    return eer, thr
