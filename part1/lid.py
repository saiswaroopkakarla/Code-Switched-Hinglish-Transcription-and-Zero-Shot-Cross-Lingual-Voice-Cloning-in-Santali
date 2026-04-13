"""
part1/lid.py — Multi-Head Frame-Level Language Identification (LID)

Architecture:
    Wav2Vec2 feature extractor (frozen) → Multi-Head Attention Encoder
    → Frame-level classifier (English / Hindi)

Training:
    - Uses Mozilla Common Voice / FLEURS for English + Hindi segments
    - Frame resolution: ~20ms (Wav2Vec2 output stride)
    - Target: F1 ≥ 0.85 per-frame on code-switched audio

Two modes:
    1. train_lid()   — fine-tune on labelled data
    2. infer_lid()   — run on raw audio, return per-frame labels + timestamps
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from utils.audio_utils import SAMPLE_RATE, to_tensor


# ─────────────────────────── Constants ──────────────────────────────────────

LANG_ENGLISH = 0
LANG_HINDI   = 1
FRAME_SHIFT_MS = 20          # Wav2Vec2 output stride ≈ 20ms
MODEL_CKPT = "facebook/wav2vec2-base"


# ─────────────────────────── Model ──────────────────────────────────────────

class MultiHeadLID(nn.Module):
    """
    Multi-head attention based frame-level LID.

    Wav2Vec2 base → Linear projection → N-head self-attention →
    Layer norm → Frame-wise binary classifier
    """

    def __init__(
        self,
        wav2vec_model_name: str = MODEL_CKPT,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Frozen feature extractor
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        wav2vec_dim = self.wav2vec.config.hidden_size    # 768 for base

        # Freeze all except last 2 transformer layers
        for name, param in self.wav2vec.named_parameters():
            if "encoder.layers.11" in name or "encoder.layers.10" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Projection
        self.proj = nn.Linear(wav2vec_dim, hidden_dim)

        # Multi-head self-attention layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_attn_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_attn_layers)
        ])

        # Frame-level classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, wav_tensor: torch.Tensor, attention_mask=None):
        """
        Args:
            wav_tensor: (B, T) raw waveform
            attention_mask: optional (B, T)
        Returns:
            logits: (B, T_frames, n_classes)
        """
        outputs = self.wav2vec(
            input_values=wav_tensor,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state    # (B, T_frames, 768)

        x = self.proj(hidden)                 # (B, T_frames, hidden_dim)

        # Multi-head self-attention with residual connections
        for attn, norm in zip(self.attn_layers, self.layer_norms):
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)

        logits = self.classifier(x)           # (B, T_frames, n_classes)
        return logits


# ─────────────────────────── Training ───────────────────────────────────────

class LIDDataset(torch.utils.data.Dataset):
    """
    Simple dataset of (wav_path, frame_labels) pairs.
    frame_labels: list of (start_sec, end_sec, lang_id) tuples
    """

    def __init__(self, samples: list, processor, max_len_sec: float = 10.0):
        self.samples = samples
        self.processor = processor
        self.max_len = int(max_len_sec * SAMPLE_RATE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import soundfile as sf
        wav_path, segment_labels = self.samples[idx]
        wav, _ = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav[:self.max_len].astype(np.float32)

        inputs = self.processor(
            wav, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        wav_tensor = inputs.input_values.squeeze(0)  # (T,)

        # Build frame labels
        n_frames = wav_tensor.shape[0] // 320   # Wav2Vec2 stride=320
        labels = torch.zeros(n_frames, dtype=torch.long)
        for start, end, lang in segment_labels:
            fs = int(start * 1000 / FRAME_SHIFT_MS)
            fe = int(end * 1000 / FRAME_SHIFT_MS)
            labels[fs:fe] = lang

        return wav_tensor, labels


def train_lid(
    train_samples: list,
    val_samples: list,
    save_path: str = "ngram_lm/lid_weights.pt",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
):
    """
    Fine-tune MultiHeadLID on labelled code-switched audio.

    train_samples / val_samples: list of (wav_path, [(start, end, lang), ...])
    """
    processor = Wav2Vec2Processor.from_pretrained(MODEL_CKPT)
    model = MultiHeadLID()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = LIDDataset(train_samples, processor)
    val_ds   = LIDDataset(val_samples, processor)

    def collate(batch):
        wavs, labs = zip(*batch)
        max_t = max(w.shape[0] for w in wavs)
        max_f = max(l.shape[0] for l in labs)
        wav_pad = torch.zeros(len(wavs), max_t)
        lab_pad = torch.full((len(labs), max_f), -100, dtype=torch.long)
        for i, (w, l) in enumerate(zip(wavs, labs)):
            wav_pad[i, :w.shape[0]] = w
            lab_pad[i, :l.shape[0]] = l
        return wav_pad, lab_pad

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader   = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for wav_batch, lab_batch in train_loader:
            wav_batch, lab_batch = wav_batch.to(device), lab_batch.to(device)
            logits = model(wav_batch)             # (B, T_f, 2)
            B, Tf, C = logits.shape
            loss = criterion(logits.view(B * Tf, C), lab_batch.view(B * Tf))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation F1
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for wav_batch, lab_batch in val_loader:
                wav_batch = wav_batch.to(device)
                logits = model(wav_batch)
                preds = logits.argmax(-1).cpu().numpy().flatten()
                labels = lab_batch.numpy().flatten()
                mask = labels != -100
                all_pred.extend(preds[mask].tolist())
                all_true.extend(labels[mask].tolist())

        from utils.metrics import compute_f1
        f1 = compute_f1(all_true, all_pred, pos_label=LANG_HINDI)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"Hindi-F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model  (F1={f1:.4f})")

    return model


# ─────────────────────────── Inference ──────────────────────────────────────

def load_lid_model(weights_path: str = "ngram_lm/lid_weights.pt") -> MultiHeadLID:
    model = MultiHeadLID()
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[LID] Loaded weights from {weights_path}")
    else:
        print("[LID] Warning: no saved weights found, using random init.")
    model.eval()
    return model


def infer_lid(
    wav: np.ndarray,
    model: MultiHeadLID,
    sr: int = SAMPLE_RATE,
    chunk_sec: float = 10.0,
    smooth_window: int = 5,
) -> dict:
    """
    Run frame-level LID on a waveform.

    Returns:
        {
          'frame_labels':   np.ndarray (n_frames,) — 0=EN, 1=HI
          'frame_probs':    np.ndarray (n_frames, 2)
          'switch_timestamps': list[float] — seconds where language changes
          'segments':       list[(start_sec, end_sec, lang_str)]
        }
    """
    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(MODEL_CKPT)
    device = next(model.parameters()).device

    # Process in chunks to handle long audio
    chunk_len = int(chunk_sec * sr)
    all_labels, all_probs = [], []

    for start in range(0, len(wav), chunk_len):
        chunk = wav[start: start + chunk_len]
        if len(chunk) < 400:     # Skip tiny trailing chunk
            continue
        inputs = processor(
            chunk, sampling_rate=sr, return_tensors="pt"
        )
        wav_t = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(wav_t)          # (1, T_f, 2)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        labels = probs.argmax(-1)

        all_probs.append(probs)
        all_labels.append(labels)

    frame_labels = np.concatenate(all_labels)
    frame_probs  = np.concatenate(all_probs)

    # Temporal smoothing (majority vote in window)
    from scipy.ndimage import uniform_filter1d
    smooth_labels = (uniform_filter1d(frame_labels.astype(float),
                                       size=smooth_window) > 0.5).astype(int)

    # Find switch timestamps
    switches = []
    for i in range(1, len(smooth_labels)):
        if smooth_labels[i] != smooth_labels[i - 1]:
            switches.append(i * FRAME_SHIFT_MS / 1000.0)

    # Build segments
    segments = []
    seg_start = 0.0
    for sw in switches:
        lang = "en" if smooth_labels[int(seg_start * 1000 / FRAME_SHIFT_MS)] == LANG_ENGLISH else "hi"
        segments.append((seg_start, sw, lang))
        seg_start = sw
    total_sec = len(smooth_labels) * FRAME_SHIFT_MS / 1000.0
    lang = "en" if smooth_labels[-1] == LANG_ENGLISH else "hi"
    segments.append((seg_start, total_sec, lang))

    return {
        "frame_labels":       smooth_labels,
        "frame_probs":        frame_probs,
        "switch_timestamps":  switches,
        "segments":           segments,
    }
