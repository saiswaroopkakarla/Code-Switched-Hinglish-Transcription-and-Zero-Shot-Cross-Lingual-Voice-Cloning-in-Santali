"""
part3/voice_embedding.py — Speaker Embedding Extraction

Extracts high-dimensional speaker embeddings from a 60-second reference recording.
Supports both:
  1. d-vector (GE2E style from torchaudio SpeakerNet)
  2. x-vector (ECAPA-TDNN via SpeechBrain)

The embedding is used to condition the TTS model for zero-shot voice cloning.
"""

import os
import numpy as np
import torch
import torchaudio
from utils.audio_utils import SAMPLE_RATE, load_audio


# ─────────────────────────── ECAPA-TDNN x-vector ────────────────────────────

def extract_xvector(
    wav_path: str,
    save_path: str = "ngram_lm/speaker_embedding.pt",
    segment_duration: float = 3.0,  # seconds per segment for robustness
) -> torch.Tensor:
    """
    Extract x-vector speaker embedding using SpeechBrain ECAPA-TDNN.

    Args:
        wav_path:          Path to reference audio (ideally 60s).
        save_path:         Where to save the embedding tensor.
        segment_duration:  Length of segments to average over.

    Returns:
        embedding: (1, D) normalised speaker embedding (D=192 for ECAPA-TDNN)
    """
    try:
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        # Segment-level embeddings for robustness
        seg_len = int(segment_duration * SAMPLE_RATE)
        embeddings = []
        for start in range(0, wav.shape[1] - seg_len, seg_len // 2):
            seg = wav[:, start: start + seg_len]
            with torch.no_grad():
                emb = classifier.encode_batch(seg)   # (1, 1, D)
            embeddings.append(emb.squeeze())

        if not embeddings:
            with torch.no_grad():
                embedding = classifier.encode_batch(wav).squeeze(0)
        else:
            embedding = torch.stack(embeddings).mean(0, keepdim=True)  # (1, D)
            # L2 normalise
            embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(embedding, save_path)
        print(f"[VoiceEmbed] x-vector shape: {embedding.shape} → saved to {save_path}")
        return embedding

    except ImportError:
        print("[VoiceEmbed] SpeechBrain not available; using fallback d-vector.")
        return extract_dvector(wav_path, save_path)


# ─────────────────────────── d-vector (GE2E) ────────────────────────────────

class DVectorNet(torch.nn.Module):
    """
    Lightweight LSTM-based d-vector network (GE2E style).
    Input: log-mel spectrogram (T, n_mels=40)
    Output: d-vector of dimension 256
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 256,
                 n_layers: int = 3, embed_dim: int = 256):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=0.1
        )
        self.linear = torch.nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_mels) → (B, embed_dim)"""
        out, _ = self.lstm(x)               # (B, T, H)
        out = out[:, -1, :]                 # Last frame: (B, H)
        embed = self.linear(out)            # (B, embed_dim)
        embed = embed / (embed.norm(dim=-1, keepdim=True) + 1e-8)
        return embed


def extract_dvector(
    wav_path: str,
    save_path: str = "ngram_lm/speaker_embedding.pt",
    n_mels: int = 40,
    segment_duration: float = 2.0,
    device: str = None,
) -> torch.Tensor:
    """
    Extract d-vector from reference recording using our LSTM encoder.
    
    If pretrained weights exist at 'ngram_lm/dvector_weights.pt', loads them.
    Otherwise uses random-weight encoder (still produces consistent embeddings
    for the same speaker if fine-tuned later).
    """
    import librosa
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DVectorNet(input_dim=n_mels).to(device)

    # Load pretrained d-vector weights if available
    dvec_weights = "ngram_lm/dvector_weights.pt"
    if os.path.exists(dvec_weights):
        model.load_state_dict(torch.load(dvec_weights, map_location=device))
        print("[DVec] Loaded pretrained d-vector weights.")
    model.eval()

    wav, sr = load_audio(wav_path, sr=SAMPLE_RATE)

    seg_len = int(segment_duration * sr)
    embeddings = []
    for start in range(0, len(wav) - seg_len, seg_len // 2):
        seg = wav[start: start + seg_len]
        mel = librosa.feature.melspectrogram(
            y=seg, sr=sr, n_mels=n_mels, n_fft=512, hop_length=128
        )
        mel_db = librosa.power_to_db(mel).T    # (T, n_mels)
        mel_t = torch.FloatTensor(mel_db).unsqueeze(0).to(device)  # (1, T, n_mels)

        with torch.no_grad():
            emb = model(mel_t)    # (1, 256)
        embeddings.append(emb.cpu())

    if not embeddings:
        # Whole file as one segment
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_mels=n_mels, n_fft=512, hop_length=128
        )
        mel_db = librosa.power_to_db(mel).T
        mel_t = torch.FloatTensor(mel_db).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(mel_t)
    else:
        embedding = torch.stack(embeddings).mean(0)

    embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(embedding, save_path)
    print(f"[DVec] d-vector shape: {embedding.shape} → saved to {save_path}")
    return embedding


# ─────────────────────────── Load saved embedding ───────────────────────────

def load_speaker_embedding(path: str = "ngram_lm/speaker_embedding.pt") -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No speaker embedding found at {path}. "
                                 "Run extract_xvector() first.")
    emb = torch.load(path, map_location="cpu")
    print(f"[VoiceEmbed] Loaded embedding shape: {emb.shape}")
    return emb
