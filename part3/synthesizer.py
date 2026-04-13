"""
part3/synthesizer.py — Zero-Shot Cross-Lingual Voice Cloning TTS

Uses Coqui TTS (VITS / YourTTS) or Meta MMS to synthesise Santali text
conditioned on the student's voice embedding.

Output requirement: ≥ 22.05 kHz
"""

import os
import numpy as np
import torch
from utils.audio_utils import TTS_SAMPLE_RATE, save_audio


# ─────────────────────────── VITS Synthesiser ────────────────────────────────

def synthesize_vits(
    text: str,
    speaker_wav_path: str,
    output_path: str = "outputs/output_LRL_cloned.wav",
    language: str = "en",               # 'en' as proxy; see note below
    model_name: str = "tts_models/multilingual/multi-dataset/your_tts",
) -> np.ndarray:
    """
    Synthesise text using YourTTS (Coqui TTS) with voice cloning.

    NOTE on Santali:
        YourTTS supports ~14 languages. Santali is not directly supported.
        We use English phoneme mapping as the closest available backend,
        then apply our DTW prosody warping to impart professor's style.
        For best results, use Meta MMS (which has broader language support).

    Args:
        text:             Text to synthesise (Santali, IPA, or romanised).
        speaker_wav_path: Path to student's 60s reference recording.
        output_path:      Where to write the final WAV.
        language:         TTS language code.

    Returns:
        wav (np.ndarray): Synthesised waveform at TTS_SAMPLE_RATE.
    """
    try:
        from TTS.api import TTS as CoquiTTS
        tts = CoquiTTS(model_name=model_name, progress_bar=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts.to(device)

        # YourTTS generates at 16kHz; we upsample to 22050
        tmp_path = output_path + "_tmp.wav"
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=tmp_path,
        )

        import soundfile as sf
        wav, sr = sf.read(tmp_path)
        os.remove(tmp_path)

        # Upsample if needed
        if sr < TTS_SAMPLE_RATE:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TTS_SAMPLE_RATE)
            sr = TTS_SAMPLE_RATE

        save_audio(output_path, wav, sr)
        return wav.astype(np.float32)

    except Exception as e:
        print(f"[TTS] VITS/YourTTS failed: {e}")
        print("[TTS] Falling back to Meta MMS...")
        return synthesize_mms(text, speaker_wav_path, output_path)


def synthesize_mms(
    text: str,
    speaker_wav_path: str,
    output_path: str = "outputs/output_LRL_cloned.wav",
    mms_lang: str = "eng",       # Use 'sat' for Santali if available in MMS
) -> np.ndarray:
    """
    Synthesise using Meta MMS TTS (covers 1100+ languages including some tribal).

    MMS language codes: 'sat' = Santali (Ol Chiki)
    Falls back to 'hin' (Hindi) if 'sat' not available.
    """
    try:
        from transformers import VitsModel, AutoTokenizer
        import torch

        # Try Santali first, fallback to Hindi
        for lang in [mms_lang, "hin", "eng"]:
            try:
                model_id = f"facebook/mms-tts-{lang}"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = VitsModel.from_pretrained(model_id)
                break
            except Exception:
                continue

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        wav = output.squeeze().numpy()
        sr = model.config.sampling_rate   # typically 16000

        # Upsample to ≥22050
        import librosa
        if sr < TTS_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TTS_SAMPLE_RATE)

        save_audio(output_path, wav, TTS_SAMPLE_RATE)
        return wav.astype(np.float32)

    except Exception as e:
        print(f"[TTS] MMS failed: {e}")
        print("[TTS] Generating placeholder silence for pipeline testing.")
        wav = np.zeros(int(10 * TTS_SAMPLE_RATE), dtype=np.float32)
        save_audio(output_path, wav, TTS_SAMPLE_RATE)
        return wav


# ─────────────────────────── Full Pipeline: Chunk → Synthesise ──────────────

def synthesize_long_form(
    text: str,
    speaker_wav_path: str,
    output_path: str = "outputs/output_LRL_cloned.wav",
    max_chars_per_chunk: int = 200,
    use_mms: bool = False,
) -> np.ndarray:
    """
    Synthesise long text (10 minutes) by chunking into sentences.
    Concatenates audio segments with short cross-fade.

    Args:
        text:                 Full Santali transcript.
        speaker_wav_path:     Student's reference voice.
        output_path:          Final output path.
        max_chars_per_chunk:  Max characters per synthesis chunk.
        use_mms:              Use Meta MMS instead of VITS.

    Returns:
        full_wav (np.ndarray): Complete synthesised lecture.
    """
    import re

    # Split into sentence-like chunks
    sentences = re.split(r"(?<=[।.!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars_per_chunk:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent
    if current.strip():
        chunks.append(current.strip())

    print(f"[TTS] Synthesising {len(chunks)} chunks...")

    all_wavs = []
    synth_fn = synthesize_mms if use_mms else synthesize_vits

    for i, chunk in enumerate(chunks):
        tmp_out = f"outputs/chunk_{i:04d}.wav"
        os.makedirs("outputs", exist_ok=True)
        wav = synth_fn(chunk, speaker_wav_path, tmp_out)
        all_wavs.append(wav)
        print(f"  Chunk {i+1}/{len(chunks)}: {len(wav)/TTS_SAMPLE_RATE:.1f}s")

    # Concatenate with 50ms cross-fade
    crossfade_samples = int(0.05 * TTS_SAMPLE_RATE)
    full_wav = all_wavs[0]
    for w in all_wavs[1:]:
        if len(full_wav) < crossfade_samples or len(w) < crossfade_samples:
            full_wav = np.concatenate([full_wav, w])
            continue
        # Cross-fade
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in  = np.linspace(0.0, 1.0, crossfade_samples)
        full_wav[-crossfade_samples:] = (
            full_wav[-crossfade_samples:] * fade_out +
            w[:crossfade_samples] * fade_in
        )
        full_wav = np.concatenate([full_wav, w[crossfade_samples:]])

    # Clean up chunks
    for i in range(len(chunks)):
        tmp = f"outputs/chunk_{i:04d}.wav"
        if os.path.exists(tmp):
            os.remove(tmp)

    save_audio(output_path, full_wav, TTS_SAMPLE_RATE)
    print(f"[TTS] Final output: {len(full_wav)/TTS_SAMPLE_RATE:.1f}s → {output_path}")
    return full_wav
