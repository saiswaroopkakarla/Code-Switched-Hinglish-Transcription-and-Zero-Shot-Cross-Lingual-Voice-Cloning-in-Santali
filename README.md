# Speech Understanding — Programming Assignment 2
## Code-Switched STT → Santali Voice Cloning Pipeline

**Roll No:** `[YOUR_ROLL_NO]`  
**GitHub:** `[YOUR_GITHUB_LINK]`

---

## Overview

End-to-end pipeline that:
1. Transcribes Hinglish (code-switched) lecture audio with constrained Whisper
2. Converts transcript to IPA and translates to **Santali** (target LRL)
3. Synthesises the Santali lecture in the student's voice via zero-shot voice cloning
4. Evaluates robustness against spoofing and adversarial attacks

---

## Directory Structure

```
PA2/
├── pipeline.py               ← Main orchestrator (run this)
├── part1/
│   ├── lid.py                ← Multi-Head Frame-Level LID (Wav2Vec2 + MHA)
│   ├── constrained_decode.py ← Whisper + N-gram Logit Bias
│   └── denoiser.py           ← Spectral Subtraction
├── part2/
│   ├── ipa_converter.py      ← Hinglish → IPA (custom G2P)
│   └── translator.py         ← Hinglish → Santali (500-word corpus)
├── part3/
│   ├── voice_embedding.py    ← x-vector / d-vector extraction
│   ├── prosody_warp.py       ← F0 + Energy + DTW warping
│   └── synthesizer.py        ← VITS / Meta MMS TTS
├── part4/
│   ├── anti_spoof.py         ← LFCC/CQCC CM + EER
│   └── adversarial.py        ← FGSM on LID
├── utils/
│   ├── audio_utils.py        ← Audio I/O, features
│   └── metrics.py            ← WER, MCD, EER
├── ngram_lm/
│   └── build_ngram.py        ← Build N-gram LM from syllabus
├── santali_corpus/           ← Generated technical dictionary
└── outputs/                  ← All output files
```

---

## Setup

```bash
# Create environment
conda create -n su_pa2 python=3.10
conda activate su_pa2

# Install dependencies
pip install -r requirements.txt

# Install espeak-ng for phonemizer (Linux)
sudo apt-get install espeak-ng

# Install ffmpeg (for audio conversion)
sudo apt-get install ffmpeg

# Install pyworld (for F0 extraction)
pip install pyworld --break-system-packages
```

---

## Usage

### Step 1: Prepare Audio

Download your lecture video, convert to WAV, extract 10-minute segment:
```bash
# Convert video to audio
ffmpeg -i lecture_video.mp4 -ac 1 -ar 16000 lecture_full.wav

# (Or let the pipeline handle it directly with the mp4)
```

Record your 60-second reference voice:
```bash
# On Linux with arecord:
arecord -d 60 -r 16000 -c 1 -f S16_LE student_voice_ref.wav
```

### Step 2: Build N-gram LM

```bash
python ngram_lm/build_ngram.py
```

### Step 3: Run Full Pipeline

```bash
python pipeline.py \
    --lecture_audio  original_segment.wav \
    --student_voice  student_voice_ref.wav \
    --output_dir     outputs/ \
    --start_sec      0 \
    --end_sec        600
```

### Step 4: Run with Meta MMS (recommended for Santali)

```bash
python pipeline.py \
    --lecture_audio  original_segment.wav \
    --student_voice  student_voice_ref.wav \
    --use_mms \
    --output_dir     outputs/
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/original_segment.wav` | Source lecture snippet (10 min) |
| `outputs/denoised_segment.wav` | After spectral subtraction |
| `outputs/transcript.txt` | Hinglish transcript (Whisper + N-gram bias) |
| `outputs/transcript_ipa.txt` | IPA representation |
| `outputs/santali_transcript.txt` | Santali translation |
| `outputs/output_LRL_cloned.wav` | Final 10-min Santali lecture (cloned voice) |
| `outputs/synthesis_flat.wav` | Ablation: flat synthesis (no prosody) |
| `outputs/adversarial_sample.wav` | FGSM adversarial sample |
| `outputs/santali_tech_dict.csv` | 500-word technical dictionary |
| `outputs/lid_result.json` | Frame-level LID output + timestamps |
| `outputs/results.json` | All evaluation metrics |

---

## Evaluation Metrics (Passing Criteria)

| Metric | Criterion | Implementation |
|--------|-----------|----------------|
| WER (English) | < 15% | `utils/metrics.py::compute_wer` |
| WER (Hindi) | < 25% | `utils/metrics.py::compute_wer` |
| MCD | < 8.0 | `utils/metrics.py::compute_mcd` |
| LID Switch Accuracy | Within 200ms | `utils/metrics.py::switching_timestamp_accuracy` |
| EER (Anti-Spoof) | < 10% | `utils/metrics.py::compute_eer` |
| Min ε (FGSM) | SNR > 40dB | `part4/adversarial.py` |

---

## Architecture Notes

### Part I: LID
- **Wav2Vec2-base** feature extractor (frozen except last 2 layers)
- 4-head self-attention encoder (2 layers, hidden=256)
- Frame-level classifier: 768 → 256 → 128 → 2
- Frame resolution: ~20ms (Wav2Vec2 stride = 320 samples @ 16kHz)
- Training: Mozilla Common Voice Hindi + English splits

### Part I: Constrained Decoding
- **Whisper-large-v3** beam search (n=5)
- N-gram LM (n=3, Kneser-Ney smoothed) trained on Speech Course Syllabus
- Logit bias: `adjusted_logit(w) = logit_whisper(w) + λ * log P_ngram(w | context)`
- λ = 2.0 (tunable), technical term boost = 3.0

### Part II: IPA Conversion
- Custom Hinglish G2P layer handles retroflex, aspirated, dental sounds
- Word-level language detection + phonemizer (espeak-ng) for English words
- 500-word Santali technical dictionary with IPA annotations

### Part III: Prosody
- WORLD vocoder for F0 + SP + AP extraction
- DTW alignment: O(T_ref × T_syn) dynamic programming
- Re-synthesis with warped F0 preserves teaching cadence

### Part IV: Spoofing
- LFCC (60-dim, linear filter bank) — distinguishes natural spectral irregularities
- Bi-LSTM with attention pooling → binary score
- EER evaluated with threshold sweep

---

## References

1. Radford et al. (2022). *Whisper: Robust Speech Recognition via Large-Scale Weak Supervision*
2. Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
3. Kong et al. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*
4. Kim et al. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)*
5. Boll (1979). *Suppression of Acoustic Noise in Speech Using Spectral Subtraction*
6. Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples (FGSM)*
7. Todisco et al. (2019). *ASVspoof 2019: Future Horizons in Spoofed and Fake Speech Detection*
8. Desplanques et al. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN*
9. Martin (2001). *Noise Power Spectral Density Estimation Based on Optimal Smoothing*
10. Prakash & Jyothi (2021). *Investigating Hinglish Code-Switching in ASR*
