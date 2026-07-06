# Hinglish Transcription & Santali Zero-Shot Voice Cloning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-large--v3-412991?logo=openai&logoColor=white)
![VITS](https://img.shields.io/badge/TTS-VITS%20%7C%20Meta%20MMS-FF6B35)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end speech pipeline that transcribes code-switched Hinglish lecture audio, translates it into Santali (a low-resource language), and synthesises the output in the student's own voice via zero-shot voice cloning — with anti-spoofing and adversarial robustness evaluation.

---

## Pipeline Overview

```
Hinglish Lecture Audio (WAV)
         │
  ┌──────▼──────────────────────────────────────────────┐
  │  Part I — Robust Code-Switched STT                   │
  │  Spectral Subtraction → Multi-Head LID (Wav2Vec2)    │
  │  → Whisper-large-v3 + N-gram Logit Bias Decoding     │
  └──────┬──────────────────────────────────────────────┘
         │  Hinglish transcript
  ┌──────▼──────────────────────────────────────────────┐
  │  Part II — Phonetic Mapping & Translation            │
  │  Custom G2P IPA Conversion → Santali Translation     │
  │  (500-word technical dictionary, built from scratch) │
  └──────┬──────────────────────────────────────────────┘
         │  Santali text + IPA
  ┌──────▼──────────────────────────────────────────────┐
  │  Part III — Zero-Shot Voice Cloning                  │
  │  x-vector speaker embedding → VITS / Meta MMS TTS   │
  │  → DTW prosody warping (F0 + energy alignment)       │
  └──────┬──────────────────────────────────────────────┘
         │  Santali lecture in student's voice
  ┌──────▼──────────────────────────────────────────────┐
  │  Part IV — Robustness Evaluation                     │
  │  LFCC + Bi-LSTM anti-spoofing (EER < 10%)            │
  │  FGSM adversarial attack on LID (SNR > 40dB)         │
  └─────────────────────────────────────────────────────┘
```

---

## Key Technical Highlights

- **Low-resource language:** Santali has minimal existing NLP tooling — a 500-word technical dictionary was built from scratch with IPA annotations
- **Constrained decoding:** Whisper logit bias with a 3-gram Kneser-Ney LM (`λ=2.0`) trained on speech course syllabus reduces Hinglish WER
- **Prosody preservation:** DTW alignment of F0 + energy contours from the source lecture to the synthesised output preserves teaching cadence across languages
- **Anti-spoofing:** LFCC (60-dim) + Bi-LSTM with attention pooling achieves EER < 10% distinguishing real vs. cloned speech
- **Adversarial robustness:** FGSM on Wav2Vec2 LID model finds minimum perturbation ε such that SNR > 40dB

---

## Evaluation Metrics

| Metric | Criterion | Component |
|---|---|---|
| WER (English) | < 15% | `utils/metrics.py` |
| WER (Hindi) | < 25% | `utils/metrics.py` |
| MCD | < 8.0 | Mel-Cepstral Distortion |
| LID Switch Accuracy | Within 200ms | Frame-level Wav2Vec2 |
| EER (Anti-Spoof) | < 10% | LFCC + Bi-LSTM CM |
| Min ε (FGSM) | SNR > 40dB | `part4/adversarial.py` |

---

## Repository Structure

```
hinglish-transcription-santali-voice-cloning/
├── pipeline.py               # Main orchestrator — run this
├── part1/
│   ├── lid.py                # Multi-head frame-level LID (Wav2Vec2 + MHA)
│   ├── constrained_decode.py # Whisper + N-gram logit bias
│   └── denoiser.py           # Spectral subtraction
├── part2/
│   ├── ipa_converter.py      # Hinglish → IPA (custom G2P)
│   └── translator.py         # Hinglish → Santali (500-word corpus)
├── part3/
│   ├── voice_embedding.py    # x-vector / d-vector extraction
│   ├── prosody_warp.py       # F0 + energy + DTW warping
│   └── synthesizer.py        # VITS / Meta MMS TTS
├── part4/
│   ├── anti_spoof.py         # LFCC/CQCC CM + EER
│   └── adversarial.py        # FGSM on LID
├── utils/
│   ├── audio_utils.py        # Audio I/O, features
│   └── metrics.py            # WER, MCD, EER
├── ngram_lm/
│   └── build_ngram.py        # Build N-gram LM from syllabus
├── santali_corpus/           # Generated technical dictionary
├── student_voice_ref.wav     # Reference voice (60s)
├── lecture_segment.wav       # Sample input segment
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/saiswaroopkakarla/hinglish-transcription-santali-voice-cloning.git
cd hinglish-transcription-santali-voice-cloning

conda create -n su_pa2 python=3.10
conda activate su_pa2

pip install -r requirements.txt

# System dependencies
sudo apt-get install espeak-ng ffmpeg
pip install pyworld --break-system-packages
```

---

## Run

### 1. Build the N-gram LM
```bash
python ngram_lm/build_ngram.py
```

### 2. Run full pipeline
```bash
python pipeline.py \
    --lecture_audio  lecture_segment.wav \
    --student_voice  student_voice_ref.wav \
    --output_dir     outputs/
```

### 3. Use Meta MMS (recommended for Santali synthesis)
```bash
python pipeline.py \
    --lecture_audio  lecture_segment.wav \
    --student_voice  student_voice_ref.wav \
    --use_mms \
    --output_dir     outputs/
```

---

## Output Files

| File | Description |
|---|---|
| `outputs/denoised_segment.wav` | After spectral subtraction |
| `outputs/transcript.txt` | Hinglish transcript (Whisper + N-gram) |
| `outputs/transcript_ipa.txt` | IPA representation |
| `outputs/santali_transcript.txt` | Santali translation |
| `outputs/output_LRL_cloned.wav` | Final Santali lecture in student's voice |
| `outputs/synthesis_flat.wav` | Ablation: flat synthesis (no prosody warping) |
| `outputs/adversarial_sample.wav` | FGSM adversarial sample |
| `outputs/santali_tech_dict.csv` | 500-word technical dictionary |
| `outputs/results.json` | All evaluation metrics |

---

## Architecture Notes

**LID (Part I):** Wav2Vec2-base frozen encoder → 4-head self-attention (2 layers, hidden=256) → frame-level classifier 768→256→128→2. Frame resolution ~20ms.

**Constrained Decoding (Part I):** Whisper-large-v3 beam search (n=5) with N-gram logit bias: `adjusted_logit(w) = logit_whisper(w) + λ × log P_ngram(w | context)`, λ=2.0, technical term boost=3.0.

**IPA Conversion (Part II):** Custom Hinglish G2P handles retroflex, aspirated, and dental sounds; word-level language detection routes English tokens through espeak-ng.

**Prosody Warping (Part III):** WORLD vocoder extracts F0 + SP + AP; DTW aligns reference and synthesised contours in O(T_ref × T_syn); re-synthesis preserves teaching cadence.

**Anti-Spoofing (Part IV):** LFCC (60-dim, linear filterbank) + Bi-LSTM with attention pooling → binary CM score; EER evaluated with full threshold sweep.

---

## References

Radford et al. (2022) · Baevski et al. (2020) · Kim et al. (2021, VITS) · Kong et al. (2020, HiFi-GAN) · Goodfellow et al. (2014, FGSM) · Todisco et al. (2019, ASVspoof) · Prakash & Jyothi (2021, Hinglish ASR)

---

## Author

**Kakarla Sai Swaroop** — M25DE1023, IIT Jodhpur M.Tech Data Engineering
