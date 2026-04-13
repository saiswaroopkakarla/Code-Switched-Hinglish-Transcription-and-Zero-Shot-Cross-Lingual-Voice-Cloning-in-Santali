"""
part1/constrained_decode.py — Constrained Beam Search via N-gram Logit Bias

Pipeline:
    1. Build a character/word N-gram LM from Speech Course Syllabus text.
    2. Wrap Whisper's beam search to add logit bias at each step:
       logit_bias = λ * log P_ngram(w | context)
    3. Technical terms get boosted so "cepstrum", "stochastic", etc. 
       are prioritised over acoustically similar words.

Mathematical formulation:
    adjusted_logit(w_t) = logit_whisper(w_t) + λ * log P_ngram(w_t | w_{t-n+1}^{t-1})

References:
    - Radford et al. (2022) Whisper
    - Brown et al. (2020) GPT-3 logit bias
"""

import os
import math
import json
import numpy as np
import torch
from collections import defaultdict, Counter
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# ─────────────────────────── Speech Syllabus ────────────────────────────────

SYLLABUS_TEXT = """
Speech Understanding course covers digital signal processing, short time Fourier transform,
spectrogram, cepstrum, mel frequency cepstral coefficients MFCC, linear predictive coding LPC,
autocorrelation, pitch detection fundamental frequency F0, voice activity detection VAD,
automatic speech recognition ASR, hidden Markov model HMM, Viterbi algorithm,
Baum-Welch algorithm forward backward algorithm, Gaussian mixture model GMM,
deep neural network DNN acoustic model, connectionist temporal classification CTC,
attention mechanism self-attention transformer architecture, BERT wav2vec2 HuBERT,
end-to-end speech recognition, sequence to sequence model, language model n-gram,
perplexity word error rate WER phone error rate PER, beam search decoding,
speaker recognition identification verification, x-vector d-vector ECAPA-TDNN,
speaker diarization, speech synthesis text to speech TTS, vocoder WaveNet WaveGlow,
neural TTS FastSpeech Tacotron VITS, voice conversion style transfer,
prosody pitch energy duration, dynamic time warping DTW, stochastic gradient descent,
backpropagation convolutional neural network recurrent neural network LSTM GRU,
code switching language identification Hinglish bilingual speech,
Indian languages Hindi Santali phonetics phonology articulatory acoustic,
formant resonance harmonic spectral envelope, framing windowing Hamming Hanning,
zero crossing rate energy entropy, noise robustness denoising spectral subtraction,
anti spoofing countermeasure LFCC CQCC equal error rate EER,
adversarial perturbation FGSM robustness
"""

TECHNICAL_TERMS = [
    "cepstrum", "spectrogram", "MFCC", "stochastic", "Viterbi", "diarization",
    "phoneme", "allophone", "formant", "prosody", "fricative", "plosive",
    "CTC", "HMM", "GMM", "wav2vec", "HuBERT", "Tacotron", "WaveNet",
    "vocoder", "spectral", "autocorrelation", "fundamental", "harmonic",
    "code-switching", "Hinglish", "bilingual", "reverb", "denoising",
    "anti-spoofing", "LFCC", "CQCC", "adversarial", "FGSM",
]


# ─────────────────────────── N-gram LM ──────────────────────────────────────

class NgramLM:
    """
    Smoothed N-gram Language Model (Kneser-Ney smoothing).
    Trained on Speech Course Syllabus text.
    """

    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n = n
        self.discount = discount
        self.ngram_counts  = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def tokenize(self, text: str) -> list:
        import re
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    def train(self, corpus: str):
        tokens = self.tokenize(corpus)
        self.vocab = set(tokens)
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i: i + self.n - 1])
            word    = tokens[i + self.n - 1]
            self.ngram_counts[context][word] += 1
            self.context_counts[context] += 1
        # Extra: give all technical terms a unigram boost by injecting
        # artificial counts so they are always preferred over OOV words.
        for term in TECHNICAL_TERMS:
            term_lower = term.lower()
            self.vocab.add(term_lower)
            # Add under a generic context so unigram backoff sees them
            self.ngram_counts[("<s>",)][term_lower] += 5
            self.context_counts[("<s>",)] += 5
        print(f"[NgramLM] Trained {self.n}-gram on {len(tokens)} tokens, "
              f"vocab={len(self.vocab)}")

    def log_prob(self, word: str, context: tuple) -> float:
        """Kneser-Ney smoothed log-probability."""
        word = word.lower()
        count = self.ngram_counts[context].get(word, 0)
        total = self.context_counts[context]

        if total == 0:
            return math.log(1.0 / max(len(self.vocab), 1))

        # Kneser-Ney
        p_kn = max(count - self.discount, 0) / total
        # Normalisation constant
        lambda_ctx = self.discount * len(self.ngram_counts[context]) / total
        # Lower-order backoff
        if self.n > 1 and len(context) > 0:
            lower_context = context[1:]
            p_lower = math.exp(self.log_prob(word, lower_context))
        else:
            p_lower = 1.0 / max(len(self.vocab), 1)

        prob = p_kn + lambda_ctx * p_lower
        return math.log(prob + 1e-10)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "n": self.n,
            "discount": self.discount,
            "ngram_counts": {
                str(k): dict(v) for k, v in self.ngram_counts.items()
            },
            "context_counts": {str(k): v for k, v in self.context_counts.items()},
            "vocab": list(self.vocab),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[NgramLM] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        lm = cls(n=data["n"], discount=data["discount"])
        lm.ngram_counts = defaultdict(
            Counter,
            {eval(k): Counter(v) for k, v in data["ngram_counts"].items()}
        )
        lm.context_counts = defaultdict(
            int, {eval(k): v for k, v in data["context_counts"].items()}
        )
        lm.vocab = set(data["vocab"])
        return lm


def build_ngram_lm(
    extra_corpus: str = "",
    n: int = 3,
    save_path: str = "ngram_lm/speech_lm.json",
) -> NgramLM:
    """Build and save the N-gram LM from syllabus + optional extra corpus."""
    lm = NgramLM(n=n)
    corpus = SYLLABUS_TEXT + " " + extra_corpus
    lm.train(corpus)
    lm.save(save_path)
    return lm


# ─────────────────────────── Logit Bias Hook ────────────────────────────────

class LogitBiasProcessor:
    """
    HuggingFace LogitsProcessor that adds N-gram LM bias at each decoding step.

    adjusted_logit(w) = logit_whisper(w) + λ * log P_ngram(w | context)
    """

    def __init__(
        self,
        lm: NgramLM,
        tokenizer,
        lambda_bias: float = 2.0,
        technical_boost: float = 3.0,
    ):
        self.lm = lm
        self.tokenizer = tokenizer
        self.lambda_bias = lambda_bias
        self.technical_boost = technical_boost
        self.context_words = []

        # Pre-tokenise technical terms for fast lookup
        self.tech_token_ids = set()
        for term in TECHNICAL_TERMS:
            ids = tokenizer.encode(" " + term, add_special_tokens=False)
            self.tech_token_ids.update(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Called by HuggingFace generate() at each step.
        input_ids: (B, T_so_far)
        scores:    (B, vocab_size)
        Returns modified scores.
        """
        batch_size = scores.shape[0]
        context_tuple = tuple(self.context_words[-(self.lm.n - 1):])

        for b in range(batch_size):
            # Boost technical term tokens
            for tid in self.tech_token_ids:
                if tid < scores.shape[1]:
                    scores[b, tid] += self.technical_boost

            # Add N-gram LM bias for top-K candidates
            top_ids = scores[b].topk(200).indices.tolist()
            for tid in top_ids:
                word = self.tokenizer.decode([tid]).strip().lower()
                if word:
                    lm_score = self.lm.log_prob(word, context_tuple)
                    scores[b, tid] += self.lambda_bias * lm_score

        # Update context with the most likely next token
        next_tokens = scores.argmax(-1)
        for nt in next_tokens.tolist():
            word = self.tokenizer.decode([nt]).strip().lower()
            if word:
                self.context_words.append(word)

        return scores


# ─────────────────────────── Whisper Inference ──────────────────────────────

def transcribe_constrained(
    wav: np.ndarray,
    sr: int = 16000,
    model_name: str = "openai/whisper-large-v3",
    lm_path: str = "ngram_lm/speech_lm.json",
    lambda_bias: float = 2.0,
    language: str = None,          # None = auto-detect (for code-switching)
    chunk_length_s: int = 30,
) -> dict:
    """
    Transcribe code-switched audio with N-gram logit bias.

    Returns:
        {
          'text':     full transcript string
          'segments': list of Whisper segments with timestamps
        }
    """
    from transformers import pipeline
    from transformers.pipelines.audio_utils import ffmpeg_read

    # Load or build N-gram LM
    if os.path.exists(lm_path):
        lm = NgramLM.load(lm_path)
    else:
        lm = build_ngram_lm(save_path=lm_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    logit_bias = LogitBiasProcessor(lm, processor.tokenizer, lambda_bias)

    # Chunk audio
    chunk_samples = chunk_length_s * sr
    all_text = []
    all_segments = []
    offset = 0.0

    for start in range(0, len(wav), chunk_samples):
        chunk = wav[start: start + chunk_samples]
        if len(chunk) < sr * 0.5:     # Skip sub-0.5s chunks
            break

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Forced decoder tokens
        forced_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        ) if language else None

        with torch.no_grad():
            # Reset LM context for each chunk
            logit_bias.context_words = []

            gen_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_ids,
                logits_processor=[logit_bias],
                num_beams=5,
                max_new_tokens=440,
            )

        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        all_text.append(text.strip())
        all_segments.append({
            "start": start / sr,
            "end": min((start + chunk_samples) / sr, len(wav) / sr),
            "text": text.strip(),
        })
        offset += chunk_length_s

    return {
        "text":     " ".join(all_text),
        "segments": all_segments,
    }
