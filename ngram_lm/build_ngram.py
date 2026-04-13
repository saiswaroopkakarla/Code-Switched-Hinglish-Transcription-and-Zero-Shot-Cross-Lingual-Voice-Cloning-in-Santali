"""
ngram_lm/build_ngram.py — Build and save the N-gram LM from the Speech Course Syllabus.

Run this first before transcription:
    python ngram_lm/build_ngram.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part1.constrained_decode import build_ngram_lm, SYLLABUS_TEXT

if __name__ == "__main__":
    print("Building N-gram Language Model from Speech Course Syllabus...")
    lm = build_ngram_lm(
        extra_corpus=SYLLABUS_TEXT,
        n=3,
        save_path="ngram_lm/speech_lm.json",
    )
    # Validate
    test_words = ["cepstrum", "spectrogram", "stochastic", "hello", "the"]
    print("\nTest log-probabilities:")
    for w in test_words:
        lp = lm.log_prob(w, ("frequency", "and"))
        print(f"  P({w} | frequency, and) = {lp:.4f}")
    print("\nDone. LM saved to ngram_lm/speech_lm.json")
