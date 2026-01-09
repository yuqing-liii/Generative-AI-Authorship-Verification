# style_features.py
"""
Simple stylometric feature extractor for English texts.

Given a list of texts, returns a NumPy array of shape (n_samples, d),
where d is the number of hand-crafted style features.
"""

import re
import string
from typing import List

import numpy as np


# A small list of common English stopwords / function words
FUNCTION_WORDS = [
    "the", "and", "to", "of", "in", "that", "it", "is", "was", "for",
    "on", "with", "as", "by", "at", "from", "this", "be", "or"
]


def _sentence_split(text: str):
    # Very simple sentence splitter
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _tokenize(text: str):
    # Simple whitespace tokenizer, lowercased
    return text.lower().split()


def compute_style_features(texts: List[str]) -> np.ndarray:
    """
    Compute stylometric features for a list of texts.

    Returns:
        feats: np.ndarray of shape (len(texts), d)
    """
    features = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text)

        chars = text
        tokens = _tokenize(text)
        sentences = _sentence_split(text)

        num_chars = len(chars)
        num_tokens = len(tokens)
        num_types = len(set(tokens))
        num_sents = len(sentences) if sentences else 1

        # Avoid division by zero
        num_tokens_safe = max(num_tokens, 1)
        num_chars_safe = max(num_chars, 1)

        # Basic length-based features
        avg_token_len = num_chars_safe / num_tokens_safe
        type_token_ratio = num_types / num_tokens_safe
        avg_sent_len_tokens = num_tokens_safe / num_sents

        # Character-level ratios
        punct_chars = sum(1 for c in chars if c in string.punctuation)
        digit_chars = sum(1 for c in chars if c.isdigit())
        upper_chars = sum(1 for c in chars if c.isupper())

        punct_ratio = punct_chars / num_chars_safe
        digit_ratio = digit_chars / num_chars_safe
        upper_ratio = upper_chars / num_chars_safe

        # Function word frequencies
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        func_freqs = []
        for fw in FUNCTION_WORDS:
            freq = token_counts.get(fw, 0) / num_tokens_safe
            func_freqs.append(freq)

        feats = [
            num_chars,
            num_tokens,
            avg_token_len,
            type_token_ratio,
            avg_sent_len_tokens,
            punct_ratio,
            digit_ratio,
            upper_ratio,
        ] + func_freqs

        features.append(feats)

    return np.array(features, dtype=np.float32)
