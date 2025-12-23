# -*- coding: utf-8 -*-
"""
Noisy Augmentation module for noise-robust fine-tuning on spoken/ASR data.

Provides data augmentation functions to make model representations invariant
to ASR noise and disfluencies.

Copyright (c) 2024
MIT License
"""

import re
import random
from typing import List, Optional


# Common filler words in spoken dialogue
FILLER_WORDS = {
    "uh",
    "um",
    "uhm",
    "uh-huh",
    "hmm",
    "hm",
    "mm",
    "mmm",
    "er",
    "err",
    "ah",
    "oh",
    "eh",
    "mhm",
    "mm-hmm",
    "like",
    "you know",
    "i mean",
    "basically",
    "actually",
    "well",
    "so",
    "yeah",
    "yep",
    "nope",
    "okay",
    "ok",
    "right",
}


def remove_fillers(text: str) -> str:
    """Remove common filler words from text."""
    words = text.lower().split()
    filtered = [w for w in words if w not in FILLER_WORDS]
    return " ".join(filtered)


def remove_punctuation(text: str) -> str:
    """Remove punctuation and lowercase."""
    return re.sub(r"[^\w\s]", "", text.lower())


def normalize_numbers(text: str) -> str:
    """Normalize number representations (e.g., '1 2 2 3' <-> '1223')."""
    # Join consecutive single digits
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    return text


def random_token_drop(text: str, drop_prob: float = 0.1, max_drops: int = 2) -> str:
    """Randomly drop 1-2 tokens from the text."""
    words = text.split()
    if len(words) <= 2:
        return text

    n_drops = min(max_drops, max(1, int(len(words) * drop_prob)))
    drop_indices = random.sample(range(len(words)), min(n_drops, len(words) - 1))

    return " ".join(w for i, w in enumerate(words) if i not in drop_indices)


def remove_repeated_words(text: str) -> str:
    """Remove consecutive repeated words (common ASR artifact)."""
    return re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text)


def augment_text(
    text: str,
    remove_filler_prob: float = 0.5,
    remove_punct_prob: float = 0.3,
    normalize_num_prob: float = 0.3,
    token_drop_prob: float = 0.2,
    remove_repeat_prob: float = 0.5,
) -> str:
    """
    Apply random augmentations to text for noise-robust training.

    Args:
        text: Input text
        remove_filler_prob: Probability of removing filler words
        remove_punct_prob: Probability of removing punctuation
        normalize_num_prob: Probability of normalizing numbers
        token_drop_prob: Probability of random token drop
        remove_repeat_prob: Probability of removing repeated words

    Returns:
        Augmented text
    """
    if random.random() < remove_filler_prob:
        text = remove_fillers(text)

    if random.random() < remove_punct_prob:
        text = remove_punctuation(text)

    if random.random() < normalize_num_prob:
        text = normalize_numbers(text)

    if random.random() < token_drop_prob:
        text = random_token_drop(text)

    if random.random() < remove_repeat_prob:
        text = remove_repeated_words(text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "empty"


def create_noisy_view(text: str) -> str:
    """Create a noisy view of the text for contrastive learning."""
    return augment_text(text)


def create_augmented_pair(text: str, label: str) -> tuple:
    """
    Create an augmented (noisy view, same label) pair for training.

    Returns:
        Tuple of (original_text, augmented_text, label)
    """
    augmented = create_noisy_view(text)
    return (text, augmented, label)
