"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""

import random
import logging
import numpy as np

from torch.utils.data import Dataset
from sentence_transformers import InputExample

from collections.abc import Iterable
from itertools import permutations

from .NoisyAugmentation import augment_text

logger = logging.getLogger(__name__)


# Speaker role tokens
SPEAKER_TOKENS = {
    "user": "[USR]",
    "system": "[SYS]",
    "usr": "[USR]",
    "sys": "[SYS]",
}


def prepend_speaker_token(text: str, speaker: str = None) -> str:
    """
    Prepend speaker role token to text.

    Args:
        text: The utterance text
        speaker: Speaker role (e.g., 'user', 'system')

    Returns:
        Text with prepended speaker token
    """
    if speaker and speaker.lower() in SPEAKER_TOKENS:
        token = SPEAKER_TOKENS[speaker.lower()]
        return f"{token} {text}"
    return text


class SimilarityDatasetFromLabels(Dataset):
    """
    Dataset for labeled similarity learning with optional augmentation and speaker tokens.
    """

    def __init__(
        self,
        data: Iterable,
        shuffle: bool = True,
        labels_as_ix: bool = False,
        balance_labels: str = "none",
        # New options
        use_augmentation: bool = False,
        augment_prob: float = 0.5,
        use_speaker_tokens: bool = False,
        speaker_column: int = None,  # Column index for speaker in data tuple
    ):
        """
        Args:
            data: Iterable of (text, label) or (text, label, speaker) tuples
            shuffle: Whether to shuffle samples
            labels_as_ix: Convert labels to indices
            balance_labels: 'none', 'extend', 'reduce'
            use_augmentation: Apply noise augmentation for robustness
            augment_prob: Probability of applying augmentation
            use_speaker_tokens: Prepend [USR]/[SYS] tokens
            speaker_column: Index of speaker role in data tuple (if present)
        """
        if balance_labels != "none":
            logger.info(
                f"Building similarity dataset from labeled instance using '{balance_labels}' strategy to balance the classes"
            )

        if use_augmentation:
            logger.info("Noise augmentation enabled for spoken/ASR robustness")

        if use_speaker_tokens:
            logger.info("Speaker role tokens [USR]/[SYS] enabled")

        data = list(data)
        self.balanced = balance_labels
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        self.augment_prob = augment_prob
        self.use_speaker_tokens = use_speaker_tokens
        self.speaker_column = speaker_column

        # Extract text, labels, and optionally speakers
        if speaker_column is not None and len(data[0]) > 2:
            self.y = np.array([item[1] for item in data])
            self.x = [item[0] for item in data]
            self.speakers = [
                item[speaker_column] if len(item) > speaker_column else None
                for item in data
            ]
        else:
            self.y = np.array([label for _, label in data])
            self.x = [text for text, _ in data]
            self.speakers = [None] * len(self.x)

        self.ix2label, counts = np.unique(self.y, return_counts=True)
        self._sorted_counts = sorted(counts, reverse=True)
        if labels_as_ix:
            self.label2ix = {label: ix for ix, label in enumerate(self.ix2label)}
            self.y = np.array([self.label2ix[label] for label in self.y])
            self.labels = np.arange(self.ix2label.shape[0])
        else:
            self.label2ix = None
            self.labels = self.ix2label
        self.regenerate_pairs()

    def _process_text(
        self, text: str, speaker: str = None, apply_augment: bool = False
    ) -> str:
        """Process text with optional augmentation and speaker tokens."""
        # Apply augmentation first (before speaker token)
        if (
            apply_augment
            and self.use_augmentation
            and random.random() < self.augment_prob
        ):
            text = augment_text(text)

        # Prepend speaker token
        if self.use_speaker_tokens:
            text = prepend_speaker_token(text, speaker)

        return text

    def regenerate_pairs(self):
        self.samples = []
        max_limit = self._sorted_counts[1] if self.balanced == "reduce" else None
        for ix, label in enumerate(self.labels):
            label_ixs = np.where(self.y == label)[0]
            sampling_ix = np.random.permutation(label_ixs)[:max_limit]
            if self.balanced == "none" or self.balanced == "reduce":
                for ix in range(len(sampling_ix)):
                    if ix + 1 > len(sampling_ix) - 1:
                        break
                    ix0 = sampling_ix[ix]
                    ix1 = sampling_ix[ix + 1]

                    # Process texts
                    text0 = self._process_text(
                        self.x[ix0], self.speakers[ix0], apply_augment=True
                    )
                    text1 = self._process_text(
                        self.x[ix1], self.speakers[ix1], apply_augment=True
                    )

                    self.samples.append(InputExample(texts=[text0, text1], label=label))
            elif self.balanced == "extend":
                for ix in range(self._sorted_counts[0]):
                    if (ix + 1) % len(sampling_ix) == 0:
                        sampling_ix = np.random.permutation(label_ixs)
                    ix0 = sampling_ix[ix % len(sampling_ix)]
                    ix1 = sampling_ix[(ix + 1) % len(sampling_ix)]

                    # Process texts
                    text0 = self._process_text(
                        self.x[ix0], self.speakers[ix0], apply_augment=True
                    )
                    text1 = self._process_text(
                        self.x[ix1], self.speakers[ix1], apply_augment=True
                    )

                    self.samples.append(InputExample(texts=[text0, text1], label=label))
            else:
                raise ValueError(
                    f"Not a valid `balance_labels` value. Received {self.balanced}, expected either 'none', 'extend', or 'reduce'"
                )

        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
