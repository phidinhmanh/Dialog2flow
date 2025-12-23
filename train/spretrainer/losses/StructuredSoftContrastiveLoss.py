# -*- coding: utf-8 -*-
"""
Structured Soft-Contrastive Loss for TOD.

Improves upon vanilla soft-contrastive loss by using structured similarity
based on dialog act and slot components:

δ = α * sim(act_i, act_j) + (1-α) * sim(slots_i, slots_j)

Copyright (c) 2024
MIT License
"""

import re
import torch
import logging
from torch import Tensor
from typing import Dict, Set, Tuple, List, Optional
from sentence_transformers import util, SentenceTransformer

logger = logging.getLogger(__name__)


def parse_label(label: str) -> Tuple[str, Set[str]]:
    """
    Parse a TOD label into act and slots components.

    Expected formats:
    - "inform slot1 slot2"
    - "request slot"
    - "greeting"
    - "domain-act slot1 slot2"

    Args:
        label: The dialog act label string

    Returns:
        Tuple of (act, set_of_slots)
    """
    label = label.lower().strip()

    # Handle domain-prefixed labels (e.g., "hotel-inform type")
    if "-" in label and not label.startswith("uh"):
        parts = label.split("-", 1)
        if len(parts) == 2:
            label = parts[1]  # Remove domain prefix

    tokens = label.split()
    if not tokens:
        return ("unknown", set())

    act = tokens[0]
    slots = set(tokens[1:]) if len(tokens) > 1 else set()

    return (act, slots)


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 1.0 if both sets are empty (same = empty).
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def compute_structured_similarity(
    label1: str,
    label2: str,
    alpha: float = 0.7,
) -> float:
    """
    Compute structured similarity between two TOD labels.

    δ = α * sim(act_i, act_j) + (1-α) * sim(slots_i, slots_j)

    Args:
        label1: First label
        label2: Second label
        alpha: Weight for act similarity (0.6-0.8 recommended)

    Returns:
        Similarity score in [0, 1]
    """
    act1, slots1 = parse_label(label1)
    act2, slots2 = parse_label(label2)

    # Act similarity: 1 if same, 0 if different
    act_sim = 1.0 if act1 == act2 else 0.0

    # Slot similarity: Jaccard
    slot_sim = jaccard_similarity(slots1, slots2)

    return alpha * act_sim + (1 - alpha) * slot_sim


def compute_structured_similarity_matrix(
    labels: List[str],
    alpha: float = 0.7,
) -> Tensor:
    """
    Compute pairwise structured similarity matrix for a list of labels.

    Args:
        labels: List of label strings
        alpha: Weight for act similarity

    Returns:
        Similarity matrix [n_labels, n_labels]
    """
    n = len(labels)
    sim_matrix = torch.zeros(n, n)

    # Parse all labels first
    parsed = [parse_label(label) for label in labels]

    for i in range(n):
        for j in range(n):
            act_i, slots_i = parsed[i]
            act_j, slots_j = parsed[j]

            act_sim = 1.0 if act_i == act_j else 0.0
            slot_sim = jaccard_similarity(slots_i, slots_j)

            sim_matrix[i, j] = alpha * act_sim + (1 - alpha) * slot_sim

    return sim_matrix


class StructuredLabelEncoder:
    """
    Encodes TOD labels using structured components (act + slots).

    Can use either:
    1. Binary encoding (act match + Jaccard slots)
    2. Embedding-based encoding (embed act and slot tokens)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        use_embeddings: bool = False,
        embedding_model: Optional[str] = None,
    ):
        self.alpha = alpha
        self.use_embeddings = use_embeddings
        self.embedding_model = None
        self.act_embeddings: Dict[str, Tensor] = {}
        self.slot_embeddings: Dict[str, Tensor] = {}

        if use_embeddings and embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model)

    def encode_labels(self, labels: List[str]) -> None:
        """Pre-compute embeddings for all unique acts and slots."""
        if not self.use_embeddings or not self.embedding_model:
            return

        all_acts = set()
        all_slots = set()

        for label in labels:
            act, slots = parse_label(label)
            all_acts.add(act)
            all_slots.update(slots)

        # Encode acts
        act_list = list(all_acts)
        if act_list:
            act_embs = self.embedding_model.encode(act_list, convert_to_tensor=True)
            self.act_embeddings = {act: emb for act, emb in zip(act_list, act_embs)}

        # Encode slots
        slot_list = list(all_slots)
        if slot_list:
            slot_embs = self.embedding_model.encode(slot_list, convert_to_tensor=True)
            self.slot_embeddings = {
                slot: emb for slot, emb in zip(slot_list, slot_embs)
            }

    def compute_similarity(self, label1: str, label2: str) -> float:
        """Compute similarity between two labels."""
        if self.use_embeddings and self.act_embeddings:
            return self._compute_embedding_similarity(label1, label2)
        return compute_structured_similarity(label1, label2, self.alpha)

    def _compute_embedding_similarity(self, label1: str, label2: str) -> float:
        """Compute similarity using embeddings."""
        act1, slots1 = parse_label(label1)
        act2, slots2 = parse_label(label2)

        # Act similarity via cosine
        if act1 in self.act_embeddings and act2 in self.act_embeddings:
            act_sim = util.cos_sim(
                self.act_embeddings[act1].unsqueeze(0),
                self.act_embeddings[act2].unsqueeze(0),
            ).item()
        else:
            act_sim = 1.0 if act1 == act2 else 0.0

        # Slot similarity via average embedding cosine
        if slots1 and slots2:
            slot_embs1 = [
                self.slot_embeddings[s] for s in slots1 if s in self.slot_embeddings
            ]
            slot_embs2 = [
                self.slot_embeddings[s] for s in slots2 if s in self.slot_embeddings
            ]

            if slot_embs1 and slot_embs2:
                avg1 = torch.stack(slot_embs1).mean(dim=0)
                avg2 = torch.stack(slot_embs2).mean(dim=0)
                slot_sim = util.cos_sim(avg1.unsqueeze(0), avg2.unsqueeze(0)).item()
            else:
                slot_sim = jaccard_similarity(slots1, slots2)
        else:
            slot_sim = jaccard_similarity(slots1, slots2)

        return self.alpha * act_sim + (1 - self.alpha) * slot_sim

    def compute_similarity_matrix(self, labels: List[str]) -> Tensor:
        """Compute pairwise similarity matrix."""
        n = len(labels)
        sim_matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = self.compute_similarity(labels[i], labels[j])

        return sim_matrix
