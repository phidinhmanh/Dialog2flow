"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""

import torch
import logging

from torch import Tensor
from typing import Iterable, Dict, Union, Optional
from accelerate import Accelerator
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

from . import BaseContrastiveLoss
from ..datasets import SimilarityDatasetFromLabels
from .StructuredSoftContrastiveLoss import (
    StructuredLabelEncoder,
    compute_structured_similarity_matrix,
)
from .CenterLoss import CenterLoss

logger = logging.getLogger(__name__)


class LabeledContrastiveLoss(BaseContrastiveLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        use_soft_labels: bool = False,
        temperature: float = 0.05,
        soft_label_temperature: float = 0.35,
        soft_label_model: str = "multi-qa-mpnet-base-dot-v1",
        is_symmetrical: bool = True,
        accelerator: Accelerator = None,
        use_contrastive_head: bool = True,
        use_abs: bool = False,
        # New improvements
        use_structured_labels: bool = False,
        structured_alpha: float = 0.7,
        use_center_loss: bool = False,
        center_loss_lambda: float = 0.001,
        num_classes: int = 100,
    ):
        """
        Soft and Vanilla Supervised Contrastive loss as described in https://arxiv.org/abs/2410.18481.

        Extended with:
        - Structured soft labels: δ = α * sim(act) + (1-α) * sim(slots)
        - Center loss regularization: L_total = L_soft + λ * Σ ||z_i − μ_{y_i}||²

        Args:
            model: SentenceTransformer model
            use_soft_labels: Whether to use soft semantic labels
            soft_label_temperature: Temperature for soft labels
            soft_label_model: Model for label embeddings
            temperature: Contrastive loss temperature
            is_symmetrical: Symmetrical loss between anchor and target
            accelerator: Optional Accelerator for multi-GPU
            use_contrastive_head: Whether to use contrastive projection head
            use_abs: Use absolute value of cosine similarity
            use_structured_labels: Use structured TOD label similarity (act + slots)
            structured_alpha: Weight for act vs slots (0.6-0.8 recommended)
            use_center_loss: Add center loss regularization
            center_loss_lambda: Weight for center loss term
            num_classes: Number of classes for center loss

        References:
            * Paper: https://arxiv.org/abs/2410.18481
        """
        super(LabeledContrastiveLoss, self).__init__(
            model=model, use_contrastive_head=use_contrastive_head
        )

        logger.info(
            f"Initializing labeled-contrastive loss with {'soft' if use_soft_labels else 'hard'} labels"
        )
        if use_soft_labels:
            logger.info(f"  > Soft label temperature: {soft_label_temperature}")
            logger.info(f"  > label embedding model: {soft_label_model}")

        if use_structured_labels:
            logger.info(f"  > Structured labels enabled (alpha={structured_alpha})")

        if use_center_loss:
            logger.info(f"  > Center loss enabled (lambda={center_loss_lambda})")

        self.accelerator = accelerator
        self.symmetrical = is_symmetrical
        self.use_abs = use_abs
        self.use_soft_labels = use_soft_labels
        self.label2embedding = None
        self.temperature = temperature
        self.soft_label_temperature = (
            temperature if soft_label_temperature is None else soft_label_temperature
        )
        self.soft_label_model = soft_label_model
        self.similarity_fct = util.cos_sim
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

        # New: Structured labels
        self.use_structured_labels = use_structured_labels
        self.structured_alpha = structured_alpha
        self.structured_encoder: Optional[StructuredLabelEncoder] = None
        self.structured_similarity_matrix: Optional[Tensor] = None

        # New: Center loss
        self.use_center_loss = use_center_loss
        self.center_loss: Optional[CenterLoss] = None
        if use_center_loss:
            emb_size = self.encoder[0].auto_model.config.hidden_size
            feat_dim = 128 if use_contrastive_head else emb_size
            self.center_loss = CenterLoss(
                num_classes=num_classes,
                feat_dim=feat_dim,
                lambda_center=center_loss_lambda,
            )

    def compute_label_embeddings(self, dataset: SimilarityDatasetFromLabels):
        """Compute label embeddings or structured similarity matrix."""
        if self.use_structured_labels:
            # Use structured similarity based on act + slots
            logger.info("Computing structured label similarity matrix (act + slots)...")
            self.structured_encoder = StructuredLabelEncoder(
                alpha=self.structured_alpha
            )
            self.structured_similarity_matrix = compute_structured_similarity_matrix(
                list(dataset.ix2label), alpha=self.structured_alpha
            )
            logger.info(f"  > Matrix shape: {self.structured_similarity_matrix.shape}")
        elif self.use_soft_labels:
            if dataset.ix2label[0].isdigit():
                # if labels are numbers makes no sense to use label embeddings...
                self.use_soft_labels = False
            else:
                self.label2embedding = (
                    SentenceTransformer(self.soft_label_model)
                    .encode(
                        dataset.ix2label, convert_to_numpy=False, convert_to_tensor=True
                    )
                    .detach()
                    .to("cpu")
                )

    def forward(
        self,
        sentence_features: Iterable[Union[Dict[str, Tensor], Tensor]],
        labels: Tensor,
    ):
        reps = [
            self.model(sentence_feature)
            if isinstance(sentence_feature, (dict, Dict))
            else sentence_feature
            for sentence_feature in sentence_features
        ]

        anchors, positives, labels = self.gather_batches_across_processes(
            reps[0], reps[1], labels
        )
        if self.use_abs:
            scores = self.similarity_fct(anchors, positives).abs() / self.temperature
        else:
            scores = self.similarity_fct(anchors, positives) / self.temperature

        if self.use_structured_labels and self.structured_similarity_matrix is not None:
            # Use pre-computed structured similarity matrix
            label_indices = labels.cpu().numpy()
            labels_sim = self.structured_similarity_matrix[label_indices][
                :, label_indices
            ]
            labels_sim = labels_sim / self.soft_label_temperature
            targets = torch.nn.functional.softmax(labels_sim, dim=1).to(
                scores.get_device()
            )
        elif self.use_soft_labels:
            if isinstance(self.label2embedding, dict):
                emb_size = next(self.label2embedding.values()).shape[1]
            else:
                emb_size = self.label2embedding.shape[1]
            label_embs = torch.zeros([labels.shape[0], emb_size])
            for label in torch.unique(labels):
                label_embs[torch.where(labels == label)] = self.label2embedding[label]
            labels_sim = (
                util.cos_sim(label_embs, label_embs) / self.soft_label_temperature
            )
            targets = torch.nn.functional.softmax(labels_sim, dim=1).to(
                scores.get_device()
            )
        else:
            targets = torch.zeros_like(scores)
            for ix, label in enumerate(labels):
                targets[ix][torch.where(labels == label)[0]] = 1
            targets = targets / targets.sum(1).view(-1, 1)

        loss = self.cross_entropy_loss(scores, targets)
        if self.symmetrical:
            loss = (loss + self.cross_entropy_loss(scores.transpose(0, 1), targets)) / 2

        # Add center loss if enabled
        if self.use_center_loss and self.center_loss is not None:
            # Use anchors embeddings for center loss
            center_loss_value = self.center_loss(anchors, labels)
            loss = loss + center_loss_value

        return self.accelerator.num_processes * loss
