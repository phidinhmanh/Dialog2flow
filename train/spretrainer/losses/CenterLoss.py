# -*- coding: utf-8 -*-
"""
Center Loss module for intra-class compactness.

Adds a regularization term to make embeddings of the same action/class
cluster more tightly around their centroid.

L_center = λ * Σ ||z_i − μ_{y_i}||²

Copyright (c) 2024
MIT License
"""

import torch
import torch.nn as nn
from torch import Tensor


class CenterLoss(nn.Module):
    """
    Center Loss for improving intra-class compactness.

    Maintains learnable class centroids and penalizes distance from embeddings
    to their corresponding centroid.

    Args:
        num_classes: Number of classes/actions
        feat_dim: Dimension of the embedding features
        lambda_center: Weight for the center loss term
        use_learnable_centers: If True, centers are learnable parameters
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        lambda_center: float = 0.001,
        use_learnable_centers: bool = False,
    ):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_center = lambda_center
        self.use_learnable_centers = use_learnable_centers

        if use_learnable_centers:
            # Learnable centers (updated via gradient)
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        else:
            # Non-learnable centers (updated via moving average)
            self.register_buffer("centers", torch.zeros(num_classes, feat_dim))
            self.register_buffer("center_counts", torch.zeros(num_classes))

    def update_centers(self, embeddings: Tensor, labels: Tensor, momentum: float = 0.9):
        """
        Update centers using exponential moving average.

        Args:
            embeddings: Batch of embeddings [batch_size, feat_dim]
            labels: Batch of labels [batch_size]
            momentum: EMA momentum (higher = slower update)
        """
        if self.use_learnable_centers:
            return  # Centers updated via gradient

        with torch.no_grad():
            unique_labels = torch.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 0:
                    class_embeddings = embeddings[mask]
                    new_center = class_embeddings.mean(dim=0)

                    if self.center_counts[label] == 0:
                        self.centers[label] = new_center
                    else:
                        self.centers[label] = (
                            momentum * self.centers[label] + (1 - momentum) * new_center
                        )
                    self.center_counts[label] += 1

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Compute center loss.

        Args:
            embeddings: Batch of embeddings [batch_size, feat_dim]
            labels: Batch of labels [batch_size]

        Returns:
            Center loss value (scalar)
        """
        batch_size = embeddings.size(0)

        # Get centers for each sample's label
        # Handle labels that might be out of range
        valid_mask = labels < self.num_classes
        if not valid_mask.all():
            # Filter out invalid labels
            embeddings = embeddings[valid_mask]
            labels = labels[valid_mask]
            batch_size = embeddings.size(0)

        if batch_size == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Get corresponding centers
        batch_centers = self.centers[labels]  # [batch_size, feat_dim]

        # Compute squared L2 distance to centers
        diff = embeddings - batch_centers
        loss = (diff**2).sum(dim=1).mean()

        # Update centers if not learnable
        if not self.use_learnable_centers:
            self.update_centers(embeddings.detach(), labels)

        return self.lambda_center * loss


class CenterLossWithContrastive(nn.Module):
    """
    Combines contrastive loss with center loss regularization.

    L_total = L_contrastive + λ * L_center
    """

    def __init__(
        self,
        contrastive_loss: nn.Module,
        num_classes: int,
        feat_dim: int,
        lambda_center: float = 0.001,
    ):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            lambda_center=lambda_center,
        )

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Compute combined loss."""
        l_contrastive = self.contrastive_loss(embeddings, labels)
        l_center = self.center_loss(embeddings, labels)
        return l_contrastive + l_center
