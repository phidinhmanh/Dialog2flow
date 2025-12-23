# -*- coding: utf-8 -*-
"""
Domain-Adversarial Loss for domain-invariant embeddings.

Uses gradient reversal to make embeddings hard to predict domain
while still being able to predict action/intent.

Copyright (c) 2024
MIT License
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from typing import Optional


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for domain adversarial training.

    Forward pass: identity function
    Backward pass: negates gradients and scales by lambda
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.

    Args:
        lambda_: Scaling factor for gradient reversal (default: 1.0)
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update lambda value (useful for scheduling)."""
        self.lambda_ = lambda_


class DomainClassifier(nn.Module):
    """
    Domain classifier head for adversarial training.

    Args:
        input_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layer
        num_domains: Number of domain classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_domains: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class DomainAdversarialLoss(nn.Module):
    """
    Domain-Adversarial Loss for learning domain-invariant representations.

    The loss encourages the encoder to produce embeddings that:
    1. Are useful for the main task (action prediction)
    2. Cannot be used to predict the domain (via gradient reversal)

    Args:
        input_dim: Dimension of input embeddings
        num_domains: Number of domain classes
        lambda_domain: Weight for domain adversarial loss
        hidden_dim: Hidden dimension of domain classifier
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        lambda_domain: float = 0.1,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.lambda_domain = lambda_domain
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.domain_classifier = DomainClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: Tensor,
        domain_labels: Tensor,
    ) -> Tensor:
        """
        Compute domain adversarial loss.

        Args:
            embeddings: Batch of embeddings [batch_size, input_dim]
            domain_labels: Batch of domain labels [batch_size]

        Returns:
            Domain adversarial loss (to be minimized)
        """
        # Apply gradient reversal
        reversed_embeddings = self.grl(embeddings)

        # Predict domain
        domain_logits = self.domain_classifier(reversed_embeddings)

        # Compute cross-entropy loss
        loss = self.criterion(domain_logits, domain_labels)

        return self.lambda_domain * loss

    def schedule_lambda(self, progress: float, gamma: float = 10.0):
        """
        Schedule lambda using the formula from DANN paper.

        λ = 2 / (1 + exp(-γ * p)) - 1

        where p is the training progress (0 to 1).

        Args:
            progress: Training progress (0 to 1)
            gamma: Scheduling parameter
        """
        import math

        new_lambda = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
        self.grl.set_lambda(new_lambda)


class DomainAdversarialContrastiveLoss(nn.Module):
    """
    Combined contrastive loss with domain adversarial regularization.

    L_total = L_contrastive + λ_domain * L_domain
    """

    def __init__(
        self,
        contrastive_loss: nn.Module,
        input_dim: int,
        num_domains: int,
        lambda_domain: float = 0.1,
    ):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.domain_loss = DomainAdversarialLoss(
            input_dim=input_dim,
            num_domains=num_domains,
            lambda_domain=lambda_domain,
        )

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        domain_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute combined loss."""
        l_contrastive = self.contrastive_loss(embeddings, labels)

        if domain_labels is not None:
            l_domain = self.domain_loss(embeddings, domain_labels)
            return l_contrastive + l_domain

        return l_contrastive
