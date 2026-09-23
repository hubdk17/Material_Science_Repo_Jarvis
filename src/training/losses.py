"""Loss functions for scalar and tensor training."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.features.tensor_transforms import cartesian_5d_to_matrix, cartesian_6d_to_matrix


class FrobeniusLoss(nn.Module):
    """Mean Frobenius norm loss between 3x3 predicted and true matrices."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, V_pred: torch.Tensor, V_true: torch.Tensor) -> torch.Tensor:
        if V_pred.ndim == 2:
            if V_pred.shape[-1] == 5:
                V_pred = cartesian_5d_to_matrix(V_pred)
            elif V_pred.shape[-1] == 6:
                V_pred = cartesian_6d_to_matrix(V_pred)

        if V_true.ndim == 2:
            if V_true.shape[-1] == 5:
                V_true = cartesian_5d_to_matrix(V_true)
            elif V_true.shape[-1] == 6:
                V_true = cartesian_6d_to_matrix(V_true)

        diff = V_pred - V_true
        frob = torch.norm(diff, p="fro", dim=(-2, -1))
        if self.reduction == "mean":
            return frob.mean()
        elif self.reduction == "sum":
            return frob.sum()
        return frob


class TensorCombinedLoss(nn.Module):
    """Combined Frobenius loss with optional auxiliary eigenvalue penalty."""
    def __init__(
        self,
        alpha_frob: float = 1.0,
        alpha_eigenval: float = 0.1,
        alpha_trace: float = 0.05
    ):
        super().__init__()
        self.alpha_frob = alpha_frob
        self.alpha_eigenval = alpha_eigenval
        self.alpha_trace = alpha_trace
        self.frob_loss = FrobeniusLoss()

    def forward(
        self,
        V_pred: torch.Tensor,
        V_true: torch.Tensor,
        eval_true: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = self.alpha_frob * self.frob_loss(V_pred, V_true)

        if self.alpha_trace > 0:
            trace_pred = torch.diagonal(V_pred, dim1=-2, dim2=-1).sum(-1)
            loss = loss + self.alpha_trace * trace_pred.abs().mean()

        if self.alpha_eigenval > 0 and eval_true is not None:
            V_sym = (V_pred + V_pred.transpose(-1, -2)) / 2.0
            evals_pred, _ = torch.linalg.eigh(V_sym)
            loss = loss + self.alpha_eigenval * F.l1_loss(evals_pred, eval_true)

        return loss
