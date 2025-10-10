# loss/center_loss.py
from __future__ import absolute_import

import torch
from torch import nn


def _pick_device(requested: str | None = None, use_gpu: bool | None = None) -> torch.device:
    """
    Choose a device in this priority:
    1) explicit requested ('cuda' | 'mps' | 'cpu')
    2) if use_gpu is True -> cuda if available, else mps if available, else cpu
    3) fallback: cuda if available, else mps if available, else cpu
    """
    if requested is not None:
        requested = requested.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        # if the requested isn't available, fall through to auto-pick

    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CenterLoss(nn.Module):
    """
    Center loss (Wen et al., ECCV 2016).

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        use_gpu (bool): kept for backward-compat; prefer passing `device`.
        device (str | torch.device | None): 'cuda' | 'mps' | 'cpu' or a torch.device.
    """
    def __init__(self, num_classes: int = 751, feat_dim: int = 2048, use_gpu: bool = True, device=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)

        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = _pick_device(str(device).lower() if isinstance(device, str) else None, use_gpu)

        # Allocate centers on the selected device (NO direct .cuda() calls)
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim, device=self.device)
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: features, shape (batch_size, feat_dim)
            labels: ground-truth labels, shape (batch_size,) dtype long
        """
        if x.dim() != 2 or x.size(1) != self.feat_dim:
            raise ValueError(f"Expected x of shape (B, {self.feat_dim}), got {tuple(x.shape)}")

        if labels.dim() != 1 or labels.size(0) != x.size(0):
            raise ValueError(f"Expected labels of shape (B,), got {tuple(labels.shape)}")

        # Ensure x/labels live on the same device as centers
        if x.device != self.centers.device:
            x = x.to(self.centers.device)
        if labels.device != self.centers.device:
            labels = labels.to(self.centers.device)

        batch_size = x.size(0)

        # Compute squared Euclidean distance between features and class centers
        # distmat[i, j] = ||x_i - c_j||^2
        x_sq = torch.pow(x, 2).sum(dim=1, keepdim=True)                    # (B, 1)
        c_sq = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).t()     # (1, C)
        distmat = x_sq + c_sq                                              # (B, C)
        distmat.addmm_(x, self.centers.t(), beta=1.0, alpha=-2.0)          # distmat += -2 * x @ centers^T

        # Build mask for the ground-truth class of each sample
        classes = torch.arange(self.num_classes, device=self.centers.device, dtype=torch.long)  # (C,)
        labels = labels.long().unsqueeze(1).expand(batch_size, self.num_classes)                # (B, C)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))                          # (B, C) bool

        # Gather distances of each sample to its corresponding class center
        # (equivalent to distmat[range(B), labels_orig])
        dist = distmat[mask]                                                                     # (B,)
        dist = dist.clamp(min=1e-12, max=1e12)                                                   # numerical stability
        loss = dist.mean()
        return loss


if __name__ == "__main__":
    # Minimal self-test across available devices
    tgt_device = None  # set to 'cuda' | 'mps' | 'cpu' to force, or leave None to auto-pick
    dev = _pick_device(tgt_device)

    B, D, C = 16, 2048, 6
    center_loss = CenterLoss(num_classes=C, feat_dim=D, device=dev)

    x = torch.rand(B, D, device=dev)
    y = torch.randint(low=0, high=C, size=(B,), device=dev)

    loss = center_loss(x, y)
    print("device:", dev)
    print("loss:", loss.item())
