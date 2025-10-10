# loss/make_loss.py
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy  # keep imports as in repo
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def _device_from_cfg(cfg) -> torch.device:
    want = str(getattr(cfg.MODEL, "DEVICE", "cpu")).lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # fallback order
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loss(cfg, num_classes):  # modified by gu + device-agnostic patch
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048  # keep as in original repo
    device = _device_from_cfg(cfg)

    # ----- Metric loss (triplet) -----
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print(f"using triplet loss with margin:{cfg.SOLVER.MARGIN}")
    else:
        print(f"expected METRIC_LOSS_TYPE should be triplet but got {cfg.MODEL.METRIC_LOSS_TYPE}")

    # ----- ID loss (cross-entropy, optionally label smoothing) -----
    use_labelsmooth = (str(cfg.MODEL.IF_LABELSMOOTH).lower() == 'on')
    if use_labelsmooth:
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    # ----- Center loss (optional; device-aware) -----
    center_criterion = None
    if str(getattr(cfg.MODEL, "IF_WITH_CENTER", "no")).lower() in ('yes', 'true', '1'):
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)

    # ----- Compose final loss function by sampler -----
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            if use_labelsmooth:
                return xent(score, target)
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            # ID loss
            if use_labelsmooth:
                if isinstance(score, list):
                    id_losses = [xent(s, target) for s in score[1:]]
                    ID_LOSS = sum(id_losses) / len(id_losses)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    id_losses = [F.cross_entropy(s, target) for s in score[1:]]
                    ID_LOSS = sum(id_losses) / len(id_losses)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # Triplet loss
            if isinstance(feat, list):
                tri_losses = [triplet(f, target)[0] for f in feat[1:]]
                TRI_LOSS = sum(tri_losses) / len(tri_losses)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
            else:
                TRI_LOSS = triplet(feat, target)[0]

            return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

    else:
        print(f"expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center "
              f"but got {cfg.DATALOADER.SAMPLER}")

        # safe default to avoid crashing if sampler is unexpected
        def loss_func(score, feat, target, *args, **kwargs):
            return F.cross_entropy(score, target)

    return loss_func, center_criterion
