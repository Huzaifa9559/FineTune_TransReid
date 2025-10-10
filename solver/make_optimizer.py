# solver/make_optimizer.py
import torch


def make_optimizer(cfg, model, center_criterion=None):
    """
    Builds the main optimizer over trainable params only,
    and an optional optimizer for CenterLoss (if provided).
    """
    params = []
    base_lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    bias_lr_factor = getattr(cfg.SOLVER, "BIAS_LR_FACTOR", 1.0)
    wd_bias = getattr(cfg.SOLVER, "WEIGHT_DECAY_BIAS", wd)
    large_fc_lr = getattr(cfg.SOLVER, "LARGE_FC_LR", False)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lr = base_lr
        weight_decay = wd

        # bias-specific lr & decay
        if "bias" in name:
            lr = base_lr * bias_lr_factor
            weight_decay = wd_bias

        # heads/classifier higher lr if requested
        if large_fc_lr and ("classifier" in name or "arcface" in name):
            lr = base_lr * 2.0
            print("Using two times learning rate for fc:", name)

        params.append({"params": [p], "lr": lr, "weight_decay": weight_decay})

    # Log trainable parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable_params / total_params) if total_params else 0.0
    print(f"Optimizer using {trainable_params:,} / {total_params:,} parameters ({pct:.2f}%)")

    # ---- Main optimizer ----
    opt_name = str(cfg.SOLVER.OPTIMIZER_NAME).lower()
    if opt_name == "sgd":
        momentum = getattr(cfg.SOLVER, "MOMENTUM", 0.9)
        optimizer = torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=wd, nesterov=True)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")

    # ---- CenterLoss optimizer (optional) ----
    optimizer_center = None
    if center_criterion is not None:
        center_lr = getattr(cfg.SOLVER, "CENTER_LR", 0.5)
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=center_lr)

    return optimizer, optimizer_center
