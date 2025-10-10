# processor/processor.py
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist

from reid.peft.lora import lora_state_dict


def _resolve_device(cfg) -> torch.device:
    want = str(getattr(cfg.MODEL, "DEVICE", "cpu")).lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # Fallback order
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_amp_context_and_scaler(device: torch.device):
    """
    Returns (autocast_ctx_factory, scaler)
    - CUDA: use torch.cuda.amp.autocast + GradScaler
    - MPS/CPU: use torch.amp.autocast (no scaler) or nullcontext
    """
    if device.type == "cuda":
        return (lambda: torch.cuda.amp.autocast(), torch.cuda.amp.GradScaler())
    # MPS supports autocast but GradScaler is CUDA-only
    if device.type == "mps":
        return (lambda: torch.amp.autocast(device_type="mps", dtype=torch.float16), None)
    # CPU: no AMP by default
    return (lambda: nullcontext(), None)


def _save_checkpoint(cfg, model, epoch):
    """Save checkpoint with LoRA adapter-only option."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch))

    # Unwrap DDP if needed
    model_to_save = model.module if hasattr(model, "module") else model

    if getattr(cfg, "LORA", None) and cfg.LORA.ENABLED and cfg.LORA.SAVE_ADAPTER_ONLY:
        # Save only LoRA adapters
        state_dict = lora_state_dict(model_to_save)
        torch.save({"adapters": state_dict}, checkpoint_path)
        logger = logging.getLogger("transreid.train")
        logger.info(f"Saved LoRA adapters only to {checkpoint_path} ({len(state_dict)} adapter tensors)")
    else:
        # Save full model state dict
        torch.save(model_to_save.state_dict(), checkpoint_path)


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = _resolve_device(cfg)
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    # Device / DDP handling
    use_ddp = bool(cfg.MODEL.DIST_TRAIN) and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_ddp:
        # For DDP, pick the specific CUDA device by local_rank
        device = torch.device("cuda", index=local_rank)
        torch.cuda.set_device(device)
        model.to(device)
        logger.info(f"Using DDP on device {device} with {torch.cuda.device_count()} GPUs")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True
        )
    else:
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    amp_ctx_factory, scaler = _get_amp_context_and_scaler(device)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        # Typical schedulers expect .step(epoch) per-epoch
        if hasattr(scheduler, "step"):
            try:
                scheduler.step(epoch)
            except TypeError:
                # some schedulers use step() without args
                scheduler.step()

        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            # Zero grads
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()

            # Move batch to device
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            # Forward + loss (AMP if available)
            with amp_ctx_factory():
                # Some TransReID forwards require labels/cam/view for JPM etc.
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

                # Add center loss term only if enabled/present
                if center_criterion is not None:
                    center_loss_weight = getattr(cfg.SOLVER, "CENTER_LOSS_WEIGHT", 0.0005)
                    loss = loss + center_loss_weight * center_criterion(feat, target)

            # Backward / step (CUDA uses scaler; others standard)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # Center optimizer step (when present)
                if (center_criterion is not None) and (optimizer_center is not None):
                    # If you scale center grads inversely to the weight, do it here.
                    center_loss_weight = getattr(cfg.SOLVER, "CENTER_LOSS_WEIGHT", 0.0005)
                    for p in center_criterion.parameters():
                        if p.grad is not None:
                            p.grad.data *= (1.0 / float(center_loss_weight))
                    optimizer_center.step()
            else:
                loss.backward()
                optimizer.step()
                if (center_criterion is not None) and (optimizer_center is not None):
                    center_loss_weight = getattr(cfg.SOLVER, "CENTER_LOSS_WEIGHT", 0.0005)
                    for p in center_criterion.parameters():
                        if p.grad is not None:
                            p.grad.data *= (1.0 / float(center_loss_weight))
                    optimizer_center.step()

            # Accuracy
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            # Synchronize for accurate timing if CUDA/MPS
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass

            if (n_iter + 1) % log_period == 0:
                # Some schedulers expose lr differently; handle common cases
                try:
                    base_lr = scheduler._get_lr(epoch)[0]
                except Exception:
                    try:
                        base_lr = scheduler.get_last_lr()[0]
                    except Exception:
                        base_lr = getattr(scheduler, "base_lrs", [cfg.SOLVER.BASE_LR])[0]
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not use_ddp:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        # Checkpoint
        if epoch % checkpoint_period == 0:
            if use_ddp:
                if dist.get_rank() == 0:
                    _save_checkpoint(cfg, model, epoch)
            else:
                _save_checkpoint(cfg, model, epoch)

        # Evaluation
        if epoch % eval_period == 0:
            if use_ddp:
                if dist.get_rank() == 0:
                    _evaluate(cfg, model, val_loader, evaluator, device, logger, epoch)
            else:
                _evaluate(cfg, model, val_loader, evaluator, device, logger, epoch)


def _evaluate(cfg, model, val_loader, evaluator, device, logger, epoch):
    model.eval()
    evaluator.reset()

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if epoch is None:
        logger.info("Validation Results")
    else:
        logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # Free device cache if CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = _resolve_device(cfg)
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    # DP for inference (only sensible on CUDA)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs for inference".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
