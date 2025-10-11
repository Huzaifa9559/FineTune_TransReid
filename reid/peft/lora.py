# reid/peft/lora.py
import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_or_dropout(p: float):
    return nn.Identity() if p <= 0 else nn.Dropout(p)


class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear as:
      y = base(x) + scaling * (B(A(dropout(x)))) + optional lora_bias
    - base params are frozen
    - only A, B (and optional lora_bias) are trainable
    """
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias_mode: str = "none",  # "none" | "lora" | "all" (keep "none" by default)
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.drop = _identity_or_dropout(dropout)
        self.bias_mode = bias_mode

        if self.r > 0:
            # A: (r, in), B: (out, r)
            self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if self.base.bias is not None and self.bias_mode in ("lora", "all"):
            self.lora_bias = nn.Parameter(torch.zeros_like(self.base.bias))
        else:
            self.register_parameter("lora_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            x_d = self.drop(x)
            # (B, in) -> (B, r) via A^T; then (B, out) via B
            update = F.linear(x_d, self.lora_A)          # (B, r)
            update = F.linear(update, self.lora_B)       # (B, out)
            y = y + self.scaling * update
        if self.lora_bias is not None:
            y = y + self.lora_bias
        return y

    @property
    def adapter_state(self):
        s = {}
        if self.lora_A is not None: s["lora_A"] = self.lora_A
        if self.lora_B is not None: s["lora_B"] = self.lora_B
        if self.lora_bias is not None: s["lora_bias"] = self.lora_bias
        return s

    def merge_into_base_(self):
        """
        One-way merge: bake LoRA weights into base.weight (and bias if present),
        then zero adapters to avoid double counting.
        """
        if self.r == 0:
            return
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
            self.base.weight += delta
            if self.lora_bias is not None and self.base.bias is not None:
                self.base.bias += self.lora_bias
            # zero adapters after merge
            nn.init.zeros_(self.lora_B)
            nn.init.zeros_(self.lora_A)
            if self.lora_bias is not None:
                nn.init.zeros_(self.lora_bias)


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a nested submodule given its dotted path name."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


def iter_linear_targets(model: nn.Module, targets: List[str]) -> Iterable[Tuple[str, nn.Linear]]:
    """
    Yield (qualified_name, linear_module) for layers whose qualified name contains
    any of the target keys. Typical ViT/timm names include:
      - attention.qkv (Linear)
      - attention.proj (Linear)
      - mlp.fc1, mlp.fc2 (Linear)
    Adjust `targets` if your repo uses different names.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in targets):
                yield name, module


def inject_lora_into_vit(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    targets: List[str],
    bias_mode: str = "none",
    #include_blocks=list(range(6, 12)),
):
    """
    Wrap matching nn.Linear modules with LoRALinear and return list of replaced names.
    """
    replaced = []
    for name, lin in list(iter_linear_targets(model, targets)):
        wrapped = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout, bias_mode=bias_mode)
        set_module_by_name(model, name, wrapped)
        replaced.append(name)
    return replaced


def mark_trainable_lora_and_head(model: nn.Module, train_head: bool = True):
    """
    Freeze everything, then unfreeze:
      - LoRA adapter params (A/B/bias),
      - classifier/BNNeck/ID head params if train_head=True.
    """
    for p in model.parameters():
        p.requires_grad = False

    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            # Only unfreeze LoRA adapter parameters, not base parameters
            if m.lora_A is not None:
                m.lora_A.requires_grad = True
            if m.lora_B is not None:
                m.lora_B.requires_grad = True
            if m.lora_bias is not None:
                m.lora_bias.requires_grad = True

    if train_head:
        for name, module in model.named_modules():
            lname = name.lower()
            if any(k in lname for k in ["classifier", "head", "bnneck", "id_head"]):
                for p in module.parameters():
                    p.requires_grad = True


def lora_state_dict(model: nn.Module) -> dict:
    """
    Return only LoRA adapter tensors in a flat dict keyed by qualified module name.
    Example keys:
      "...attn.qkv.lora_A", "...attn.qkv.lora_B", "...attn.qkv.lora_bias"
    """
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                sd[f"{name}.lora_A"] = module.lora_A
                sd[f"{name}.lora_B"] = module.lora_B
            if module.lora_bias is not None:
                sd[f"{name}.lora_bias"] = module.lora_bias
    return sd


def load_lora_state_dict(model: nn.Module, adapter_sd: dict, strict: bool = False):
    """
    Load LoRA adapter tensors back into matching modules. Missing keys are ignored unless strict=True.
    """
    missing = []
    for k, v in adapter_sd.items():
        # split "...module_path.param_name"
        try:
            mod_name, tensor_name = k.rsplit(".", 1)
            mod = get_module_by_name(model, mod_name)
            if not isinstance(v, torch.nn.Parameter):
                v = nn.Parameter(v)
            setattr(mod, tensor_name, v)
        except Exception:
            missing.append(k)
    if strict and missing:
        raise RuntimeError(f"Missing LoRA keys: {missing}")


def maybe_merge_lora(model: nn.Module, enabled: bool, merge_at_eval: bool):
    """
    If enabled and merge_at_eval=True, register a forward-pre hook that merges adapters
    into base weights once the model is put into eval mode (idempotent for a session).
    """
    if not enabled or not merge_at_eval:
        return

    merged_once = {"done": False}

    def merge_hook(m, *args, **kwargs):
        if merged_once["done"]:
            return
        for _, module in m.named_modules():
            if isinstance(module, LoRALinear):
                module.merge_into_base_()
        merged_once["done"] = True

    # Merge when the first forward happens after .eval() is set.
    model.register_forward_pre_hook(merge_hook, with_kwargs=False)
