#!/usr/bin/env python3
"""
Debug smoke test for LoRA integration in TransReID.

This script:
1. Builds the model with LoRA configuration
2. Runs a single forward pass on random input
3. Prints parameter counts and confirms LoRA injection
"""

import sys
import os
import torch
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model import make_model
from datasets import make_dataloader


def main():
    parser = argparse.ArgumentParser(description="LoRA Debug Smoke Test")
    parser.add_argument(
        "--config_file", 
        default="configs/Market/vit_transreid_stride_lora.yml", 
        help="path to config file", 
        type=str
    )
    args = parser.parse_args()

    # Load configuration
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    print("=" * 60)
    print("LoRA Debug Smoke Test")
    print("=" * 60)
    print(f"Config file: {args.config_file}")
    print(f"LoRA enabled: {cfg.LORA.ENABLED}")
    print(f"LoRA R: {cfg.LORA.R}")
    print(f"LoRA Alpha: {cfg.LORA.ALPHA}")
    print(f"LoRA targets: {cfg.LORA.TARGETS}")
    print(f"Model type: {cfg.MODEL.NAME}")
    print(f"Transformer type: {cfg.MODEL.TRANSFORMER_TYPE}")
    print("=" * 60)

    # Create dataloader to get dataset info
    try:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
        print(f"Dataset: {cfg.DATASETS.NAMES}")
        print(f"Number of classes: {num_classes}")
        print(f"Camera num: {camera_num}, View num: {view_num}")
    except Exception as e:
        print(f"Warning: Could not load dataloader ({e}), using defaults")
        num_classes = 751  # Market-1501 default
        camera_num = 6
        view_num = 1

    print("=" * 60)

    # Build model
    print("Building model...")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("PARAMETER COUNTS:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    print("=" * 60)

    # Debug utility: Print all nn.Linear layer names
    print("\n" + "=" * 60)
    print("DEBUG: ALL LINEAR LAYER NAMES IN MODEL")
    print("=" * 60)
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    
    if linear_layers:
        print(f"Found {len(linear_layers)} Linear layers:")
        for name in linear_layers[:10]:  # Show first 10
            print(f"  - {name}")
        if len(linear_layers) > 10:
            print(f"  ... and {len(linear_layers) - 10} more")
        
        # Check which ones match our targets
        print("\n" + "-" * 60)
        print("MATCHING TARGET PATTERNS:")
        targets = cfg.LORA.TARGETS if cfg.LORA.ENABLED else ["qkv", "proj", "fc1", "fc2"]
        for target in targets:
            matching = [name for name in linear_layers if target in name]
            print(f"  '{target}': {len(matching)} matches")
            if matching:
                print(f"    Examples: {matching[:3]}")
    else:
        print("No Linear layers found!")
    
    print("=" * 60)

    # Check if LoRA was actually injected
    lora_layers = 0
    lora_layer_names = []
    for name, module in model.named_modules():
        if 'lora' in name.lower() or 'LoRALinear' in str(type(module)):
            lora_layers += 1
            lora_layer_names.append(name)
            if lora_layers <= 5:  # Show first 5
                print(f"Found LoRA layer: {name}")
    
    if lora_layers > 5:
        print(f"... and {lora_layers - 5} more LoRA layers")

    if cfg.LORA.ENABLED:
        if lora_layers > 0:
            print(f"✅ LoRA injection successful: Found {lora_layers} LoRA layers")
        else:
            print("❌ LoRA injection failed: No LoRA layers found")
    else:
        print("ℹ️  LoRA disabled in config")

    print("=" * 60)

    # Run forward pass
    print("Running forward pass...")
    model.eval()
    
    # Create random input tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
    
    print(f"Input shape: {input_tensor.shape}")
    
    try:
        with torch.no_grad():
            # For transformer models, we need camera and view labels
            if cfg.MODEL.NAME == 'transformer':
                cam_label = torch.zeros(batch_size, dtype=torch.long)
                view_label = torch.zeros(batch_size, dtype=torch.long)
                output = model(input_tensor, cam_label=cam_label, view_label=view_label)
            else:
                output = model(input_tensor)
        
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        
        # Check for NaN or Inf
        if torch.isnan(output).any():
            print("❌ Warning: Output contains NaN values!")
        elif torch.isinf(output).any():
            print("❌ Warning: Output contains Inf values!")
        else:
            print("✅ Output values are finite")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
