#!/usr/bin/env python3
"""
Debug utility: overlay enclosed regions + U-Net wall mask on a floorplan image.

Usage:
    python utils/debug_enclosed.py <image_path> [--checkpoint weights/epoch_040.pth]
                                                [--black-threshold 60]
                                                [--no-unet]

Displays a matplotlib figure with 4 panels:
  1. Original image
  2. Enclosed regions (each region a unique color)
  3. U-Net wall mask (red overlay)
  4. Combined: enclosed regions + U-Net mask + original
"""

import argparse
import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.walls import _build_enclosed_regions, build_grey_wall_image


def make_region_colormap(num_labels: int, labels: np.ndarray,
                         min_region_area: int = 50) -> np.ndarray:
    """Create an RGBA overlay where each enclosed region has a unique color."""
    h, w = labels.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    for label_id in range(1, num_labels):
        region_mask = labels == label_id
        area = int(np.sum(region_mask))
        if area < min_region_area:
            continue

        # Golden-ratio hue spacing for distinct colors
        hue = (label_id * 0.618033988749895) % 1.0
        rgb = hsv_to_rgb([hue, 0.7, 0.9])
        overlay[region_mask, 0] = rgb[0]
        overlay[region_mask, 1] = rgb[1]
        overlay[region_mask, 2] = rgb[2]
        overlay[region_mask, 3] = 0.45

    return overlay


def run_unet_mask(image_rgb: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """Run U-Net and return the wall mask (H, W) binary uint8."""
    import torch
    from detect_unet import build_model, load_checkpoint, get_val_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    h_orig, w_orig = image_rgb.shape[:2]
    transform = get_val_transform(1024)
    augmented = transform(image=image_rgb)
    inp = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_idx = probs.max(dim=1)
        pred = pred_idx.squeeze(0).cpu().numpy()
        low_conf = max_prob.squeeze(0).cpu().numpy() < 0.5
        pred[low_conf] = 0

    del inp, logits, probs, max_prob, pred_idx

    pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig),
                           interpolation=cv2.INTER_NEAREST)
    wall_mask = (pred_full == 1).astype(np.uint8) * 255
    return wall_mask


def make_unet_overlay(wall_mask: np.ndarray) -> np.ndarray:
    """Create a red RGBA overlay from the U-Net wall mask."""
    h, w = wall_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    mask_bool = wall_mask > 0
    overlay[mask_bool, 0] = 0.9   # red
    overlay[mask_bool, 1] = 0.15
    overlay[mask_bool, 2] = 0.15
    overlay[mask_bool, 3] = 0.5
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Debug enclosed regions + U-Net wall mask overlay"
    )
    parser.add_argument("image", help="Path to floorplan image")
    parser.add_argument("--checkpoint", default="weights/epoch_040.pth",
                        help="U-Net checkpoint path")
    parser.add_argument("--black-threshold", type=int, default=60,
                        help="Grayscale threshold for black outlines")
    parser.add_argument("--no-unet", action="store_true",
                        help="Skip U-Net, show only enclosed regions")
    args = parser.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Cannot load image: {args.image}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Enclosed regions
    num_labels, labels = _build_enclosed_regions(
        img_bgr, black_threshold=args.black_threshold,
    )
    region_overlay = make_region_colormap(num_labels, labels)
    print(f"Found {num_labels - 1} enclosed regions")

    # 2. U-Net wall mask
    wall_mask = None
    unet_overlay = None
    if not args.no_unet:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}, skipping U-Net")
        else:
            wall_mask = run_unet_mask(img_rgb, args.checkpoint)
            unet_overlay = make_unet_overlay(wall_mask)
            print(f"U-Net wall mask: {int(np.sum(wall_mask > 0))} wall pixels")

    # 3. Plot
    has_unet = unet_overlay is not None
    ncols = 4 if has_unet else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 8))
    fig.suptitle("Enclosed Regions Debug", fontsize=16, fontweight="bold")

    # Panel 1: Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 2: Enclosed regions
    axes[1].imshow(img_rgb)
    axes[1].imshow(region_overlay)
    axes[1].set_title(f"Enclosed Regions ({num_labels - 1})")
    axes[1].axis("off")

    if has_unet:
        # Panel 3: U-Net wall mask
        axes[2].imshow(img_rgb)
        axes[2].imshow(unet_overlay)
        axes[2].set_title("U-Net Wall Mask")
        axes[2].axis("off")

        # Panel 4: Combined — regions + unet
        axes[3].imshow(img_rgb)
        axes[3].imshow(unet_overlay)
        axes[3].imshow(region_overlay)

        # Add coverage % text per region
        wall_bin = wall_mask > 0
        for label_id in range(1, num_labels):
            region_mask = labels == label_id
            area = int(np.sum(region_mask))
            if area < 50:
                continue
            coverage = int(np.sum(wall_bin & region_mask)) / area
            ys, xs = np.where(region_mask)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            color = "lime" if coverage >= 0.5 else "red"
            axes[3].text(cx, cy, f"{coverage:.0%}", fontsize=7,
                         color=color, fontweight="bold",
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15",
                                   facecolor="black", alpha=0.6))
        axes[3].set_title("Combined (green ≥50%, red <50%)")
        axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    # 4. Generate grey wall image if U-Net was run
    if has_unet:
        grey_img = build_grey_wall_image(
            img_bgr, labels, num_labels, wall_mask,
            black_threshold=args.black_threshold,
        )
        out_path = os.path.splitext(args.image)[0] + "_grey_walls.png"
        cv2.imwrite(out_path, grey_img)
        print(f"Grey wall image saved: {out_path}")

        # Show it
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 10))
        ax2.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Grey Wall Image (input for color-based detector)")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
