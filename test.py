#!/usr/bin/env python3
"""
inference_unet.py — Load a UNet floor-plan segmentation checkpoint and run
inference on every image in a directory, displaying a colour overlay.

Usage
-----
    python inference_unet.py \
        --checkpoint  path/to/epoch_010.pth   (or best_unet.pth) \
        --images      path/to/images_dir \
        --img-size    1024 \
        --alpha       0.45 \
        --save-dir    results          (optional: save overlay PNGs here)

The checkpoint can be either:
  • An epoch checkpoint  (dict with key 'model')
  • A bare state_dict    (best_unet.pth)

Close the Matplotlib window to advance to the next image,
or use --no-display to only save results.
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════════
# Config — must match training notebook
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ["background", "wall", "window", "door"]
NUM_CLASSES = 4
ENCODER = "resnet34"

# RGB palette matching the training notebook
PALETTE = np.array(
    [
        [0, 0, 0],        # 0 background — black
        [200, 0, 0],      # 1 wall       — red
        [0, 200, 0],      # 2 window     — green
        [0, 160, 200],    # 3 door       — cyan
    ],
    dtype=np.uint8,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(num_classes: int = NUM_CLASSES, encoder: str = ENCODER) -> torch.nn.Module:
    """Instantiate the same SMP UNet architecture used during training."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,       # we load our own weights
        in_channels=3,
        classes=num_classes,
        activation=None,            # raw logits
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """Load either an epoch checkpoint dict or a bare state_dict."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        # Epoch checkpoint (has 'model', 'optimizer', 'scheduler', …)
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", "?")
        miou = state.get("best_val_iou", None)
        print(f"✅  Loaded epoch checkpoint  epoch={epoch}"
              + (f"  best_val_mIoU={miou:.4f}" if miou else ""))
    else:
        # Bare state dict (best_unet.pth)
        model.load_state_dict(state)
        print("✅  Loaded bare state_dict (best_unet.pth style)")
    model.eval()
    return model


def get_val_transform(img_size: int):
    """Deterministic val/test transform — resize + ImageNet normalise."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert single-channel class mask → RGB colour image (H, W, 3)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE):
        rgb[mask == cls_id] = color
    return rgb


def overlay_prediction(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend the colour mask on top of the original image using Pillow."""
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize mask to original image dimensions (nearest-neighbour)
    colour_rgb = colorize_mask(mask)
    colour_pil = Image.fromarray(colour_rgb).resize(
        (w_orig, h_orig), resample=Image.NEAREST
    )
    mask_pil = Image.fromarray(mask.astype(np.uint8)).resize(
        (w_orig, h_orig), resample=Image.NEAREST
    )

    base_pil = Image.fromarray(image_rgb)
    blended = Image.blend(base_pil, colour_pil, alpha=alpha)

    # Only apply blend where mask is not background
    fg_mask = np.array(mask_pil) != 0
    result = np.array(base_pil)
    result[fg_mask] = np.array(blended)[fg_mask]
    return result


def display_results(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    fname: str,
) -> None:
    """Show original, coloured mask, and overlay side-by-side with Matplotlib."""
    h_orig, w_orig = image_rgb.shape[:2]

    colour_rgb = colorize_mask(mask)
    colour_resized = np.array(
        Image.fromarray(colour_rgb).resize((w_orig, h_orig), resample=Image.NEAREST)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(fname, fontsize=12)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(colour_resized)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    # Legend patches
    patches = [
        mpatches.Patch(color=tuple(c / 255 for c in PALETTE[i]), label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.show()


def collect_images(directory: str) -> list[str]:
    """Glob all image files in a directory (non-recursive)."""
    paths = sorted(
        p
        for p in glob.glob(os.path.join(directory, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="UNet floor-plan inference — overlay segmentation on images",
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to .pth checkpoint (epoch_*.pth or best_unet.pth)",
    )
    parser.add_argument(
        "--images", "-i", required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--img-size", type=int, default=1024,
        help="Resize dimension for model input (default: 1024, must match training)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45,
        help="Overlay transparency 0-1 (default: 0.45)",
    )
    parser.add_argument(
        "--save-dir", "-s", default=None,
        help="If set, save overlay images to this directory",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't show Matplotlib windows (useful for headless / save-only mode)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device, e.g. 'cpu' or 'cuda:0' (auto-detected if omitted)",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model()
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # ── Images ────────────────────────────────────────────────────────────────
    image_paths = collect_images(args.images)
    if not image_paths:
        print(f"⚠️  No images found in '{args.images}'")
        sys.exit(1)
    print(f"Found {len(image_paths)} image(s) in '{args.images}'")

    # ── Optional save dir ─────────────────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving overlays to '{args.save_dir}'")

    # ── Transform ─────────────────────────────────────────────────────────────
    transform = get_val_transform(args.img_size)

    # ── Inference loop ────────────────────────────────────────────────────────
    for idx, img_path in enumerate(image_paths, 1):
        fname = os.path.basename(img_path)

        # Read image with Pillow, convert to RGB numpy array
        try:
            raw_rgb = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [{idx}/{len(image_paths)}] SKIP (unreadable): {fname} — {e}")
            continue

        h_orig, w_orig = raw_rgb.shape[:2]

        # Preprocess
        augmented = transform(image=raw_rgb)
        inp = augmented["image"].unsqueeze(0).to(device)  # [1, 3, H, W]

        # Forward pass
        with torch.no_grad():
            logits = model(inp)                                        # [1, C, H, W]
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()      # [H, W]

        # Build overlay at original resolution
        overlay = overlay_prediction(raw_rgb, pred, alpha=args.alpha)

        # Per-class pixel stats
        unique, counts = np.unique(pred, return_counts=True)
        total_px = pred.size
        stats = "  ".join(
            f"{CLASS_NAMES[c]}:{cnt/total_px*100:.1f}%"
            for c, cnt in zip(unique, counts)
        )
        print(f"  [{idx}/{len(image_paths)}] {fname}  ({w_orig}×{h_orig})  {stats}")

        # Save overlay as PNG via Pillow
        if args.save_dir:
            out_name = f"overlay_{os.path.splitext(fname)[0]}.png"
            out_path = os.path.join(args.save_dir, out_name)
            Image.fromarray(overlay).save(out_path)

        # Display with Matplotlib
        if not args.no_display:
            display_results(raw_rgb, pred, overlay, fname)

    print("\n✅  Done!")


if __name__ == "__main__":
    main()