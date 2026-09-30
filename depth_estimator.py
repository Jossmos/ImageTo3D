"""Depth estimator using the official Depth-Anything-V2 implementation.

This module uses the code in `Depth-Anything-V2/depth_anything_v2/dpt.py`.
If you cloned the repository into the workspace root (folder
`Depth-Anything-V2`), this file will import the `DepthAnythingV2` class and
load a local checkpoint saved under `checkpoints/`.

Usage:
    est = DepthEstimator()
    depth = est.estimate('path/to/image.jpg')  # numpy float32 HxW in 0..1 on CUDA
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - runtime environment may lack torch
    torch = None  # type: ignore

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

# Ensure the cloned repo path is importable when running from workspace root
REPO_DIR = os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2')
if not os.path.isdir(REPO_DIR):
    # also allow plain folder name if script run from workspace root
    alt = os.path.join(os.getcwd(), 'Depth-Anything-V2')
    if os.path.isdir(alt):
        REPO_DIR = alt

if os.path.isdir(REPO_DIR):
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception:
    DepthAnythingV2 = None  # type: ignore


CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


class DepthEstimator:
    def __init__(self, encoder: str = 'vitl', checkpoint: str = 'checkpoints/depth_anything_v2_vitl.pth'):
        if torch is None:
            raise RuntimeError('torch is required for DepthEstimator')

        if DepthAnythingV2 is None:
            raise RuntimeError('Depth-Anything-V2 code not found. Clone the repo at Depth-Anything-V2 in the workspace.')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if encoder not in CONFIGS:
            raise ValueError(f'Unknown encoder: {encoder}')

        cfg = CONFIGS[encoder]
        self.model = DepthAnythingV2(**cfg)

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Please download it to checkpoints/.')

        # Load checkpoint (map to CPU first, then move)
        state = torch.load(checkpoint, map_location='cpu')
        # Some checkpoints are dicts with a 'model' key
        if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            sd = state['model']
        else:
            sd = state

        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device).eval()

    def estimate(self, image: Union[str, np.ndarray, Image.Image]):
        if isinstance(image, str):
            if Image is None:
                raise RuntimeError('Pillow required to load image paths')
            img = cv2.imread(image)
            if img is None:
                raise RuntimeError(f'Failed to read image: {image}')
        elif isinstance(image, np.ndarray):
            img = image
        elif Image is not None and isinstance(image, Image.Image):
            # convert PIL -> numpy BGR
            arr = np.array(image.convert('RGB'))
            img = arr[:, :, ::-1]
        else:
            raise TypeError('Unsupported image type')

        # Ensure contiguous BGR uint8 image for model.infer_image
        img = img.astype(np.uint8)

        with torch.no_grad():
            depth = self.model.infer_image(img)

        depth = np.asarray(depth, dtype=np.float32)
        mn = float(depth.min())
        mx = float(depth.max())
        if mx - mn > 1e-6:
            depth = (depth - mn) / (mx - mn)
        else:
            depth = depth - mn

        return depth

    def estimate_batch(self, images: Sequence[Union[str, np.ndarray, Image.Image]]):
        return [self.estimate(img) for img in images]

    @staticmethod
    def colorize_depth(depth: np.ndarray) -> np.ndarray:
        d = np.clip(depth, 0.0, 1.0)
        h, w = d.shape
        cmap = np.zeros((h, w, 3), dtype=np.uint8)
        cmap[..., 0] = (np.clip(2 * d - 0.0, 0, 1) * 255).astype(np.uint8)
        cmap[..., 1] = (np.clip(1.0 - np.abs(d - 0.5) * 2.0, 0, 1) * 255).astype(np.uint8)
        cmap[..., 2] = (np.clip(1.0 - d * 2.0, 0, 1) * 255).astype(np.uint8)
        return cmap

    @staticmethod
    def save_depth_png16(path: Union[str, os.PathLike], depth: np.ndarray) -> None:
        from PIL import Image as PILImage

        d = np.clip(depth, 0.0, 1.0)
        arr16 = (d * 65535.0).astype(np.uint16)
        img = PILImage.fromarray(arr16, mode='I;16')
        img.save(str(path))

