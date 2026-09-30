"""Standalone video-to-3D reconstruction pipeline.

This module is intentionally isolated from the existing Blender image pipeline.
It provides an end-to-end workflow:
1. Extract frames from a video at a configurable interval.
2. Run local Depth Anything V2 inference on each frame and save raw depth maps.
3. Run COLMAP reconstruction through subprocess calls.
4. Fuse the depth maps with COLMAP camera poses using Open3D TSDF integration.
5. Reconstruct a watertight mesh with Poisson surface reconstruction and export it.

The pipeline prefers a Hugging Face / transformers depth model when available,
and falls back to the local official Depth-Anything-V2 repository that already
ships with this workspace.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise RuntimeError("open3d is required for the video-to-3D pipeline") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise RuntimeError("torch is required for the video-to-3D pipeline") from exc

from PIL import Image
import tempfile
import os

# Note: remove_background from image_pipeline is available if needed in the future
# For video reconstruction, we use depth-based masking instead (see generate_masked_frames)
try:
    from image_pipeline import remove_background
    REMOVE_BG_AVAILABLE = True
except Exception:
    REMOVE_BG_AVAILABLE = False


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@dataclass(slots=True)
class FrameRecord:
    """Bookkeeping for a single extracted frame."""

    index: int
    timestamp_sec: float
    image_path: Path
    depth_path: Path | None = None


@dataclass(slots=True)
class COLMAPCamera:
    """Parsed camera entry from COLMAP text models."""

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def to_o3d_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        fx, fy, cx, cy = _camera_params_to_intrinsics(self.model, self.params)
        return o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)


@dataclass(slots=True)
class COLMAPImagePose:
    """Parsed image pose entry from COLMAP text models."""

    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    def to_extrinsic(self) -> np.ndarray:
        rotation = _qvec_to_rotmat(self.qvec)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = self.tvec
        return extrinsic


@dataclass(slots=True)
class COLMAPPoint3D:
    """Parsed sparse 3D point entry from COLMAP text models."""

    point_id: int
    xyz: np.ndarray
    track_image_ids: tuple[int, ...]


class DepthAnythingV2Predictor:
    """Depth Anything V2 inference wrapper with transformers-first loading."""

    def __init__(
        self,
        *,
        depth_model_id: str,
        depth_checkpoint: str | Path,
        encoder: str = "vitl",
        input_size: int = 518,
        device: str | None = None,
    ) -> None:
        self.depth_model_id = depth_model_id
        self.depth_checkpoint = Path(depth_checkpoint)
        self.encoder = encoder
        self.input_size = input_size
        self.device = device or self._auto_device()

        self._hf_processor = None
        self._hf_model = None
        self._local_model = None
        self._local_repo_path = Path(__file__).resolve().parent / "Depth-Anything-V2"

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_transformers_model(self) -> bool:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception:
            return False

        try:
            self._hf_processor = AutoImageProcessor.from_pretrained(self.depth_model_id)
            self._hf_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_id)
            self._hf_model.to(self.device).eval()
            return True
        except Exception:
            self._hf_processor = None
            self._hf_model = None
            return False

    def _load_local_model(self) -> None:
        if not self._local_repo_path.is_dir():
            raise FileNotFoundError(
                f"Local Depth-Anything-V2 repo not found at {self._local_repo_path}"
            )

        if str(self._local_repo_path) not in sys.path:
            sys.path.insert(0, str(self._local_repo_path))

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except Exception as exc:
            raise RuntimeError("Failed to import the local Depth-Anything-V2 implementation") from exc

        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unknown Depth Anything encoder: {self.encoder}")

        if not self.depth_checkpoint.is_file():
            raise FileNotFoundError(f"Depth checkpoint not found: {self.depth_checkpoint}")

        model = DepthAnythingV2(**MODEL_CONFIGS[self.encoder])
        state_dict = torch.load(self.depth_checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        self._local_model = model.to(self.device).eval()

    def _ensure_model(self) -> None:
        if self._hf_model is not None or self._local_model is not None:
            return

        if not self._load_transformers_model():
            self._load_local_model()

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Predict a raw depth map for one BGR frame."""

        self._ensure_model()
        height, width = image_bgr.shape[:2]

        if self._hf_model is not None and self._hf_processor is not None:
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            inputs = self._hf_processor(images=pil_image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            outputs = self._hf_model(**inputs)
            predicted_depth = outputs.predicted_depth
            if predicted_depth.ndim == 3:
                predicted_depth = predicted_depth.unsqueeze(1)

            depth = torch.nn.functional.interpolate(
                predicted_depth,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            return depth[0].float().cpu().numpy().astype(np.float32)

        if self._local_model is None:
            raise RuntimeError("No depth model loaded; both transformers and local fallback failed.")
        depth = self._local_model.infer_image(image_bgr, self.input_size)
        return np.asarray(depth, dtype=np.float32)


class VideoTo3DPipeline:
    """End-to-end video-to-mesh reconstruction pipeline."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        working_dir: str | Path,
        output_mesh_path: str | Path = "final_output.obj",
        frame_interval_seconds: float = 1.0,
        depth_model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
        depth_checkpoint: str | Path = Path("checkpoints") / "depth_anything_v2_vitl.pth",
        depth_encoder: str = "vitl",
        input_size: int = 518,
        colmap_executable: str = "colmap",
        colmap_camera_model: str = "SIMPLE_RADIAL",
        matcher: str = "sequential",
        use_gpu: bool = False,
        min_metric_depth_m: float = 0.3,
        max_metric_depth_m: float = 8.0,
        depth_trunc_m: float | None = None,
        tsdf_voxel_length: float = 0.02,
        poisson_depth: int = 9,
        depth_align_percentile: float = 95.0,
        clean_working_dir: bool = False,
        max_frames: int | None = None,
        max_close_depth_m: float = 2.0,
        skip_colmap_alignment: bool = False,
    ) -> None:
        self.video_path = Path(video_path)
        self.working_dir = Path(working_dir)
        output_mesh_path = Path(output_mesh_path)
        self.output_mesh_path = output_mesh_path if output_mesh_path.is_absolute() else self.video_path.parent / output_mesh_path
        self.frame_interval_seconds = frame_interval_seconds
        self.depth_model_id = depth_model_id
        self.depth_checkpoint = Path(depth_checkpoint)
        self.depth_encoder = depth_encoder
        self.input_size = input_size
        self.colmap_executable = self._resolve_colmap_executable(colmap_executable)
        self.colmap_camera_model = colmap_camera_model
        self.matcher = matcher
        self.use_gpu = use_gpu
        self.min_metric_depth_m = min_metric_depth_m
        self.max_metric_depth_m = max_metric_depth_m
        self.depth_trunc_m = depth_trunc_m
        self.tsdf_voxel_length = tsdf_voxel_length
        self.poisson_depth = poisson_depth
        self.depth_align_percentile = depth_align_percentile
        self.clean_working_dir = clean_working_dir
        self.max_frames = max_frames
        self.max_close_depth_m = max_close_depth_m
        self.skip_colmap_alignment = skip_colmap_alignment

        if self.depth_trunc_m is None:
            self.depth_trunc_m = self.tsdf_voxel_length * 500.0

        self.video_parent_dir = self.video_path.parent
        self.frames_dir = self.video_parent_dir / "frames"
        self.depths_dir = self.video_parent_dir / "depth"
        self.frames_masked_dir = self.video_parent_dir / "frames_masked"
        self.colmap_dir = self.working_dir / "colmap_output"
        self.colmap_db_path = self.colmap_dir / "database.db"
        self.colmap_sparse_dir = self.colmap_dir / "sparse"
        self.colmap_text_dir = self.colmap_dir / "sparse_text"
        self.colmap_undistorted_dir = self.colmap_dir / "undistorted"
        self.predictions_manifest = self.working_dir / "frame_manifest.json"

        self.depth_predictor = DepthAnythingV2Predictor(
            depth_model_id=self.depth_model_id,
            depth_checkpoint=self.depth_checkpoint,
            encoder=self.depth_encoder,
            input_size=self.input_size,
        )

    def run(self) -> Path:
        """Execute the full pipeline and return the final mesh path."""

        self._validate_inputs()
        self._prepare_working_tree()

        frames = self.extract_frames()
        self.generate_depth_maps(frames)
        self.generate_motion_visualizations(frames)
        # Generate masked frames (remove.bg preferred, else depth mask)
        self.generate_masked_frames(frames, use_removebg=REMOVE_BG_AVAILABLE)
        sparse_model_dir = self.run_colmap_reconstruction()
        text_model_dir = self.convert_colmap_model_to_text(sparse_model_dir)
        final_mesh = self.fuse_and_mesh(text_model_dir, frames)

        if final_mesh != self.output_mesh_path:
            self.output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
            if final_mesh.is_file() and final_mesh != self.output_mesh_path:
                shutil.copy2(final_mesh, self.output_mesh_path)
                final_mesh = self.output_mesh_path

        return final_mesh

    def _validate_inputs(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if shutil.which(self.colmap_executable) is None:
            fallback_paths = [
                Path(r"C:\Program Files\COLMAP\colmap.exe"),
                Path(r"C:\Program Files (x86)\COLMAP\colmap.exe"),
            ]
            for fallback_path in fallback_paths:
                if fallback_path.is_file():
                    self.colmap_executable = str(fallback_path)
                    return
            raise FileNotFoundError(
                "COLMAP not found. Install from https://github.com/colmap/colmap/releases "
                "and add to PATH, or pass --colmap-executable with the full path."
            )

    def _prepare_working_tree(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.video_parent_dir.mkdir(parents=True, exist_ok=True)
        self.frames_masked_dir.mkdir(parents=True, exist_ok=True)

        if self.clean_working_dir:
            for path in [self.frames_dir, self.depths_dir, self.colmap_dir, self.output_mesh_path]:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.depths_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_sparse_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_text_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_undistorted_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self) -> list[FrameRecord]:
        """Extract frames at the configured temporal interval into working_dir/frames."""

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0.0:
            fps = 30.0

        frame_step = max(1, int(round(fps * self.frame_interval_seconds)))
        records: list[FrameRecord] = []
        frame_index = 0
        saved_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_step == 0:
                image_path = self.frames_dir / f"frame_{saved_index:06d}.png"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    raise RuntimeError(f"Failed to save frame: {image_path}")

                records.append(
                    FrameRecord(
                        index=saved_index,
                        timestamp_sec=frame_index / fps,
                        image_path=image_path,
                    )
                )
                saved_index += 1

                if self.max_frames is not None and saved_index >= self.max_frames:
                    break

            frame_index += 1

        capture.release()

        if not records:
            raise RuntimeError("No frames were extracted from the video")

        self._write_manifest(records, fps)
        return records

    def _write_manifest(self, records: Sequence[FrameRecord], fps: float) -> None:
        manifest = {
            "video_path": str(self.video_path),
            "fps": fps,
            "frame_interval_seconds": self.frame_interval_seconds,
            "frames": [
                {
                    "index": record.index,
                    "timestamp_sec": record.timestamp_sec,
                    "image_path": str(record.image_path),
                }
                for record in records
            ],
        }
        self.predictions_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def generate_depth_maps(self, frames: Sequence[FrameRecord]) -> None:
        """Predict and persist a raw depth .npy for each extracted frame, plus PNG visualization."""

        for record in frames:
            image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read frame for depth inference: {record.image_path}")

            depth = self.depth_predictor.predict(image_bgr)
            depth_path = self.depths_dir / f"{record.image_path.stem}.npy"
            np.save(depth_path, depth.astype(np.float32))
            record.depth_path = depth_path

            # Also save a PNG visualization of the depth map
            depth_viz = depth.copy()
            # Normalize to 0-255 for visualization (handling NaNs and infinities)
            valid_mask = np.isfinite(depth_viz)
            if np.any(valid_mask):
                depth_min = np.nanmin(depth_viz[valid_mask])
                depth_max = np.nanmax(depth_viz[valid_mask])
                if depth_max > depth_min:
                    depth_viz = (depth_viz - depth_min) / (depth_max - depth_min) * 255.0
                else:
                    depth_viz = np.ones_like(depth_viz) * 128.0
            depth_viz[~valid_mask] = 0.0  # Invalid pixels as black
            depth_viz_uint8 = np.clip(depth_viz, 0, 255).astype(np.uint8)

            # Apply a colormap for better visualization
            depth_viz_color = cv2.applyColorMap(depth_viz_uint8, cv2.COLORMAP_TURBO)
            depth_png_path = self.depths_dir / f"{record.image_path.stem}.png"
            cv2.imwrite(str(depth_png_path), depth_viz_color)

    def generate_motion_visualizations(self, frames: Sequence[FrameRecord]) -> None:
        """Generate optical flow visualizations to show motion between frames."""
        
        motion_dir = self.video_parent_dir / "motion"
        motion_dir.mkdir(parents=True, exist_ok=True)

        prev_gray = None
        prev_path = None

        for i, record in enumerate(frames):
            image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and i > 0:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # Visualize flow
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(image_bgr)
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                hsv[..., 2] = 255
                flow_viz = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Also draw flow vectors on the image
                step = 20
                for y in range(0, flow.shape[0], step):
                    for x in range(0, flow.shape[1], step):
                        fx, fy = flow[y, x]
                        if mag[y, x] > 0.5:  # Only draw significant flow
                            cv2.arrowedLine(
                                flow_viz,
                                (x, y),
                                (int(x + fx), int(y + fy)),
                                (0, 255, 0),
                                1,
                                tipLength=0.3,
                            )

                motion_png_path = motion_dir / f"{record.image_path.stem}_to_{i:06d}.png"
                cv2.imwrite(str(motion_png_path), flow_viz)

                # Log motion statistics
                motion_magnitude = np.mean(mag)
                print(f"[motion] Frame {i-1} → {i}: avg motion magnitude = {motion_magnitude:.2f}")

            prev_gray = gray
            prev_path = record.image_path

    def generate_masked_frames(self, frames: Sequence[FrameRecord], use_removebg: bool = True) -> None:
        """Generate masked frames using depth-based masking with morphological cleanup.

        Masked frames are saved to `frames_masked/` next to the video.
        Note: remove.bg is NOT used here because it crops images (wrong dimensions for COLMAP).
        """
        self.frames_masked_dir.mkdir(parents=True, exist_ok=True)

        for record in frames:
            src = record.image_path
            dst = self.frames_masked_dir / f"{src.stem}.png"

            # Use depth mask (preserves original frame dimensions)
            if record.depth_path and Path(record.depth_path).is_file():
                try:
                    img = Image.open(src).convert("RGBA")
                    depth = np.load(record.depth_path)
                    mask = np.isfinite(depth) & (depth > 0) & (depth <= self.max_close_depth_m)
                    
                    # Morphological cleanup: dilate to fill holes, erode to smooth edges
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask_uint8 = mask.astype(np.uint8) * 255
                    # Close: dilate then erode (fills holes)
                    mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
                    # Open: erode then dilate (removes noise)
                    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    # Fill remaining interior holes
                    contours, _ = cv2.findContours(mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        mask_filled = np.zeros_like(mask_clean)
                        for i, contour in enumerate(contours):
                            cv2.drawContours(mask_filled, [contour], 0, 255, -1)
                        mask_clean = mask_filled
                    
                    alpha = mask_clean
                    alpha_img = Image.fromarray(alpha.astype(np.uint8))
                    rgba = img.copy()
                    rgba.putalpha(alpha_img.resize(rgba.size, resample=Image.NEAREST))
                    rgba.save(dst)
                    continue
                except Exception:
                    pass

            # Last fallback: copy original frame (no mask)
            shutil.copy2(src, dst)

    def run_colmap_reconstruction(self) -> Path:
        """Run COLMAP feature extraction, matching, mapping, and undistortion."""

        frames_path = self.frames_masked_dir if any(self.frames_masked_dir.iterdir()) else self.frames_dir

        feature_command = [
            self.colmap_executable,
            "feature_extractor",
            "--database_path",
            str(self.colmap_db_path),
            "--image_path",
            str(frames_path),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            self.colmap_camera_model,
        ]
        if self.use_gpu:
            feature_command.extend(["--SiftExtraction.use_gpu", "1"])
        self._run_command(feature_command)

        if self.matcher == "sequential":
            matcher_command = [
                self.colmap_executable,
                "sequential_matcher",
                "--database_path",
                str(self.colmap_db_path),
            ]
            if self.use_gpu:
                matcher_command.extend(["--SiftMatching.use_gpu", "1"])
        elif self.matcher == "exhaustive":
            matcher_command = [
                self.colmap_executable,
                "exhaustive_matcher",
                "--database_path",
                str(self.colmap_db_path),
            ]
            if self.use_gpu:
                matcher_command.extend(["--SiftMatching.use_gpu", "1"])
        else:
            raise ValueError(f"Unsupported matcher: {self.matcher}")

        self._run_command(matcher_command)

        self._run_command([
            self.colmap_executable,
            "mapper",
            "--database_path",
            str(self.colmap_db_path),
            "--image_path",
            str(frames_path),
            "--output_path",
            str(self.colmap_sparse_dir),
        ])

        sparse_model_dir = self._select_best_sparse_model(self.colmap_sparse_dir)

        self._run_command([
            self.colmap_executable,
            "image_undistorter",
            "--image_path",
            str(frames_path),
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            str(self.colmap_undistorted_dir),
            "--output_type",
            "COLMAP",
        ])

        return self._find_sparse_model_dir(self.colmap_undistorted_dir)

    def convert_colmap_model_to_text(self, sparse_model_dir: Path) -> Path:
        """Convert the COLMAP model to text files for easy parsing."""

        self.colmap_text_dir.mkdir(parents=True, exist_ok=True)
        self._run_command([
            self.colmap_executable,
            "model_converter",
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            str(self.colmap_text_dir),
            "--output_type",
            "TXT",
        ])

        self._require_colmap_text_model(self.colmap_text_dir)
        return self.colmap_text_dir

    def fuse_and_mesh(self, colmap_text_dir: Path, frames: Sequence[FrameRecord]) -> Path:
        """Fuse depth maps with COLMAP poses and export a watertight mesh."""

        cameras, images, points3d = _read_colmap_text_model(colmap_text_dir)
        image_lookup = {image.name: image for image in images.values()}

        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.tsdf_voxel_length,
            sdf_trunc=self.tsdf_voxel_length * 4.0,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        integrated = 0
        for record in frames:
            image_name = record.image_path.name
            colmap_image = image_lookup.get(image_name)
            if colmap_image is None:
                continue

            camera = cameras.get(colmap_image.camera_id)
            if camera is None:
                continue

            if record.depth_path is None or not record.depth_path.is_file():
                continue

            color_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if color_bgr is None:
                continue

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            depth_raw = np.load(record.depth_path).astype(np.float32)
            
            if self.skip_colmap_alignment:
                # Simple depth scaling: normalize relative depth to metric range
                depth_metric = depth_raw.copy()
                valid_mask = np.isfinite(depth_metric) & (depth_metric > 0)
                if np.any(valid_mask):
                    vmin, vmax = np.percentile(depth_metric[valid_mask], [5, 95])
                    if vmax > vmin:
                        depth_metric[valid_mask] = (depth_metric[valid_mask] - vmin) / (vmax - vmin) * (self.max_metric_depth_m - self.min_metric_depth_m) + self.min_metric_depth_m
                    depth_metric[~valid_mask] = 0.0
            else:
                # COLMAP-based alignment
                colmap_pts3d_uvd = self._get_colmap_pts_for_frame(
                    colmap_image=colmap_image,
                    camera=camera,
                    points3d=points3d,
                )
                depth_metric = self._align_depth_to_colmap(
                    depth_raw,
                    colmap_pts3d_uvd,
                    percentile=self.depth_align_percentile,
                )

            # Clip depth to max_close_depth_m to focus on foreground object
            depth_metric = np.minimum(depth_metric, self.max_close_depth_m)

            if depth_metric.shape[:2] != color_rgb.shape[:2]:
                depth_metric = cv2.resize(
                    depth_metric,
                    (color_rgb.shape[1], color_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)

            # Scale camera intrinsics to match the actual image dimensions
            intrinsic = camera.to_o3d_intrinsic()
            actual_h, actual_w = color_rgb.shape[:2]
            scale_x = actual_w / camera.width
            scale_y = actual_h / camera.height
            intrinsic.set_intrinsics(
                actual_w,
                actual_h,
                intrinsic.get_focal_length()[0] * scale_x,
                intrinsic.get_focal_length()[1] * scale_y,
                intrinsic.get_principal_point()[0] * scale_x,
                intrinsic.get_principal_point()[1] * scale_y,
            )

            color_o3d = o3d.geometry.Image(color_rgb)
            depth_o3d = o3d.geometry.Image(depth_metric)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=self.depth_trunc_m,
                convert_rgb_to_intensity=False,
            )

            tsdf_volume.integrate(
                rgbd,
                intrinsic,
                colmap_image.to_extrinsic(),
            )
            integrated += 1

        if integrated == 0:
            raise RuntimeError("No COLMAP frames could be fused into the TSDF volume")

        fused_pcd = tsdf_volume.extract_point_cloud()
        raw_fused_pcd = fused_pcd

        if len(fused_pcd.points) > 0:
            downsampled_pcd = fused_pcd.voxel_down_sample(voxel_size=self.tsdf_voxel_length)
            if len(downsampled_pcd.points) > 0:
                fused_pcd = downsampled_pcd

        # Filter out far points to keep only the main foreground object
        if len(fused_pcd.points) > 0:
            points = np.asarray(fused_pcd.points)
            distances = np.linalg.norm(points, axis=1)
            close_mask = distances <= self.max_close_depth_m
            if np.sum(close_mask) > 0:
                fused_pcd = fused_pcd.select_by_index(np.where(close_mask)[0])

        # Apply lenient outlier removal, but keep the previous cloud if filtering is too aggressive.
        if len(fused_pcd.points) > 100:
            filtered_pcd, _ = fused_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=5.0)
            if len(filtered_pcd.points) > 0:
                fused_pcd = filtered_pcd

        if len(fused_pcd.points) == 0 and len(raw_fused_pcd.points) > 0:
            fused_pcd = raw_fused_pcd

        if len(fused_pcd.points) == 0:
            raise RuntimeError("TSDF fusion produced an empty point cloud")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            fused_pcd,
            depth=self.poisson_depth,
        )

        density_values = np.asarray(densities)
        if density_values.size > 0:
            density_threshold = np.quantile(density_values, 0.02)
            remove_mask = density_values < density_threshold
            mesh.remove_vertices_by_mask(remove_mask)

        bbox = fused_pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        self.output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(self.output_mesh_path), mesh):
            raise RuntimeError(f"Failed to write mesh to {self.output_mesh_path}")

        return self.output_mesh_path

    def _align_depth_to_colmap(
        self,
        depth_rel: np.ndarray,
        colmap_pts3d_uvd: np.ndarray,
        percentile: float = 95.0,
    ) -> np.ndarray:
        """Align relative Depth Anything depth to COLMAP metric depth."""

        depth_rel = np.asarray(depth_rel, dtype=np.float32)
        colmap_pts3d_uvd = np.asarray(colmap_pts3d_uvd, dtype=np.float32)

        if colmap_pts3d_uvd.shape[0] < 4:
            lo = float(np.percentile(depth_rel, 1.0))
            hi = float(np.percentile(depth_rel, 99.0))
            normalized = np.clip((depth_rel - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            return (
                self.min_metric_depth_m
                + normalized * (self.max_metric_depth_m - self.min_metric_depth_m)
            ).astype(np.float32)

        us = colmap_pts3d_uvd[:, 0].astype(np.int32)
        vs = colmap_pts3d_uvd[:, 1].astype(np.int32)
        d_metric = colmap_pts3d_uvd[:, 2].astype(np.float32)

        height, width = depth_rel.shape
        valid = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
        us = us[valid]
        vs = vs[valid]
        d_metric = d_metric[valid]

        if us.shape[0] < 4:
            return depth_rel

        d_pred = depth_rel[vs, us]
        design = np.stack([d_pred, np.ones_like(d_pred)], axis=1)
        result, _, _, _ = np.linalg.lstsq(design, d_metric, rcond=None)
        scale, shift = float(result[0]), float(result[1])

        if scale <= 0.0:
            design[:, 0] = -d_pred
            result, _, _, _ = np.linalg.lstsq(design, d_metric, rcond=None)
            scale, shift = float(result[0]), float(result[1])
            depth_rel = -depth_rel

        aligned = scale * depth_rel + shift

        if np.any(np.isfinite(aligned)):
            lo = float(np.percentile(aligned[np.isfinite(aligned)], 100.0 - percentile))
            hi = float(np.percentile(aligned[np.isfinite(aligned)], percentile))
            if hi > lo:
                aligned = np.clip(aligned, lo, hi)

        return np.clip(aligned, 0.01, None).astype(np.float32)

    @staticmethod
    def _resolve_colmap_executable(colmap_executable: str) -> str:
        """Resolve COLMAP executable path, checking common Windows locations."""
        # If it's already a full path or in PATH, use it as-is
        if Path(colmap_executable).is_absolute():
            return colmap_executable
        
        # Try to find colmap in PATH first
        from shutil import which
        colmap_in_path = which(colmap_executable)
        if colmap_in_path:
            return colmap_in_path
        
        # Check common Windows installation locations
        candidates = [
            Path("C:/COLMAP/bin/colmap.exe"),
            Path("C:/Program Files/COLMAP/bin/colmap.exe"),
            Path("C:/Program Files (x86)/COLMAP/bin/colmap.exe"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # If not found, return the original and let subprocess raise an error
        return colmap_executable

    def _run_command(self, command: Sequence[str]) -> None:
        result = subprocess.run(
            list(command),
            cwd=str(self.working_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "COLMAP command failed: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                    cmd=" ".join(command),
                    stdout=result.stdout.strip(),
                    stderr=result.stderr.strip(),
                )
            )

    @staticmethod
    def _select_best_sparse_model(sparse_root: Path) -> Path:
        model_dirs = [path for path in sparse_root.iterdir() if path.is_dir()]
        if not model_dirs:
            raise RuntimeError(f"No sparse reconstruction directory found in {sparse_root}")

        def model_score(path: Path) -> tuple[int, int]:
            cameras = path / "cameras.bin"
            images = path / "images.bin"
            points = path / "points3D.bin"
            score = sum(int(file.exists()) for file in (cameras, images, points))
            return score, int(path.name) if path.name.isdigit() else -1

        model_dirs.sort(key=model_score, reverse=True)
        best = model_dirs[0]
        if not (best / "cameras.bin").exists() and not (best / "cameras.txt").exists():
            raise RuntimeError(f"Selected sparse model is incomplete: {best}")
        return best

    @staticmethod
    def _find_sparse_model_dir(root: Path) -> Path:
        candidates = []
        for path in root.rglob("cameras.bin"):
            candidates.append(path.parent)
        if candidates:
            return candidates[0]
        for path in root.rglob("cameras.txt"):
            candidates.append(path.parent)
        if candidates:
            return candidates[0]
        raise RuntimeError(f"No sparse COLMAP model found under {root}")

    @staticmethod
    def _require_colmap_text_model(text_dir: Path) -> None:
        required = [text_dir / "cameras.txt", text_dir / "images.txt"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"COLMAP text export missing files: {', '.join(str(p) for p in missing)}")

    def _get_colmap_pts_for_frame(
        self,
        *,
        colmap_image: COLMAPImagePose,
        camera: COLMAPCamera,
        points3d: dict[int, COLMAPPoint3D],
    ) -> np.ndarray:
        """Project COLMAP sparse 3D points visible in the current frame into image UVD."""

        fx, fy, cx, cy = _camera_params_to_intrinsics(camera.model, camera.params)
        rotation = _qvec_to_rotmat(colmap_image.qvec)
        translation = np.asarray(colmap_image.tvec, dtype=np.float64).reshape(3, 1)

        pts_uvd: list[tuple[float, float, float]] = []
        for point in points3d.values():
            if colmap_image.image_id not in point.track_image_ids:
                continue

            world_point = np.asarray(point.xyz, dtype=np.float64).reshape(3, 1)
            camera_point = rotation @ world_point + translation
            depth = float(camera_point[2, 0])
            if depth <= 0.0:
                continue

            u = fx * float(camera_point[0, 0]) / depth + cx
            v = fy * float(camera_point[1, 0]) / depth + cy
            pts_uvd.append((u, v, depth))

        if not pts_uvd:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(pts_uvd, dtype=np.float32)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    if qvec.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {qvec.shape}")

    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _camera_params_to_intrinsics(model: str, params: Sequence[float]) -> tuple[float, float, float, float]:
    params = tuple(float(value) for value in params)
    if model == "SIMPLE_PINHOLE":
        if len(params) < 3:
            raise ValueError(f"Invalid SIMPLE_PINHOLE params: {params}")
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        if len(params) < 3:
            raise ValueError(f"Invalid {model} params: {params}")
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        if len(params) < 4:
            raise ValueError(f"Invalid {model} params: {params}")
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    raise NotImplementedError(f"Unsupported COLMAP camera model: {model}")


def _read_colmap_text_model(model_dir: Path) -> tuple[dict[int, COLMAPCamera], dict[int, COLMAPImagePose], dict[int, COLMAPPoint3D]]:
    cameras = _read_cameras_txt(model_dir / "cameras.txt")
    images = _read_images_txt(model_dir / "images.txt")
    points3d = _read_points3d_txt(model_dir / "points3D.txt")
    return cameras, images, points3d


def _read_cameras_txt(path: Path) -> dict[int, COLMAPCamera]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing COLMAP cameras file: {path}")

    cameras: dict[int, COLMAPCamera] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = tuple(float(value) for value in parts[4:])
            cameras[camera_id] = COLMAPCamera(camera_id, model, width, height, params)
    return cameras


def _read_images_txt(path: Path) -> dict[int, COLMAPImagePose]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing COLMAP images file: {path}")

    images: dict[int, COLMAPImagePose] = {}
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]

    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed COLMAP images.txt line: {line}")

        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        images[image_id] = COLMAPImagePose(image_id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name)

        if index < len(lines):
            index += 1

    return images


def _read_points3d_txt(path: Path) -> dict[int, COLMAPPoint3D]:
    if not path.is_file():
        return {}

    points3d: dict[int, COLMAPPoint3D] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            track_entries = parts[8:]
            track_image_ids = tuple(int(track_entries[i]) for i in range(0, len(track_entries), 2))
            points3d[point_id] = COLMAPPoint3D(point_id=point_id, xyz=xyz, track_image_ids=track_image_ids)

    return points3d


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone video-to-3D reconstruction pipeline")
    parser.add_argument("--video-path", required=True, help="Path to the input video file")
    parser.add_argument("--working-dir", required=True, help="Directory used for frames, depths, and COLMAP output")
    parser.add_argument("--output-mesh", default="final_output.obj", help="Final mesh path (.obj or .stl)")
    parser.add_argument("--frame-interval", type=float, default=1.0, help="Extract one frame every N seconds")
    parser.add_argument("--depth-model-id", default="depth-anything/Depth-Anything-V2-Large-hf", help="Transformers model id")
    parser.add_argument("--depth-checkpoint", default=str(Path("checkpoints") / "depth_anything_v2_vitl.pth"), help="Local Depth Anything checkpoint path")
    parser.add_argument("--depth-encoder", default="vitl", choices=sorted(MODEL_CONFIGS.keys()), help="Local Depth Anything encoder")
    parser.add_argument("--input-size", type=int, default=518, help="Depth model input size")
    parser.add_argument("--colmap-executable", default="colmap", help="COLMAP executable name or full path")
    parser.add_argument("--colmap-camera-model", default="SIMPLE_RADIAL", help="Camera model used during COLMAP feature extraction")
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"], help="COLMAP matching strategy")
    parser.add_argument("--use-gpu", action="store_true", help="Hint COLMAP to use GPU features when supported")
    parser.add_argument("--min-depth-m", type=float, default=0.3, help="Minimum pseudo-metric depth used for TSDF fusion")
    parser.add_argument("--max-depth-m", type=float, default=8.0, help="Maximum pseudo-metric depth used for TSDF fusion")
    parser.add_argument("--depth-trunc-m", type=float, default=0.0, help="Depth truncation distance; 0 = auto (5x voxel size)")
    parser.add_argument("--tsdf-voxel-length", type=float, default=0.02, help="TSDF voxel size in pseudo-meters")
    parser.add_argument("--poisson-depth", type=int, default=9, help="Poisson reconstruction depth")
    parser.add_argument("--depth-align-percentile", type=float, default=95.0, help="Robust percentile used for COLMAP depth alignment")
    parser.add_argument("--max-close-depth-m", type=float, default=2.0, help="Maximum depth to consider for foreground object (cuts far geometry)")
    parser.add_argument("--skip-colmap-alignment", action="store_true", help="Skip COLMAP-based depth alignment and use simple scaling instead")
    parser.add_argument("--clean", action="store_true", help="Delete previous working outputs before running")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on the number of extracted frames")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pipeline = VideoTo3DPipeline(
        video_path=args.video_path,
        working_dir=args.working_dir,
        output_mesh_path=args.output_mesh,
        frame_interval_seconds=args.frame_interval,
        depth_model_id=args.depth_model_id,
        depth_checkpoint=args.depth_checkpoint,
        depth_encoder=args.depth_encoder,
        input_size=args.input_size,
        colmap_executable=args.colmap_executable,
        colmap_camera_model=args.colmap_camera_model,
        matcher=args.matcher,
        use_gpu=args.use_gpu,
        min_metric_depth_m=args.min_depth_m,
        max_metric_depth_m=args.max_depth_m,
        depth_trunc_m=args.depth_trunc_m if args.depth_trunc_m > 0 else None,
        tsdf_voxel_length=args.tsdf_voxel_length,
        poisson_depth=args.poisson_depth,
        depth_align_percentile=args.depth_align_percentile,
        clean_working_dir=args.clean,
        max_frames=args.max_frames,
        max_close_depth_m=args.max_close_depth_m,
        skip_colmap_alignment=args.skip_colmap_alignment,
    )

    try:
        final_mesh = pipeline.run()
    except Exception as exc:
        print(f"[video-to-3d][error] {exc}", file=sys.stderr)
        return 1

    print(f"[video-to-3d][success] Mesh written to: {final_mesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
