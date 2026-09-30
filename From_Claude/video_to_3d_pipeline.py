"""
video_to_3d_pipeline.py
=======================
Independent end-to-end pipeline for reconstructing a 3D mesh (.obj) from a
video file using Depth Anything V2, COLMAP, and Open3D.

This module is completely isolated from any existing Blender / single-image
pipeline present in the parent project.  It exposes one public class
(`VideoTo3DPipeline`) and an argparse-based CLI entry-point so it can be
invoked directly:

    python video_to_3d_pipeline.py --video my_video.mp4 --working_dir ./out

Author : Expert CV / Python implementation
License: MIT
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party  (heavy imports are deferred inside methods to keep top-level
# import time low and to avoid polluting the parent project's namespace when
# this file is imported rather than run as __main__)
# ---------------------------------------------------------------------------
# cv2, numpy, torch, transformers, open3d  →  imported lazily inside methods

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tunable parameters for the VideoTo3DPipeline.

    Attributes
    ----------
    frame_interval_sec : float
        Extract one frame every *frame_interval_sec* seconds.
    depth_model_name : str
        Hugging Face model ID for Depth Anything V2.
    colmap_exe : str
        Name / full path of the COLMAP executable.  Resolved via shutil.which.
    tsdf_voxel_length : float
        Voxel size (metres) for Open3D TSDF volume integration.
    tsdf_sdf_trunc : float
        Truncation distance for the TSDF volume.
    poisson_depth : int
        Octree depth for Poisson surface reconstruction.
    depth_align_percentile : float
        Percentile used for robust scale/shift alignment (0–100).
    device : str
        Torch device string, e.g. "cuda" or "cpu".
    """

    frame_interval_sec: float = 1.0
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"
    colmap_exe: str = "colmap"
    tsdf_voxel_length: float = 0.02
    tsdf_sdf_trunc: float = 0.04
    poisson_depth: int = 9
    depth_align_percentile: float = 95.0
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Small helper: read COLMAP binary/text model
# ---------------------------------------------------------------------------

@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]          # fx, fy, cx, cy (+ distortion)


@dataclass
class ColmapImage:
    image_id: int
    qvec: List[float]            # quaternion [qw, qx, qy, qz]
    tvec: List[float]            # translation [tx, ty, tz]
    camera_id: int
    name: str                    # filename, e.g. "frame_0042.png"


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class VideoTo3DPipeline:
    """End-to-end pipeline: video file → 3D mesh (.obj).

    Steps
    -----
    1. Frame extraction          (OpenCV)
    2. Depth map generation      (Depth Anything V2 via HuggingFace transformers)
    3. Camera pose estimation    (COLMAP via subprocess)
    4. Scale-aware depth fusion  (scale+shift alignment per frame)
    5. TSDF volume integration   (Open3D)
    6. Poisson surface recon     (Open3D)
    7. Mesh export               (.obj)

    Parameters
    ----------
    video_path : str | Path
        Path to the input video file.
    working_dir : str | Path
        Root folder for all intermediate and final outputs.
    config : PipelineConfig, optional
        Tunable parameters.  Defaults are used when omitted.
    """

    def __init__(
        self,
        video_path: str | Path,
        working_dir: str | Path,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        self.working_dir = Path(working_dir).resolve()
        self.config = config or PipelineConfig()

        # Sub-directories created lazily
        self.frames_dir: Path = self.working_dir / "frames"
        self.depths_dir: Path = self.working_dir / "depths"
        self.colmap_dir: Path = self.working_dir / "colmap_output"
        self.colmap_db: Path = self.colmap_dir / "database.db"
        self.colmap_sparse: Path = self.colmap_dir / "sparse"
        self.output_mesh: Path = self.working_dir / "final_output.obj"

        self._validate_inputs()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Raise early with clear messages if prerequisites are missing."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        colmap_resolved = shutil.which(self.config.colmap_exe)
        if colmap_resolved is None:
            raise EnvironmentError(
                f"COLMAP executable '{self.config.colmap_exe}' not found in PATH.\n"
                "Install COLMAP (https://colmap.github.io/install.html) and make sure "
                "it is accessible from your shell."
            )
        logger.info("COLMAP found: %s", colmap_resolved)

    def _make_dirs(self) -> None:
        for d in (self.frames_dir, self.depths_dir, self.colmap_sparse):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – Frame Extraction
    # ------------------------------------------------------------------

    def extract_frames(self) -> List[Path]:
        """Extract frames from the video at the configured interval.

        Returns
        -------
        List[Path]
            Sorted list of paths to the saved PNG frames.
        """
        import cv2  # deferred import

        self._make_dirs()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning("Could not read FPS from video; assuming 30.")
            fps = 30.0

        interval_frames = max(1, int(round(fps * self.config.frame_interval_sec)))
        logger.info(
            "Video FPS=%.2f  →  extracting 1 frame every %d frame(s).",
            fps, interval_frames,
        )

        saved: List[Path] = []
        frame_idx = 0
        extracted_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval_frames == 0:
                out_path = self.frames_dir / f"frame_{extracted_idx:05d}.png"
                cv2.imwrite(
                    str(out_path), frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],   # lossless
                )
                saved.append(out_path)
                extracted_idx += 1
            frame_idx += 1

        cap.release()
        logger.info("Extracted %d frames → %s", len(saved), self.frames_dir)

        if len(saved) < 3:
            raise RuntimeError(
                f"Only {len(saved)} frame(s) extracted. "
                "COLMAP needs at least 3 overlapping views."
            )
        return sorted(saved)

    # ------------------------------------------------------------------
    # Step 2 – Depth Map Generation
    # ------------------------------------------------------------------

    def generate_depth_maps(self, frame_paths: List[Path]) -> List[Path]:
        """Run Depth Anything V2 inference on each frame.

        Saves raw (relative) depth values as float32 `.npy` arrays.

        Parameters
        ----------
        frame_paths : List[Path]
            Paths to PNG frames produced by :meth:`extract_frames`.

        Returns
        -------
        List[Path]
            Paths to the saved `.npy` depth arrays (same order as input).
        """
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")

        logger.info(
            "Loading Depth Anything V2 model '%s' on %s …",
            self.config.depth_model_name, device,
        )
        processor = AutoImageProcessor.from_pretrained(self.config.depth_model_name)
        model = AutoModelForDepthEstimation.from_pretrained(
            self.config.depth_model_name
        ).to(device)
        model.eval()

        depth_paths: List[Path] = []

        for i, fp in enumerate(frame_paths):
            image = Image.open(fp).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Interpolate to original resolution
            predicted_depth = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=image.size[::-1],   # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()   # float32 array, shape (H, W)

            npy_path = self.depths_dir / (fp.stem + ".npy")
            np.save(str(npy_path), predicted_depth.astype(np.float32))
            depth_paths.append(npy_path)

            if (i + 1) % 10 == 0 or i == len(frame_paths) - 1:
                logger.info("  Depth maps: %d / %d", i + 1, len(frame_paths))

        logger.info("Depth maps saved → %s", self.depths_dir)
        return depth_paths

    # ------------------------------------------------------------------
    # Step 3 – Camera Pose Estimation via COLMAP
    # ------------------------------------------------------------------

    def run_colmap(self) -> None:
        """Run COLMAP feature extraction, matching, and sparse mapper.

        Outputs the sparse reconstruction to ``colmap_output/sparse/``.
        """
        self.colmap_dir.mkdir(parents=True, exist_ok=True)

        colmap = shutil.which(self.config.colmap_exe)  # already validated

        def _run(args: List[str], step_name: str) -> None:
            cmd = [colmap] + args
            logger.info("COLMAP [%s]: %s", step_name, " ".join(cmd))
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"COLMAP step '{step_name}' failed "
                    f"(exit {result.returncode}):\n{result.stdout[-2000:]}"
                )
            logger.debug("COLMAP [%s] output:\n%s", step_name, result.stdout[-1000:])

        # 1) Feature extraction
        _run(
            [
                "feature_extractor",
                "--database_path", str(self.colmap_db),
                "--image_path",    str(self.frames_dir),
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "1",
            ],
            "feature_extractor",
        )

        # 2) Exhaustive matching  (sequential matching works better for video)
        _run(
            [
                "sequential_matcher",
                "--database_path", str(self.colmap_db),
                "--SiftMatching.use_gpu", "1",
            ],
            "sequential_matcher",
        )

        # 3) Sparse reconstruction (mapper)
        _run(
            [
                "mapper",
                "--database_path",   str(self.colmap_db),
                "--image_path",      str(self.frames_dir),
                "--output_path",     str(self.colmap_sparse),
            ],
            "mapper",
        )

        # COLMAP mapper writes to numbered sub-folders (0, 1, …); keep model 0
        model_dirs = sorted(self.colmap_sparse.glob("[0-9]*"))
        if not model_dirs:
            raise RuntimeError(
                "COLMAP mapper produced no output.  "
                "Possible causes: too few features, too little overlap, "
                "or all-black/textureless video."
            )
        logger.info(
            "COLMAP sparse model written to: %s  (%d model(s) found)",
            model_dirs[0], len(model_dirs),
        )
        # Expose the best (largest) model as self.colmap_model_dir
        self.colmap_model_dir: Path = model_dirs[0]

    # ------------------------------------------------------------------
    # Step 3b – Parse COLMAP text model
    # ------------------------------------------------------------------

    def _export_colmap_to_text(self) -> None:
        """Convert COLMAP binary model to text format for easy parsing."""
        text_dir = self.colmap_model_dir / "text"
        text_dir.mkdir(exist_ok=True)
        colmap = shutil.which(self.config.colmap_exe)
        result = subprocess.run(
            [
                colmap, "model_converter",
                "--input_path",  str(self.colmap_model_dir),
                "--output_path", str(text_dir),
                "--output_type", "TXT",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"COLMAP model_converter failed:\n{result.stdout[-1000:]}"
            )
        self.colmap_text_dir: Path = text_dir

    def _parse_colmap_cameras(self) -> Dict[int, ColmapCamera]:
        """Parse ``cameras.txt`` from a COLMAP text model."""
        cameras: Dict[int, ColmapCamera] = {}
        cam_file = self.colmap_text_dir / "cameras.txt"
        with open(cam_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id  = int(parts[0])
                model   = parts[1]
                width   = int(parts[2])
                height  = int(parts[3])
                params  = [float(p) for p in parts[4:]]
                cameras[cam_id] = ColmapCamera(cam_id, model, width, height, params)
        return cameras

    def _parse_colmap_images(self) -> Dict[str, ColmapImage]:
        """Parse ``images.txt`` from a COLMAP text model.

        Returns a dict keyed by image filename (basename).
        """
        images: Dict[str, ColmapImage] = {}
        img_file = self.colmap_text_dir / "images.txt"
        with open(img_file) as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

        # Every two lines: header + point2D list
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            img_id   = int(parts[0])
            qvec     = [float(x) for x in parts[1:5]]   # qw qx qy qz
            tvec     = [float(x) for x in parts[5:8]]
            cam_id   = int(parts[8])
            name     = parts[9]
            images[name] = ColmapImage(img_id, qvec, tvec, cam_id, name)
        return images

    @staticmethod
    def _qvec_to_rotation_matrix(qvec: List[float]):
        """Convert quaternion [qw, qx, qy, qz] → 3×3 rotation matrix (numpy)."""
        import numpy as np
        qw, qx, qy, qz = qvec
        R = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ], dtype=np.float64)
        return R

    # ------------------------------------------------------------------
    # Step 4 – Scale/Shift Alignment  (critical correctness step)
    # ------------------------------------------------------------------

    @staticmethod
    def _align_depth_to_colmap(
        depth_rel: "np.ndarray",
        colmap_pts3d_uvd: "np.ndarray",
        percentile: float = 95.0,
    ) -> Tuple["np.ndarray", float, float]:
        """Align relative Depth Anything depth to COLMAP metric scale.

        Solves:  depth_metric ≈ scale * depth_relative + shift
        using the least-squares fit between COLMAP 3-D point depths
        (projected into the current camera) and the corresponding
        Depth Anything predictions.

        Parameters
        ----------
        depth_rel : ndarray (H, W)
            Raw relative depth from Depth Anything.
        colmap_pts3d_uvd : ndarray (N, 3)
            Array of (u, v, d_metric) tuples: pixel coordinate and
            COLMAP-metric depth of visible 3-D points in this frame.
        percentile : float
            Points beyond this percentile of residual are rejected (robust fit).

        Returns
        -------
        depth_aligned : ndarray (H, W)
            Scale-and-shift corrected depth map in COLMAP metric units.
        scale : float
        shift : float
        """
        import numpy as np

        if colmap_pts3d_uvd.shape[0] < 4:
            # Not enough correspondences; return unscaled depth
            logger.warning(
                "Too few COLMAP 3-D points for depth alignment (%d).  "
                "Returning unnormalized depth.",
                colmap_pts3d_uvd.shape[0],
            )
            return depth_rel, 1.0, 0.0

        us = colmap_pts3d_uvd[:, 0].astype(int)
        vs = colmap_pts3d_uvd[:, 1].astype(int)
        d_metric = colmap_pts3d_uvd[:, 2]

        H, W = depth_rel.shape
        valid = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        us, vs, d_metric = us[valid], vs[valid], d_metric[valid]
        if us.shape[0] < 4:
            return depth_rel, 1.0, 0.0

        d_pred = depth_rel[vs, us]

        # Robust least squares:  [d_pred | 1] @ [scale, shift]^T = d_metric
        A = np.stack([d_pred, np.ones_like(d_pred)], axis=1)
        result, _, _, _ = np.linalg.lstsq(A, d_metric, rcond=None)
        scale, shift = float(result[0]), float(result[1])

        if scale <= 0:
            # Depth Anything can produce inverted disparity for some checkpoints;
            # flip the sign and re-solve
            A[:, 0] = -d_pred
            result, _, _, _ = np.linalg.lstsq(A, d_metric, rcond=None)
            scale, shift = float(result[0]), float(result[1])
            depth_rel = -depth_rel

        depth_aligned = (scale * depth_rel + shift).clip(min=0.01)
        return depth_aligned, scale, shift

    # ------------------------------------------------------------------
    # Step 5 – TSDF Fusion + Poisson Meshing
    # ------------------------------------------------------------------

    def fuse_and_mesh(
        self,
        frame_paths: List[Path],
        depth_paths: List[Path],
    ) -> Path:
        """Fuse depth maps into a TSDF volume and extract a Poisson mesh.

        Parameters
        ----------
        frame_paths : List[Path]
            RGB frames (same order as depth_paths).
        depth_paths : List[Path]
            Aligned `.npy` depth arrays.

        Returns
        -------
        Path
            Path to the saved ``final_output.obj``.
        """
        import cv2
        import numpy as np
        import open3d as o3d

        # ------------------------------------------------------------------
        # 5a  Parse COLMAP model
        # ------------------------------------------------------------------
        self._export_colmap_to_text()
        cameras  = self._parse_colmap_cameras()
        images   = self._parse_colmap_images()

        logger.info(
            "COLMAP: %d camera(s), %d registered image(s).",
            len(cameras), len(images),
        )

        # ------------------------------------------------------------------
        # 5b  Build TSDF volume
        # ------------------------------------------------------------------
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config.tsdf_voxel_length,
            sdf_trunc=self.config.tsdf_sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        integrated = 0
        for frame_p, depth_p in zip(frame_paths, depth_paths):
            frame_name = frame_p.name
            if frame_name not in images:
                logger.debug("Frame '%s' not registered by COLMAP; skipping.", frame_name)
                continue

            img_meta  = images[frame_name]
            cam       = cameras[img_meta.camera_id]

            # --- Intrinsics -------------------------------------------
            # COLMAP SIMPLE_RADIAL / PINHOLE / RADIAL layouts differ;
            # we handle the common cases.
            if cam.model in ("PINHOLE", "SIMPLE_PINHOLE"):
                if cam.model == "PINHOLE":
                    fx, fy, cx, cy = cam.params[:4]
                else:
                    f, cx, cy = cam.params[:3]
                    fx = fy = f
            elif cam.model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
                fx = cam.params[0]
                fy = cam.params[1] if cam.model == "OPENCV" else cam.params[0]
                cx, cy = cam.params[-2] if cam.model == "OPENCV" else (cam.params[1], cam.params[2])
                if cam.model == "SIMPLE_RADIAL":
                    fx = fy = cam.params[0]
                    cx, cy = cam.params[1], cam.params[2]
            else:
                # Fallback: assume first 4 params are fx fy cx cy
                logger.warning("Unknown COLMAP camera model '%s'; assuming fx fy cx cy.", cam.model)
                fx, fy, cx, cy = cam.params[:4]

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=cam.width, height=cam.height,
                fx=fx, fy=fy, cx=cx, cy=cy,
            )

            # --- Extrinsics (world-to-camera) --------------------------
            R = self._qvec_to_rotation_matrix(img_meta.qvec)
            t = np.array(img_meta.tvec, dtype=np.float64).reshape(3, 1)
            # 4×4 extrinsic matrix  [R | t; 0 0 0 1]
            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = R
            extrinsic[:3,  3] = t.ravel()

            # --- Load & align depth ------------------------------------
            depth_rel = np.load(str(depth_p)).astype(np.float64)

            # Build COLMAP 3-D point correspondences for alignment
            # (we project registered 3D points visible in this image)
            # For simplicity, we use the COLMAP depth directly via re-projection
            # of the sparse point cloud; here we build them from points3D.txt
            pts3d_uvd = self._get_colmap_pts_for_frame(
                img_meta, R, t, fx, fy, cx, cy, cam.width, cam.height
            )
            depth_aligned, scale, shift = self._align_depth_to_colmap(
                depth_rel, pts3d_uvd, self.config.depth_align_percentile
            )
            logger.debug(
                "Frame %-20s  scale=%.4f  shift=%.4f  pts=%d",
                frame_name, scale, shift, pts3d_uvd.shape[0],
            )

            # --- Load RGB image ----------------------------------------
            bgr = cv2.imread(str(frame_p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Resize depth to match frame dimensions if needed
            h, w = rgb.shape[:2]
            if depth_aligned.shape != (h, w):
                depth_aligned = cv2.resize(
                    depth_aligned, (w, h), interpolation=cv2.INTER_LINEAR
                )

            # --- Create Open3D RGBD image ------------------------------
            depth_o3d = o3d.geometry.Image(depth_aligned.astype(np.float32))
            color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,           # depth already in metres
                depth_trunc=10.0,
                convert_rgb_to_intensity=False,
            )

            vol.integrate(rgbd, intrinsic, extrinsic)
            integrated += 1

        logger.info("Integrated %d frames into TSDF volume.", integrated)
        if integrated == 0:
            raise RuntimeError(
                "No frames were integrated.  "
                "All frames may have been rejected by COLMAP."
            )

        # ------------------------------------------------------------------
        # 5c  Extract Poisson mesh
        # ------------------------------------------------------------------
        logger.info("Extracting mesh from TSDF …")
        mesh = vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        logger.info(
            "Raw mesh: %d vertices, %d triangles.",
            len(mesh.vertices), len(mesh.triangles),
        )

        # Clean up isolated geometry
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Optional: keep only the largest connected component
        triangle_clusters, cluster_n_triangles, _ = (
            mesh.cluster_connected_triangles()
        )
        import numpy as np
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        largest_cluster = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

        logger.info(
            "Cleaned mesh: %d vertices, %d triangles.",
            len(mesh.vertices), len(mesh.triangles),
        )

        # ------------------------------------------------------------------
        # 5d  Export
        # ------------------------------------------------------------------
        o3d.io.write_triangle_mesh(
            str(self.output_mesh), mesh,
            write_ascii=True, write_vertex_normals=True,
        )
        logger.info("Mesh saved → %s", self.output_mesh)
        return self.output_mesh

    # ------------------------------------------------------------------
    # Helper – sparse 3-D points visible in a given frame
    # ------------------------------------------------------------------

    def _get_colmap_pts_for_frame(
        self,
        img_meta: ColmapImage,
        R: "np.ndarray",
        t: "np.ndarray",
        fx: float, fy: float, cx: float, cy: float,
        W: int, H: int,
    ) -> "np.ndarray":
        """Project COLMAP sparse 3-D points into image coords + metric depth.

        Returns ndarray of shape (N, 3) with columns [u, v, depth_metric].
        Returns empty array if points3D.txt is missing or has no content.
        """
        import numpy as np

        pts3d_file = self.colmap_text_dir / "points3D.txt"
        if not pts3d_file.exists():
            return np.empty((0, 3))

        pts_uvd: List[Tuple[float, float, float]] = []

        with open(pts3d_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # Check if this point is visible in the current image
                # Track part starts at index 8: [IMAGE_ID, POINT2D_IDX, …]
                track_data = parts[8:]
                image_ids_in_track = [int(track_data[i]) for i in range(0, len(track_data), 2)]
                if img_meta.image_id not in image_ids_in_track:
                    continue

                X = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                # Transform to camera coordinates
                X_cam = R @ X + t.ravel()
                depth = X_cam[2]
                if depth <= 0:
                    continue
                u = fx * X_cam[0] / depth + cx
                v = fy * X_cam[1] / depth + cy
                pts_uvd.append((u, v, depth))

        if not pts_uvd:
            return np.empty((0, 3))
        return np.array(pts_uvd, dtype=np.float64)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute the full pipeline end-to-end.

        Returns
        -------
        Path
            Path to the final ``final_output.obj`` mesh file.
        """
        logger.info("=" * 60)
        logger.info("VideoTo3DPipeline  START")
        logger.info("  Video      : %s", self.video_path)
        logger.info("  Working dir: %s", self.working_dir)
        logger.info("=" * 60)

        # Step 1
        logger.info("[1/4] Extracting frames …")
        frame_paths = self.extract_frames()

        # Step 2
        logger.info("[2/4] Generating depth maps …")
        depth_paths = self.generate_depth_maps(frame_paths)

        # Step 3
        logger.info("[3/4] Running COLMAP SfM …")
        self.run_colmap()

        # Step 4 + 5
        logger.info("[4/4] Fusing depth maps and meshing …")
        mesh_path = self.fuse_and_mesh(frame_paths, depth_paths)

        logger.info("=" * 60)
        logger.info("Pipeline COMPLETE.  Output → %s", mesh_path)
        logger.info("=" * 60)
        return mesh_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_to_3d_pipeline",
        description=(
            "Convert a video file to a 3D mesh (.obj) using "
            "Depth Anything V2, COLMAP, and Open3D.  "
            "This script is completely independent of any Blender pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True, metavar="PATH",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--working_dir", required=True, metavar="DIR",
        help="Root directory for all intermediate and final output files.",
    )
    parser.add_argument(
        "--frame_interval", type=float, default=1.0, metavar="SEC",
        help="Extract one frame every N seconds.",
    )
    parser.add_argument(
        "--depth_model",
        default="depth-anything/Depth-Anything-V2-Large-hf",
        metavar="HF_MODEL_ID",
        help="Hugging Face model ID for Depth Anything V2.",
    )
    parser.add_argument(
        "--colmap_exe", default="colmap", metavar="EXE",
        help="Name or full path of the COLMAP executable.",
    )
    parser.add_argument(
        "--tsdf_voxel", type=float, default=0.02, metavar="METRES",
        help="TSDF voxel length in metres.",
    )
    parser.add_argument(
        "--tsdf_trunc", type=float, default=0.04, metavar="METRES",
        help="TSDF truncation distance in metres.",
    )
    parser.add_argument(
        "--poisson_depth", type=int, default=9, metavar="N",
        help="Octree depth for Poisson surface reconstruction.",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Torch device for Depth Anything V2 inference.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        frame_interval_sec   = args.frame_interval,
        depth_model_name     = args.depth_model,
        colmap_exe           = args.colmap_exe,
        tsdf_voxel_length    = args.tsdf_voxel,
        tsdf_sdf_trunc       = args.tsdf_trunc,
        poisson_depth        = args.poisson_depth,
        device               = args.device,
    )

    pipeline = VideoTo3DPipeline(
        video_path=args.video,
        working_dir=args.working_dir,
        config=config,
    )

    try:
        output = pipeline.run()
        print(f"\nSuccess!  3D mesh saved to: {output}")
        sys.exit(0)
    except (FileNotFoundError, EnvironmentError, IOError, RuntimeError) as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
