# VideoTo3DPipeline — README

End-to-end reconstruction of a 3D mesh (`.obj`) from a video file.
Completely isolated from any existing Blender / single-image pipeline.

---

## Architecture

```
video_to_3d_pipeline.py
│
├── PipelineConfig          dataclass – all tunable knobs
├── ColmapCamera / Image    lightweight parse models
└── VideoTo3DPipeline       the main class
    │
    ├── extract_frames()            Step 1  – OpenCV
    ├── generate_depth_maps()       Step 2  – Depth Anything V2
    ├── run_colmap()                Step 3  – COLMAP via subprocess
    ├── _align_depth_to_colmap()    Step 3b – scale+shift alignment  ★
    ├── fuse_and_mesh()             Step 4  – Open3D TSDF + Poisson
    └── run()                       orchestrator
```

★ The scale/shift alignment step is the critical correctness fix
that most tutorials omit.  Depth Anything V2 returns **relative**
(unitless) depth; COLMAP operates in **metric** space.  Without
alignment the fused geometry is distorted.

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements_video3d.txt

# 2. Install COLMAP (not a pip package)
#    Ubuntu/Debian:
sudo apt install colmap
#    macOS:
brew install colmap
#    Windows: download from https://github.com/colmap/colmap/releases
```

---

## Usage

### Command line

```bash
python video_to_3d_pipeline.py \
    --video     /path/to/my_scene.mp4 \
    --working_dir ./reconstruction_out \
    --frame_interval 0.5 \
    --device cuda
```

### Python API

```python
from video_to_3d_pipeline import VideoTo3DPipeline, PipelineConfig

config = PipelineConfig(
    frame_interval_sec=0.5,
    tsdf_voxel_length=0.015,   # finer voxels → more detail, more RAM
    device="cuda",
)

pipeline = VideoTo3DPipeline(
    video_path="my_scene.mp4",
    working_dir="./out",
    config=config,
)
mesh_path = pipeline.run()   # returns Path to final_output.obj
```

---

## Output directory layout

```
working_dir/
├── frames/
│   ├── frame_00000.png
│   └── ...
├── depths/
│   ├── frame_00000.npy
│   └── ...
├── colmap_output/
│   ├── database.db
│   └── sparse/
│       └── 0/
│           ├── cameras.bin / cameras.txt
│           ├── images.bin  / images.txt
│           └── points3D.bin / points3D.txt
└── final_output.obj        ← ★ your mesh
```

---

## CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Path to input video |
| `--working_dir` | required | Output root directory |
| `--frame_interval` | `1.0` | Seconds between extracted frames |
| `--depth_model` | `…V2-Large-hf` | HuggingFace Depth Anything V2 model ID |
| `--colmap_exe` | `colmap` | Path/name of COLMAP binary |
| `--tsdf_voxel` | `0.02` | TSDF voxel size in metres |
| `--tsdf_trunc` | `0.04` | TSDF truncation distance in metres |
| `--poisson_depth` | `9` | Octree depth for Poisson reconstruction |
| `--device` | `cuda` | Torch device (`cuda` or `cpu`) |
| `--log_level` | `INFO` | Logging verbosity |

---

## Tips for best results

| Scenario | Recommendation |
|---|---|
| Large/long video | Use `--depth_model …V2-Small-hf` to save GPU RAM |
| Fine surface detail | Lower `--tsdf_voxel` to `0.01` (needs more RAM) |
| Noisy/sparse result | Increase `--frame_interval` to reduce redundant views |
| COLMAP fails | Ensure scene has sufficient texture; avoid pure white walls |
| CPU-only machine | Pass `--device cpu` and edit `use_gpu 0` in run_colmap() |

---

## Known limitations

- Monocular depth (no stereo/LiDAR) → scale alignment is approximate
- COLMAP fails on textureless / repetitive surfaces
- Very large scenes may exceed GPU VRAM for TSDF integration

For production-quality reconstruction consider:
- [MonST3R](https://monst3r.github.io/) – video-native 3D reconstruction
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) – photorealistic NeRF alternative
