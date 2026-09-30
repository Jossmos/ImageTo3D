# ImageTo3D Pipeline

Automated pipeline for converting 2D images into 3D relief STL models ready for 3D printing.

---

## Workflow Overview (3 Steps)

```
[ Input Image ] ──► [ Step 1: rembg ] ──► [ Step 2: Depth-Anything-V2 ] ──► [ Step 3: Blender ] ──► [ 3D STL Model ]
                     (Background removal)       (Monocular depth map)           (Heightfield mesh & solid base)
```

### Step 1: Background Removal
- **Engine**: Local open-source `rembg` (no API key required) or optional Remove.bg API fallback
- **Input**: JPG/PNG image
- **Output**: RGBA PNG with transparent background (`1_removebg_output.png`)

### Step 2: Depth Map Estimation
- **Engine**: Local Depth-Anything-V2 model (runs on GPU with PyTorch)
- **Input**: Alpha-cutout PNG from Step 1
- **Output**: 16-bit / grayscale depth map (`2_depth_map_output.png`)

### Step 3: 3D Relief & Solid Mesh Generation
- **Engine**: Headless Blender (Python script: `blender_displace.py` / `blender_apply_alpha_solidify.py`)
- **Method**: Heightfield mesh generation + pixel-accurate alpha masking + solid manifold base extrusion
- **Input**: Depth map + alpha cutout
- **Output**: Watertight, 3D-printable manifold STL (`3_3d_model_output.stl`)

---

## Installation & Setup

### Prerequisites
- **Python**: 3.10 or newer (tested on Python 3.11)
- **Blender**: 5.0+ installed (runs in background headless mode)
- **CUDA-compatible GPU**: Recommended for fast local depth estimation (CPU fallback supported)

### 1. Clone the Repository
```bash
git clone https://github.com/Jossmos/ImageTo3D.git
cd ImageTo3D
```

### 2. Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Clone Depth-Anything-V2 & Download Weights
Clone the official Depth-Anything-V2 repository into the workspace root:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

Create a `checkpoints/` folder and download the pre-trained weights (e.g., ViT-Large):
```bash
mkdir checkpoints
```
Download [`depth_anything_v2_vitl.pth`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it into `checkpoints/depth_anything_v2_vitl.pth`.

### 5. Verify Blender Path
Ensure the path to Blender in `image_pipeline.py` matches your local installation:
```python
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
```

---

## Usage

### 1. Full Pipeline Execution
To process an image through all 3 stages:
```bash
python image_pipeline.py
```
By default, the script reads `INPUT_IMAGE_PATH` and generates output files in the same directory or specified `OUTPUT_FOLDER_PATH`.

You can also pass environment variables:
```powershell
$env:INPUT_IMAGE_PATH="path/to/photo.jpg"
python image_pipeline.py
```

### 2. Drag & Drop (Windows Batch)
Simply drag and drop an image file onto `run_pipeline.bat` in Windows File Explorer:
```bash
run_pipeline.bat "C:\path\to\your_image.jpg"
```

### 3. Blender-Only Mode (`SKIP_API`)
If you already have depth and alpha maps and only want to tune or regenerate the 3D mesh:
```powershell
$env:SKIP_API="1"
$env:DEPTH_PATH="path/to/2_depth_map_output.png"
$env:ALPHA_PATH="path/to/1_removebg_output.png"
python -c "import image_pipeline as p; p.INPUT_IMAGE_PATH='dummy'; p.main()"
```
Or with the batch script:
```bash
run_pipeline.bat "dummy.png" --skip "path/to/depth.png" "path/to/alpha.png"
```

### 4. Convert `.blend` directly to Solid STL
Drag and drop any saved `.blend` file onto `DragDrop_Blend_to_Solid.bat` or run:
```bash
python pipeline_blend_to_solid.py "path/to/model.blend"
```

---

## Outputs

| Step | Output File | Description |
|:---:|:---|:---|
| 1 | `1_removebg_output.png` | RGBA image with background removed |
| 2 | `2_depth_map_output.png` | Grayscale depth estimation map (whiter = higher elevation) |
| 3 | `3_3d_model_output.stl` | Watertight, solid 3D manifold model ready for slicing / printing |

---

## Mesh Validation & Repair

To inspect mesh topology and verify manifoldness:
```python
import trimesh

mesh = trimesh.load('3_3d_model_output.stl', force='mesh')
print("Watertight:", mesh.is_watertight)
print("Winding consistent:", mesh.is_winding_consistent)

# Optional automated repair
from trimesh import repair
repair.fill_holes(mesh)
mesh.export('3_3d_model_output_repaired.stl')
```

---

## Video-to-3D Pipeline

An experimental reconstruction pipeline is available in `video_to_3d_pipeline.py`.

### Features
- Frame extraction from video to `working_dir/frames`
- Monocular depth map generation stored as `.npy`
- Structure-from-Motion (COLMAP) integration
- Open3D TSDF depth fusion
- Direct export to `.obj` or `.stl`

### Usage
```bash
pip install -r requirements_video3d.txt
python video_to_3d_pipeline.py --video-path path/to/video.mp4 --working-dir path/to/workdir --output-mesh final_output.obj
```

---

## Troubleshooting

- **Blender executable not found**:
  Update `BLENDER_PATH` in `image_pipeline.py` or specify via your environment variables.
- **Out of GPU Memory (OOM)**:
  Reduce sample resolution or switch to the Base/Small encoder variant (`vits` or `vitb`) in `depth_estimator.py`.
- **Mesh non-manifold**:
  Use `validate_stl.py` or run `trimesh.repair.fill_holes()`.

---

## License & Credits

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) by TikTok / HKU
- [rembg](https://github.com/danielgatis/rembg)
- [Blender](https://www.blender.org/)
- [Trimesh](https://trimesh.org/)
