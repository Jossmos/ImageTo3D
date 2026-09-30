"""Geoptimaliseerde image-processing pipeline met rembg model-caching.

Dit is een variant van image_pipeline.py die het rembg-model EENMAAL in het geheugen laadt
en hergebruikt voor meerdere afbeeldingen. Dit is veel sneller voor batch-verwerking.

Gebruik:
  python image_pipeline_optimized.py input_folder output_folder

waar input_folder alle te verwerken .png/.jpg bestanden bevat.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# rembg import
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    remove = None
    new_session = None

try:
    from depth_estimator import DepthEstimator
    DEPTH_ESTIMATOR_AVAILABLE = True
except Exception:
    DepthEstimator = None  # type: ignore
    DEPTH_ESTIMATOR_AVAILABLE = False

# ==============================
# Globale configuratie
# ==============================
BLENDER_PATH: str = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
# Globale rembg session (geladen eenmaal)
REMBG_SESSION = None


def initialize_rembg_session():
    """Laad het rembg-model eenmaal in het geheugen."""
    global REMBG_SESSION
    if REMBG_SESSION is None and REMBG_AVAILABLE:
        try:
            print("[Init] rembg-model wordt geladen (dit kan de eerste keer even duren)...")
            REMBG_SESSION = new_session()
            print("[Init] rembg-model succesvol geladen in geheugen.")
        except Exception as exc:
            print(f"[Init][Warning] Kon rembg-session niet initialiseren: {exc}. Valt terug op standaard.")
            REMBG_SESSION = None


def _generate_local_depth_map(input_image_path: Path, output_path: Path) -> Optional[str]:
    """Genereer lokaal een eenvoudige depth map zonder cloud-API."""
    print("[Stap 2] Start: local depth map genereren (geen API / geen abonnement nodig)...")

    try:
        with Image.open(input_image_path) as image:
            rgba_image = image.convert("RGBA")

        rgba_array = np.asarray(rgba_image, dtype=np.float32) / 255.0
        rgb = rgba_array[:, :, :3]
        alpha = rgba_array[:, :, 3]

        # Luminance als basis, alpha als mask zodat transparante pixels donker blijven.
        luminance = (
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        )

        depth = np.clip((luminance * alpha) + (alpha * 0.15), 0.0, 1.0)
        depth_image = Image.fromarray((depth * 255.0).astype(np.uint8), mode="L")

        # Kleine blur om pixelruis en harde randen wat te verzachten.
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=1.2))
        depth_image = ImageOps.autocontrast(depth_image)

        depth_image.save(output_path)
        print(f"[Stap 2][Success] Lokale depth map opgeslagen: {output_path}")
        return str(output_path)

    except Exception as exc:
        print(f"[Stap 2][Error] Lokale depth generatie mislukt: {exc}")
        return None


def remove_background_optimized(
    input_image_path: str,
    output_path: str,
    session=None,
) -> Optional[str]:
    """Verwijder achtergrond met optioneel rembg-session (voor model-caching).

    Args:
        input_image_path: Pad naar origineel afbeelding.
        output_path: Pad waar output PNG moet worden opgeslagen.
        session: Optionele rembg session (indien None, maakt standaard)
    Returns:
        Pad naar output bij succes, anders None.
    """
    if not REMBG_AVAILABLE:
        print("[Stap 1][Error] rembg niet geïnstalleerd. Voer 'pip install rembg' uit.")
        return None

    input_p = Path(input_image_path)
    output_p = Path(output_path)

    if not input_p.is_file():
        print(f"[Stap 1][Error] Inputbestand niet gevonden: {input_p}")
        return None

    try:
        # Lees originele afbeelding
        with Image.open(input_p) as img:
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                img_rgb = Image.new("RGB", img.size, (255, 255, 255))
                img_rgb.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            else:
                img_rgb = img.convert("RGB")

        # Pas rembg toe (met of zonder gecachte session)
        if session is not None:
            img_no_bg = remove(img_rgb, session=session)
        else:
            img_no_bg = remove(img_rgb)

        # Zorg voor RGBA output
        if img_no_bg.mode != "RGBA":
            img_no_bg = img_no_bg.convert("RGBA")

        # Sla op
        img_no_bg.save(output_p, "PNG")
        print(f"[Stap 1][Success] {input_p.name} → {output_p.name}")
        return str(output_p)

    except Exception as exc:
        print(f"[Stap 1][Error] {input_p.name} mislukt: {exc}")
        return None


def generate_depth_map(
    no_bg_image_path: str,
    output_folder_path: str,
) -> Optional[str]:
    """Genereer depth map lokaal via Depth-Anything-V2."""
    print("[Stap 2] Start: depth map genereren...")

    input_path = Path(no_bg_image_path)
    output_path = Path(output_folder_path) / "2_depth_map_output.png"

    if not input_path.is_file():
        print(f"[Stap 2][Error] Inputbestand niet gevonden: {input_path}")
        return None

    try:
        if DEPTH_ESTIMATOR_AVAILABLE:
            try:
                encoder = os.environ.get("DEPTH_ENCODER", "vitl").strip().lower()
                checkpoint = os.environ.get(
                    "DEPTH_CHECKPOINT_PATH",
                    str(Path("checkpoints") / "depth_anything_v2_vitl.pth"),
                )
                estimator = DepthEstimator(encoder=encoder, checkpoint=checkpoint)
                depth_arr = estimator.estimate(str(input_path))
                estimator.save_depth_png16(output_path, depth_arr)
                print(f"[Stap 2][Success] Depth map opgeslagen: {output_path}")
                return str(output_path)
            except Exception as exc:
                print(f"[Stap 2][Warning] Depth-Anything-V2 faalde: {exc}. Valt terug op lokaal.")

        return _generate_local_depth_map(input_path, output_path)

    except Exception as exc:
        print(f"[Stap 2][Error] Fout: {exc}")
        return None


def _validate_depth_and_alpha(depth_path: Path, alpha_path: Path) -> tuple[bool, bool, str]:
    """Voer validatie uit op depth- en alpha-bestanden."""
    msgs = []
    depth_ok = False
    alpha_ok = False

    try:
        d = Image.open(depth_path).convert("L")
        da = np.asarray(d, dtype=np.uint8)
        std = float(da.std())
        mn = int(da.min())
        mx = int(da.max())
        msgs.append(f"depth: min={mn}, max={mx}, std={std:.2f}")
        if std >= 4.0 and (mx - mn) >= 10:
            depth_ok = True
        else:
            msgs.append("depth lijkt weinig variatie te hebben")
    except Exception as e:
        msgs.append(f"depth validatie faalde: {e}")

    try:
        a = Image.open(alpha_path).convert("RGBA")
        aa = np.asarray(a)[:, :, 3].astype(np.uint8)
        nonzero = int((aa > 16).sum())
        total = int(aa.size)
        ratio = nonzero / total if total > 0 else 0.0
        msgs.append(f"alpha: nontransparent={nonzero}/{total} ({ratio*100:.2f}%)")
        if ratio >= 0.005:
            alpha_ok = True
        else:
            msgs.append("alpha lijkt bijna volledig transparant")
    except Exception as e:
        msgs.append(f"alpha validatie faalde: {e}")

    return depth_ok, alpha_ok, " | ".join(msgs)


def generate_3d_model(
    depth_map_pad: str,
    alpha_map_pad: str,
    output_stl_pad: str,
    extra_opts: list[str] | None = None,
) -> bool:
    """Genereer 3D-model via Blender."""
    print("[Stap 3] Start: 3D-model genereren...")

    if not Path(depth_map_pad).is_file():
        print(f"[Stap 3][Error] Depth map niet gevonden: {depth_map_pad}")
        return False

    if not Path(alpha_map_pad).is_file():
        print(f"[Stap 3][Error] Alpha map niet gevonden: {alpha_map_pad}")
        return False

    try:
        command = [
            BLENDER_PATH,
            "--background",
            "--python",
            "blender_displace.py",
            "--",
            depth_map_pad,
            alpha_map_pad,
            output_stl_pad,
        ]

        if extra_opts:
            command.extend([str(x) for x in extra_opts])

        print(f"[Stap 3] Blender: {' '.join(command)}")

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(f"[Stap 3][Blender] {result.stdout}")

        print(f"[Stap 3][Success] STL gegenereerd: {output_stl_pad}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[Stap 3][Error] Blender faalde (exit code {e.returncode})")
        if e.stderr:
            print(f"[Stap 3][Blender Error] {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"[Stap 3][Error] Blender niet gevonden: {BLENDER_PATH}")
        return False
    except Exception as e:
        print(f"[Stap 3][Error] Fout: {e}")
        return False


def process_image(
    input_path: str,
    output_folder: str,
    rembg_session=None,
) -> bool:
    """Verwerk één afbeelding volledig (stap 1-3)."""
    input_p = Path(input_path)
    output_dir = Path(output_folder)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Verwerking: {input_p.name}")
    print(f"{'='*60}")

    # Stap 1: Remove background
    alpha_path = remove_background_optimized(
        input_image_path=str(input_p),
        output_path=str(output_dir / "1_removebg_output.png"),
        session=rembg_session,
    )

    if not alpha_path:
        print(f"❌ Stap 1 mislukt voor {input_p.name}")
        return False

    # Stap 2: Generate depth
    depth_path = generate_depth_map(
        no_bg_image_path=alpha_path,
        output_folder_path=str(output_dir),
    )

    if not depth_path:
        print(f"❌ Stap 2 mislukt voor {input_p.name}")
        return False

    # Stap 3: Generate 3D model
    stl_path = str(output_dir / "3_3d_model_output.stl")
    success = generate_3d_model(
        depth_map_pad=depth_path,
        alpha_map_pad=alpha_path,
        output_stl_pad=stl_path,
        extra_opts=["apply_modifiers_for_stl=false"],
    )

    if success:
        print(f"✅ {input_p.name} volledig verwerkt → {stl_path}")
    else:
        print(f"❌ Stap 3 mislukt voor {input_p.name}")

    return success


def batch_process(input_folder: str, output_base_folder: str) -> None:
    """Verwerk alle afbeeldingen in een folder (met model-caching)."""
    input_dir = Path(input_folder)
    output_dir = Path(output_base_folder)

    if not input_dir.is_dir():
        print(f"[Error] Input folder niet gevonden: {input_dir}")
        return

    # Vind alle beeldbestanden
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"[Error] Geen afbeeldingen gevonden in {input_dir}")
        return

    print(f"[Batch] Gevonden: {len(image_files)} afbeeldingen")
    print(f"[Batch] Output folder: {output_dir}")

    # Initialiseer rembg-session eenmaal voor alle afbeeldingen
    initialize_rembg_session()

    success_count = 0
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Bezig...")
        img_output_folder = output_dir / img_file.stem
        success = process_image(
            input_path=str(img_file),
            output_folder=str(img_output_folder),
            rembg_session=REMBG_SESSION,
        )
        if success:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"[Batch] Klaar! {success_count}/{len(image_files)} succesvol verwerkt.")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        batch_process(input_folder, output_folder)
    else:
        print("Gebruik:")
        print("  python image_pipeline_optimized.py <input_folder> <output_folder>")
        print("\nVoorbeeld:")
        print("  python image_pipeline_optimized.py ./Images ./output")
        sys.exit(1)
