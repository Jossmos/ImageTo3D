"""Geautomatiseerde image-processing pipeline: rembg (lokaal) + lokale depth + Blender 3D.

Dit script verwerkt een lokale afbeelding in drie stappen:
1) rembg (lokaal, open-source): achtergrond verwijderen → PNG met alpha-kanaal
2) Depth map genereren → lokaal via de officiële Depth-Anything-V2 code
3) Blender: 3D-reliëf (STL) genereren met pixel-exacte alpha-uitsnijding
   - Bouwt heightfield mesh direct uit depth-map (robuust, geen modifiers)
   - Appliceert alpha-masking om pixels < 0.5 alpha weg te snijden
   - Genereert solid blok met bottom/side faces voor 3D-printen

SKIP_API-mode: Sla stappen 1-2 over, draai alleen Blender (handig voor iteratie)

OPMERKING: rembg is een lokale, open-source library die geen API-key nodig heeft.
Het model wordt eenmaal in het geheugen geladen en hergebruikt voor meerdere afbeeldingen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# rembg import
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    remove = None

try:
    from depth_estimator import DepthEstimator
    DEPTH_ESTIMATOR_AVAILABLE = True
except Exception:
    DepthEstimator = None  # type: ignore
    DEPTH_ESTIMATOR_AVAILABLE = False

# ==============================
# Globale configuratie
# ==============================
INPUT_IMAGE_PATH: str = os.environ.get("INPUT_IMAGE_PATH", "input/example.png")
# Laat leeg om op te slaan naast de inputafbeelding.
OUTPUT_FOLDER_PATH: str = ""
# Als OUTPUT_FOLDER_PATH leeg is: kies of resultaten in een submap "output" moeten komen.
USE_OUTPUT_SUBFOLDER: bool = False

# Blender executable path. Pas dit aan naar je absolute Blender-pad als nodig.
BLENDER_PATH: str = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"


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
        
        # Pas gamma correctie toe om spitse delen af te ronden
        gamma = float(os.environ.get("DEPTH_GAMMA", "0.6")) # 0.6 is een goede standaard voor rondere gezichten/snuiten
        if gamma != 1.0:
            depth = np.power(depth, gamma)
            
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


def remove_background(
    input_image_path: str,
    output_folder_path: str,
) -> Optional[str]:
    """Verwijder de achtergrond van een afbeelding met rembg (lokaal, open-source).

    Args:
        input_image_path: Pad naar de originele invoerafbeelding.
        output_folder_path: Map waarin de output moet worden opgeslagen.
    Returns:
        Het pad naar de achtergrondloze afbeelding bij succes, anders None.
    """
    print("[Stap 1] Start: achtergrond verwijderen via rembg (lokaal)...")

    if not REMBG_AVAILABLE:
        print("[Stap 1][Error] rembg is niet geïnstalleerd. Voer 'pip install rembg' uit.")
        return None

    input_path = Path(input_image_path)
    output_path = Path(output_folder_path) / "1_removebg_output.png"

    if not input_path.is_file():
        print(f"[Stap 1][Error] Invoerbestand niet gevonden: {input_path}")
        return None

    try:
        # Open origineel afbeelding
        with Image.open(input_path) as img:
            # Zorg ervoor dat we RGB hebben als input
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                img_rgb = Image.new("RGB", img.size, (255, 255, 255))
                img_rgb.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            else:
                img_rgb = img.convert("RGB")

        # Pas rembg toe (lokaal model)
        # Beperk het aantal draden om memory-spikes in ONNX runtime te verminderen
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        from rembg import new_session
        
        # 'bria-rmbg' is geselecteerd als het beste model (vergelijkbaar met remove.bg).
        primary_model = os.environ.get("REMBG_MODEL", "bria-rmbg")
        max_dim_env = int(os.environ.get("REMBG_MAX_SIZE", "1280"))
        
        # Fallback strategie voor memory problemen:
        attempts = [
            (primary_model, max_dim_env),
            (primary_model, 1024),
            (primary_model, 800),
        ]
        if primary_model != "u2net":
            attempts.append(("u2net", max_dim_env))
            
        img_no_bg = None
        last_exc = None
        
        for current_model, current_max_dim in attempts:
            try:
                print(f"[Stap 1] Poging met model '{current_model}' (max resolutie {current_max_dim}px)...")
                
                # Kopieer en verklein afbeelding voor deze poging
                current_img = img_rgb.copy()
                if max(current_img.size) > current_max_dim:
                    current_img.thumbnail((current_max_dim, current_max_dim), Image.Resampling.LANCZOS)
                
                # Zorg voor cpu execution provider om memory errors met half-installed CUDA te voorkomen
                session = new_session(current_model, providers=['CPUExecutionProvider'])
                
                img_no_bg = remove(
                    current_img,
                    session=session,
                    post_process_mask=False, # Uitgeschakeld omdat de originele randen van bria-rmbg perfect waren
                )
                print(f"[Stap 1] Succes met model '{current_model}' op max resolutie {current_max_dim}px!")
                break
            except Exception as exc:
                print(f"[Stap 1] Waarschuwing: Poging mislukt: {exc}")
                last_exc = exc
                
        if img_no_bg is None:
            raise RuntimeError(f"Alle rembg pogingen mislukt. Laatste fout: {last_exc}")

        # Zorg ervoor dat output RGBA is (voor alpha-kanaal)
        if img_no_bg.mode != "RGBA":
            img_no_bg = img_no_bg.convert("RGBA")

        # Sla op
        img_no_bg.save(output_path, "PNG")
        print(f"[Stap 1][Success] Achtergrond verwijderd: {output_path}")
        return str(output_path)

    except FileNotFoundError as exc:
        print(f"[Stap 1][Error] Bestand niet gevonden: {exc}")
        return None
    except OSError as exc:
        print(f"[Stap 1][Error] Bestandsfout: {exc}")
        return None
    except Exception as exc:
        print(f"[Stap 1][Error] rembg verwerking mislukt: {exc}")
        return None


def generate_depth_map(
    no_bg_image_path: str,
    output_folder_path: str,
) -> Optional[str]:
    """Genereer een depth map lokaal via Depth-Anything-V2.

    Args:
        no_bg_image_path: Pad naar de afbeelding zonder achtergrond.
        output_folder_path: Map waarin de depth map moet worden opgeslagen.
    Returns:
        Het pad naar de opgeslagen depth map bij succes, anders None.
    """
    print("[Stap 2] Start: depth map genereren via officiële lokale Depth-Anything-V2...")

    input_path = Path(no_bg_image_path)
    output_path = Path(output_folder_path) / "2_depth_map_output.png"

    if not input_path.is_file():
        print(f"[Stap 2][Error] Inputbestand zonder achtergrond niet gevonden: {input_path}")
        return None

    try:
        # Use the official local repo + downloaded checkpoint.
        if DEPTH_ESTIMATOR_AVAILABLE:
            try:
                encoder = os.environ.get("DEPTH_ENCODER", "vitl").strip().lower()
                checkpoint = os.environ.get(
                    "DEPTH_CHECKPOINT_PATH",
                    str(Path("checkpoints") / "depth_anything_v2_vitl.pth"),
                )
                estimator = DepthEstimator(encoder=encoder, checkpoint=checkpoint)
                depth_arr = estimator.estimate(str(input_path))
                
                # Pas gamma correctie toe om te voorkomen dat voorste objecten (snuit) enorm spits worden
                # terwijl de achterliggende delen (hoofd) te vlak worden.
                # Gamma < 1 lift de middentonen op waardoor het model ronder wordt en minder extreme pieken heeft.
                gamma = float(os.environ.get("DEPTH_GAMMA", "0.6")) # 0.6 is default, pas aan indien nodig (bv 0.4 tot 1.0)
                if gamma != 1.0:
                    depth_arr = np.power(depth_arr, gamma)
                    
                estimator.save_depth_png16(output_path, depth_arr)
                print(f"[Stap 2][Success] Depth map opgeslagen: {output_path}")
                return str(output_path)
            except Exception as exc:
                print(f"[Stap 2][Warning] Depth-Anything-V2 faalde: {exc}. Falling back to simple local method.")

        return _generate_local_depth_map(input_path, output_path)

    except Exception as exc:
        print(f"[Stap 2][Error] Fout tijdens depth map generatie: {exc}")
        return None


def _validate_depth_and_alpha(depth_path: Path, alpha_path: Path) -> tuple[bool, bool, str]:
    """Voer eenvoudige validatie uit op depth- en alpha-bestanden.

    Returns: (depth_ok, alpha_ok, message)
    - depth_ok: True als depth-beeld voldoende variatie bevat
    - alpha_ok: True als alpha-beeld echte alpha-informatie bevat
    """
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
        # Heuristiek: depth moet enige variatie hebben
        if std >= 4.0 and (mx - mn) >= 10:
            depth_ok = True
        else:
            msgs.append("depth lijkt weinig variatie te bevatten (mogelijk geen echte depth-map)")
    except Exception as e:
        msgs.append(f"depth validatie faalde: {e}")

    try:
        a = Image.open(alpha_path).convert("RGBA")
        aa = np.asarray(a)[:, :, 3].astype(np.uint8)
        nonzero = int((aa > 16).sum())
        total = int(aa.size)
        ratio = nonzero / total if total > 0 else 0.0
        msgs.append(f"alpha: nontransparent_pixels={nonzero}/{total} ({ratio*100:.2f}%)")
        # Heuristiek: er moet minstens 0.5% niet-transparante pixels
        if ratio >= 0.005:
            alpha_ok = True
        else:
            msgs.append("alpha lijkt bijna volledig transparant (mogelijk niet geldig)")
    except Exception as e:
        msgs.append(f"alpha validatie faalde: {e}")

    return depth_ok, alpha_ok, " | ".join(msgs)


def generate_3d_model(
    depth_map_pad: str,
    alpha_map_pad: str,
    output_stl_pad: str,
    extra_opts: list[str] | None = None,
) -> bool:
    """Genereer een 3D-model via Blender (headless).

    Dit roept blender_displace.py aan welke:
    - Heightfield mesh bouwt direct uit depth-map (pixel-nauwkeurig)
    - Alpha-masking toepast (verwijdert pixels waar alpha < 0.5)
    - Solid block genereert met bottom/side faces voor 3D-printen
    - STL exporteert (manifold, watertight)

    Args:
        depth_map_pad: Pad naar depth map (grayscale, van stap 2).
        alpha_map_pad: Pad naar alpha map (RGBA, van stap 1).
        output_stl_pad: Pad voor output STL-bestand.

    Returns:
        True bij succes, False bij fout.
    """
    print("[Stap 3] Start: 3D-model genereren via Blender (heightfield + alpha-masking)...")

    if not Path(depth_map_pad).is_file():
        print(f"[Stap 3][Error] Depth map niet gevonden: {depth_map_pad}")
        return False

    if not Path(alpha_map_pad).is_file():
        print(f"[Stap 3][Error] Alpha map niet gevonden: {alpha_map_pad}")
        return False

    try:
        # Voer Blender uit met het displacement script
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

        # Append any extra CLI options (key=value strings) the user supplied
        if extra_opts:
            # Ensure items are strings
            command.extend([str(x) for x in extra_opts])

        print(f"[Stap 3] Blender command: {' '.join(command)}")

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        # Print Blender output voor debugging
        if result.stdout:
            print(f"[Stap 3][Blender Output] {result.stdout}")

        print(f"[Stap 3][Success] 3D-model gegenereerd: {output_stl_pad}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[Stap 3][Error] Blender subprocess faalde met exit code {e.returncode}.")
        if e.stderr:
            print(f"[Stap 3][Blender Error] {e.stderr}")
        return False
    except FileNotFoundError:
        print(
            f"[Stap 3][Error] Blender niet gevonden op pad: {BLENDER_PATH}. "
            "Pas BLENDER_PATH aan in dit script of voeg Blender toe aan PATH."
        )
        return False
    except Exception as e:
        print(f"[Stap 3][Error] Fout tijdens 3D-model generatie: {e}")
        return False


def main() -> None:
    """Voer de volledige pipeline uit."""
    print("[Pipeline] Start verwerking...")

    input_path = Path(INPUT_IMAGE_PATH)
    if not input_path.is_file():
        print(f"[Pipeline][Error] Inputbestand niet gevonden: {INPUT_IMAGE_PATH}")
        return

    if OUTPUT_FOLDER_PATH.strip():
        resolved_output_dir = Path(OUTPUT_FOLDER_PATH)
    else:
        resolved_output_dir = input_path.parent / "output" if USE_OUTPUT_SUBFOLDER else input_path.parent

    os.makedirs(resolved_output_dir, exist_ok=True)
    print(f"[Pipeline] Output map klaar: {resolved_output_dir.resolve()}")
    # Controleer of we de API-stappen moeten overslaan (handig voor iteratief Blender-werk)
    skip_api_flag = os.environ.get("SKIP_API", "").lower() in ("1", "true", "yes")

    if skip_api_flag:
        print("[Pipeline] SKIP_API is ingeschakeld — stap 1 en 2 worden overgeslagen.")
        # Als de gebruiker specifieke paden wil gebruiken, kan dat via env-vars
        depth_path = os.environ.get("DEPTH_PATH")
        alpha_path = os.environ.get("ALPHA_PATH")

        if depth_path is None:
            candidate = resolved_output_dir / "2_depth_map_output.png"
            depth_path = str(candidate)

        if alpha_path is None:
            candidate = resolved_output_dir / "1_removebg_output.png"
            alpha_path = str(candidate)

        print(f"[Pipeline] Verwachte depth map: {depth_path}")
        print(f"[Pipeline] Verwachte alpha map: {alpha_path}")

        if not Path(depth_path).is_file():
            print(f"[Pipeline][Error] Depth map niet gevonden (SKIP_API): {depth_path}")
            return

        if not Path(alpha_path).is_file():
            print(f"[Pipeline][Error] Alpha map niet gevonden (SKIP_API): {alpha_path}")
            return

        removebg_output_path = alpha_path

    else:
        removebg_output_path = remove_background(
            input_image_path=INPUT_IMAGE_PATH,
            output_folder_path=str(resolved_output_dir),
        )

        if not removebg_output_path:
            print("[Pipeline] Gestopt: stap 1 is mislukt.")
            return

        depth_path = generate_depth_map(
            no_bg_image_path=removebg_output_path,
            output_folder_path=str(resolved_output_dir),
        )

        if not depth_path:
            print("[Pipeline] Gestopt: stap 2 is mislukt.")
            return

    # Stap 3: 3D-model genereren via Blender
    output_stl_path = os.path.join(str(resolved_output_dir), "3_3d_model_output.stl")

    # Read optional BLENDER_OPTS from environment. Expected format: "key=val key2=val2"
    blender_opts_raw = os.environ.get("BLENDER_OPTS", "").strip()
    blender_opts = []
    if blender_opts_raw:
        # simple split on whitespace; users can quote values if needed
        blender_opts = blender_opts_raw.split()

    # Keep modifiers editable by default in saved .blend unless explicitly overridden.
    has_apply_modifiers_flag = any(
        str(opt).lower().startswith("apply_modifiers_for_stl=") for opt in blender_opts
    )
    if not has_apply_modifiers_flag:
        blender_opts.append("apply_modifiers_for_stl=false")

    success_3d = generate_3d_model(
        depth_map_pad=depth_path,
        alpha_map_pad=removebg_output_path,
        output_stl_pad=output_stl_path,
        extra_opts=blender_opts if blender_opts else None,
    )

    if not success_3d:
        print("[Pipeline] Klaar met fouten: stap 3 is mislukt.")
        return

    print("[Pipeline] Success: alle stappen succesvol afgerond!")


if __name__ == "__main__":
    main()
