# ImageTo3D Pipeline

Automatische conversie van 2D-afbeeldingen naar 3D-reliëf STL-bestanden voor 3D-printen.

## Werkwijze (3 stappen)

### Stap 1: Achtergrond verwijderen
- **Tool**: Remove.bg API
- **Input**: JPG/PNG afbeelding met achtergrond
- **Output**: PNG met transparante achtergrond (`1_removebg_output.png`)

### Stap 2: Dieptemap genereren
- **Tool**: Lokale Depth-Anything-V2 GPU inference
- **Input**: PNG van stap 1
- **Output**: Grayscale depth-map (`2_depth_map_output.png`)

### Stap 3: 3D-reliëf genereren
- **Tool**: Blender headless + Python script
- **Methode**: Heightfield mesh + alpha-masking (pixel-nauwkeurig)
- **Input**: Depth-map + alpha-map
- **Output**: Manifold STL voor 3D-printen (`3_3d_model_output.stl`)

---

## Installatie

### Vereisten
- **Python 3.8+** (aanbevolen 3.10+)
- **Blender 5.0+** (headless-modus)
- **Internet** (voor Remove.bg API; depth draait lokaal op GPU)

### Setup

```bash
# 1. Clone/download de repository
cd ImageTo3D

# 2. Maak een Python virtual environment
python -m venv .venv

# 3. Activeer venv (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Installeer dependencies
pip install requests gradio_client trimesh numpy

# 5. (Optioneel) Controleer Blender-pad in image_pipeline.py
# Zoek naar: BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
```

---

## Gebruik

### 1. **Volledige pipeline** (alle 3 stappen)

```python
# Zet API-keys in en voer uit
python image_pipeline.py
# Input: bepaald door INPUT_IMAGE_PATH in script
# Output: naast input-afbeelding of in OUTPUT_FOLDER_PATH
```

Pas de volgende variabelen in `image_pipeline.py` aan:
```python
INPUT_IMAGE_PATH = "path/to/your/image.jpg"
OUTPUT_FOLDER_PATH = ""  # Laat leeg = op dezelfde locatie als input
REMOVE_BG_API_KEY = "jouw-api-key-hier"  # Haal op van remove.bg
```

### 2. **Blender-only** (sla API-stappen over)

Handig voor iteratie of testen (voorkomt API-limieten).

```python
# Met environment-variabelen
import os
os.environ['SKIP_API'] = '1'
os.environ['DEPTH_PATH'] = 'pad/naar/depth.png'
os.environ['ALPHA_PATH'] = 'pad/naar/alpha.png'

import image_pipeline as p
p.INPUT_IMAGE_PATH = 'dummy'
p.main()
```

Of via PowerShell:
```powershell
$env:SKIP_API='1'
$env:DEPTH_PATH='C:\...\2_depth_map_output.png'
$env:ALPHA_PATH='C:\...\1_removebg_output.png'
.\.venv\Scripts\python.exe -c "import image_pipeline as p; p.INPUT_IMAGE_PATH='dummy'; p.main()"
```

### 3. **Drag-and-drop** (Windows batch)

```bash
run_pipeline.bat "C:\path\to\image.jpg"
```

Voor Blender-only:
```bash
run_pipeline.bat "C:\path\to\image.jpg" --skip "C:\path\to\depth.png" "C:\path\to\alpha.png"
```

---

## Outputs

| Stap | Bestand | Beschrijving |
|------|---------|-------------|
| 1 | `1_removebg_output.png` | PNG met alpha-kanaal (transparante achtergrond) |
| 2 | `2_depth_map_output.png` | Grayscale dieptemap (witter = hoger) |
| 3 | `3_3d_model_output.stl` | Solid 3D-blok, geschikt voor 3D-printen |

---

## Technische Details

### Blender-script (`blender_displace.py`)

De kern van stap 3. Het script:

1. **Laadt afbeeldingen**: Depth-map en alpha-map
2. **Bouwt heightfield mesh**:
   - Samples depth-map met ~300 punten (instelbaar)
   - Schaal adapteert zich naar afbeeldingsverhouding
3. **Appliceert alpha-masking**:
   - Verwijdert vertices/faces waar alpha < 0.5
   - Pixel-exacte contouruitsnede
4. **Maakt solid block**:
   - Berekent perimeter van gemaskeerde mesh
   - Voegt bottom-face toe (Z = -2.0)
   - Voegt side-faces toe (z-hoogte interpolatie)
5. **Exporteert STL**:
   - Fallback ASCII-writer (voor headless-omgevingen)
   - Triangulatie en normal-recalc

### Validatie & Reparatie

Gebruik `trimesh` voor validatie:

```python
import trimesh
m = trimesh.load('3_3d_model_output.stl', force='mesh')
print('Watertight:', m.is_watertight)
print('Winding consistent:', m.is_winding_consistent)

# Reparatie
from trimesh import repair
repair.fill_holes(m)
m.export('3_3d_model_output_repaired.stl')
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'requests'"
```bash
pip install requests gradio_client
```

### "REMOVE_BG_API_KEY is leeg"
- Meld aan bij [remove.bg](https://remove.bg)
- Kopieer je API-key naar `image_pipeline.py`

### "Hugging Face quota exceeded"
- Wacht enkele uren, of
- Gebruik `SKIP_API=1` mode (Blender-only)
- De depth-stap draait lokaal via Depth-Anything-V2 en heeft geen HF-token nodig

### "Blender not found"
- Pas `BLENDER_PATH` aan in `image_pipeline.py`
- Zorg dat Blender 5.0+ is geïnstalleerd

### "STL is niet watertight"
- Normaal voor complex geometrie; gebruik `trimesh` reparatie
- Of verhoog `max_height` in `blender_displace.py` voor meer detail

---

## Performance Tips

1. **Verlaag sample-resolutie** (voor snellere Blender-runs):
   ```python
   # In blender_displace.py, regel ~329
   max_samples = 150  # In plaats van 300
   ```

2. **Gebruik SKIP_API** voor Blender-iteratie (geen API-delays)

3. **Cacheer depth-maps** als je dezelfde afbeelding herhaaldelijk verwerkt

---

## Licentie & Credits

- **Remove.bg**: Externe API (requires account + quota)
- **Depth-Anything-V2**: lokaal via de gekloonde officiële GitHub-repo
- **Blender**: Open-source 3D-software
- **trimesh**: Python 3D-library voor validatie

---

## Nieuwe Video-to-3D Pipeline

Voor video-naar-mesh reconstructie staat er een volledig losse pipeline klaar in `video_to_3d_pipeline.py`.

### Kenmerken

- Extraheert frames uit video naar `working_dir/frames`
- Schrijft raw depth maps naar `working_dir/depths` als `.npy`
- Draait COLMAP via `subprocess`
- Fuseert depth + poses met Open3D TSDF
- Exporteert de finale mesh als `final_output.obj` of `.stl`

### Dependencies

Gebruik de aparte dependency lijst:

```bash
pip install -r requirements_video3d.txt
```

### Voorbeeld

```bash
python video_to_3d_pipeline.py --video-path path/to/input.mp4 --working-dir path/to/workdir --output-mesh final_output.obj
```

---

## Contact & Support

Voor bugs of vragen: update `blender_displace.py` of `image_pipeline.py` naar de nieuwste versie.

---

**Versie**: 1.0 (mai 2026)
**Status**: Productie-gereed (heightfield + alpha-masking)
