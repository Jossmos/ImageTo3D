"""
Tweede Pipeline: Zet een handmatig aangepast .blend bestand om naar een massief, uitsneden STL-model.

Gebruik:
    python pipeline_blend_to_solid.py --blend my_mesh.blend --alpha my_alpha.png --output result.stl

Of via environment variables:
    set BLEND_FILE=my_mesh.blend
    set ALPHA_MAP=my_alpha.png
    python pipeline_blend_to_solid.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Blender executable path. Pas dit aan als Blender ergens anders is geïnstalleerd.
BLENDER_PATH: str = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

def main():
    parser = argparse.ArgumentParser(description="Zet een .blend bestand om naar een massief STL met alpha-uitsnede.")
    parser.add_argument("dropped_file", nargs='?', help="Pad naar het gedropte .blend bestand (voor drag-and-drop)")
    parser.add_argument("--blend", help="Pad naar het .blend bestand", default=os.environ.get("BLEND_FILE"))
    parser.add_argument("--alpha", help="Pad naar de alpha map (.png)", default=os.environ.get("ALPHA_MAP"))
    parser.add_argument("--output", help="Pad voor output STL", default=os.environ.get("OUTPUT_STL"))
    
    args = parser.parse_args()
    
    if args.dropped_file:
        args.blend = args.dropped_file
    
    if not args.blend:
        print("[Pipeline] Fout: Sleep een .blend bestand op de snelkoppeling, of gebruik --blend.")
        sys.exit(1)
        
    blend_path = Path(args.blend).resolve()
    
    if not args.alpha:
        # Zoek automatisch naar "1_removebg_output.png" in dezelfde map
        candidate_alpha = blend_path.parent / "1_removebg_output.png"
        if candidate_alpha.is_file():
            args.alpha = str(candidate_alpha)
        else:
            print(f"[Pipeline] Fout: --alpha niet opgegeven en kan {candidate_alpha.name} niet automatisch vinden in {blend_path.parent}.")
            sys.exit(1)
            
    alpha_path = Path(args.alpha).resolve()
    
    if not args.output:
        # Als er geen output is opgegeven, sla op naast het blend bestand
        output_path = blend_path.with_name(blend_path.stem + "_solid.stl")
    else:
        output_path = Path(args.output).resolve()
        
    print("="*60)
    print(" 2e Pipeline: Blend -> Solid STL ")
    print("="*60)
    print(f"[Pipeline] Blend bestand: {blend_path}")
    print(f"[Pipeline] Alpha map    : {alpha_path}")
    print(f"[Pipeline] Output STL   : {output_path}")
    
    if not blend_path.is_file():
        print(f"[Pipeline] Fout: Blend bestand niet gevonden op {blend_path}")
        sys.exit(1)
        
    if not alpha_path.is_file():
        print(f"[Pipeline] Fout: Alpha map niet gevonden op {alpha_path}")
        sys.exit(1)
        
    # Check of blender.exe bestaat
    if not Path(BLENDER_PATH).is_file():
        print(f"[Pipeline] Fout: Blender executable niet gevonden op {BLENDER_PATH}.")
        print("Pas BLENDER_PATH aan in pipeline_blend_to_solid.py.")
        sys.exit(1)

    # Commando opbouwen:
    # blender <blend_file> -b -P blender_apply_alpha_solidify.py -- <alpha_map> <output_stl>
    cmd = [
        BLENDER_PATH,
        str(blend_path),
        "--background",
        "--python-exit-code", "1",
        "--python",
        "blender_apply_alpha_solidify.py",
        "--",
        str(alpha_path),
        str(output_path)
    ]
    
    print(f"\n[Pipeline] Blender starten...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"[Pipeline] Succes! Het massieve STL-bestand is opgeslagen op:\n{output_path}")
    except subprocess.CalledProcessError as e:
        print("[Pipeline] Fout tijdens uitvoeren van Blender script:")
        print(e.stderr)
        if e.stdout:
            print("\nBlender Output:\n", e.stdout)
        sys.exit(1)
    except Exception as e:
        print(f"[Pipeline] Onverwachte fout: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
