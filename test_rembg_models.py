import os
from pathlib import Path
from PIL import Image
from rembg import remove, new_session

input_image = r"C:\Users\jossm\ImageTo3D\Images\Rakker\20260514_103830.jpg"
output_dir = r"C:\Users\jossm\ImageTo3D\Images\Rakker"

# We test the top 3 state-of-the-art models available in rembg
models = [
    "bria-rmbg",         # Bria AI RMBG 1.4 - Extremely good for general & hair
    "birefnet-general",  # BiRefNet - State of the art high res
    "isnet-general-use"  # IS-Net - Also very good general purpose
]

if not os.path.exists(input_image):
    print(f"File not found: {input_image}")
    exit(1)

with Image.open(input_image) as img:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        img_rgb.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
    else:
        img_rgb = img.convert("RGB")

print("Start testing models...")

for model_name in models:
    try:
        print(f"Loading session for {model_name}...")
        session = new_session(model_name)
        
        # We also test with and without post_process_mask
        for post_process in [False, True]:
            print(f"Processing with {model_name} (post_process={post_process})...")
            result = remove(
                img_rgb, 
                session=session, 
                post_process_mask=post_process
            )
            
            suffix = "_postprocess" if post_process else ""
            out_path = Path(output_dir) / f"test_{model_name}{suffix}.png"
            result.save(out_path)
            print(f"Saved: {out_path}")
            
    except Exception as e:
        print(f"Failed with {model_name}: {e}")

print("Done! Check the /Rakker folder for the results.")
