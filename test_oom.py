import os
import sys
from PIL import Image
from rembg import new_session, remove

input_path = r"C:\Users\jossm\ImageTo3D\Images\group\WhatsApp Image 2026-05-14 at 15.36.05.jpeg"

def test_size(max_dim):
    print(f"--- Testing max_dim={max_dim} ---", flush=True)
    img = Image.open(input_path).convert('RGB')
    if max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    print(f"Image size to process: {img.size}", flush=True)

    try:
        session = new_session("bria-rmbg")
        res = remove(img, session=session)
        print("Success!", flush=True)
    except Exception as e:
        print("Error:", e, flush=True)

test_size(1024)
test_size(None)
