import os
from pathlib import Path

# Configure SKIP mode and paths
os.environ['SKIP_API'] = '1'
os.environ['DEPTH_PATH'] = r'C:\Users\jossm\ImageTo3D\Images\Mila\2_depth_map_output.png'
os.environ['ALPHA_PATH'] = r'C:\Users\jossm\ImageTo3D\Images\Mila\1_removebg_output.png'

# Set the input image path (used to determine output folder if OUTPUT_FOLDER_PATH is empty)
import image_pipeline as p
p.INPUT_IMAGE_PATH = r'C:\Users\jossm\ImageTo3D\Images\Mila\2_depth_map_output.png'
p.OUTPUT_FOLDER_PATH = ''
p.USE_OUTPUT_SUBFOLDER = False

if __name__ == '__main__':
    p.main()
