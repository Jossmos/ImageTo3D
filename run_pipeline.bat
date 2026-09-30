@echo off
setlocal enabledelayedexpansion
REM Drag-and-drop pipeline runner
REM Sleep een foto op dit bestand om het automatisch te verwerken

if "%~1"=="" (
    echo.
    echo Fout: Geen bestand opgegeven!
    echo Sleep een afbeelding op dit batch-bestand om het te verwerken.
    echo.
    pause
    exit /b 1
)

set "IMAGE_PATH=%~1"
if not exist "!IMAGE_PATH!" (
    echo.
    echo Fout: Bestand niet gevonden: !IMAGE_PATH!
    echo.
    pause
    exit /b 1
)

echo.
echo Processing: !IMAGE_PATH!
echo.

cd /d "%~dp0"

REM Optional: second argument --skip to bypass API steps and run Blender-only.
REM Usage: run_pipeline.bat "C:\path\to\image.png" --skip [depth_path] [alpha_path]
if /I "%~2"=="--skip" (
    set "SKIP_API=1"
    if not "%~3"=="" set "DEPTH_PATH=%~3"
    if not "%~4"=="" set "ALPHA_PATH=%~4"
)

REM Optioneel: API key voor Remove.bg (indien cloud fallback gebruikt wordt)
REM if not defined REMOVE_BG_API_KEY set "REMOVE_BG_API_KEY=your_key_here"

echo.
echo You can now optionally enter Blender modifier options to tune output.
echo Example: max_height=2.0 smoothing_iters=3 smoothing_factor=0.3 alpha_threshold=0.45
set /p BLENDER_OPTS="Blender options (leave empty for defaults): "

"%~dp0.venv311\Scripts\python.exe" -c "import os; _opts=r'!BLENDER_OPTS!'.strip(); _low=_opts.lower(); _opts=(_opts + ' ' if _opts else '') + 'apply_modifiers_for_stl=false' if 'apply_modifiers_for_stl=' not in _low else _opts; os.environ['BLENDER_OPTS']=_opts; import image_pipeline as p; p.INPUT_IMAGE_PATH=r'!IMAGE_PATH!'; p.OUTPUT_FOLDER_PATH=''; p.USE_OUTPUT_SUBFOLDER=False; p.main()"

echo.
echo Klaar! Resultaten zijn opgeslagen in dezelfde map als je foto.
echo.
pause
