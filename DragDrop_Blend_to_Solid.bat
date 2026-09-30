@echo off
title Blend to Solid Pipeline
echo =======================================================
echo          ImageTo3D: Drag and Drop Processor
echo =======================================================
echo.

if "%~1"=="" (
    echo [Fout] Je hebt geen bestand meegesleept!
    echo Sleep een .blend bestand bovenop deze snelkoppeling.
    echo.
    pause
    exit /b 1
)

echo Start verwerking van: %1
echo.

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%pipeline_blend_to_solid.py" "%~1"

echo.
echo Druk op een toets om dit venster te sluiten...
pause >nul
