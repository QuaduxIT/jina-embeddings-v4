@echo off
REM Copyright (c) 2025-2026 Quadux IT GmbH
REM    ____                  __              __________   ______          __    __  __
REM   / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
REM  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
REM / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
REM \___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/

REM License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
REM Author: Walter Hoffmann
REM
REM Jina Embeddings v4 API - Docker Start Script
REM Usage: start.bat [--cpu]
REM Model is embedded in image - no volume needed!

setlocal

set IMAGE_NAME=quaduxit/jina-embeddings-v4
set CONTAINER_NAME=jina-embed-v4
set HOST_PORT=8090

REM Check for --cpu flag
set CPU_ENV=
set GPU_FLAG=--gpus all
if "%1"=="--cpu" (
    set CPU_ENV=-e FORCE_CPU=1
    set GPU_FLAG=
    echo Running in CPU mode
) else (
    echo Running in GPU mode
)

REM Stop and remove existing container
echo Stopping existing container...
docker rm -f %CONTAINER_NAME% 2>nul

REM Build the image (model download happens here)
echo Building Docker image...
docker build -t %IMAGE_NAME% build/

REM Run the container (no volume needed - model is in image)
echo Starting container...
docker run -d ^
    --name %CONTAINER_NAME% ^
    --restart always ^
    %GPU_FLAG% ^
    -p %HOST_PORT%:8000 ^
    %CPU_ENV% ^
    %IMAGE_NAME%

echo.
echo Container started! Waiting for API to be ready...
echo.

REM Wait for Uvicorn to be ready (check logs every 2 seconds, max 5 minutes)
set MAX_WAIT=150
set WAIT_COUNT=0
:wait_loop
timeout /t 2 /nobreak >nul
docker logs %CONTAINER_NAME% 2>&1 | findstr /C:"Uvicorn running on http://0.0.0.0:8000" >nul
if %errorlevel%==0 (
    echo.
    echo ========================================
    echo API is ready! Available at http://localhost:%HOST_PORT%
    echo ========================================
    echo.
    echo Endpoints:
    echo   GET  /health       - Health check
    echo   POST /embed/text   - Text embeddings
    echo   POST /embed/image  - Image embeddings
    echo.
    goto :done
)
set /a WAIT_COUNT+=1
if %WAIT_COUNT% geq %MAX_WAIT% (
    echo.
    echo ERROR: Timeout waiting for API to start!
    echo Check logs with: docker logs %CONTAINER_NAME%
    goto :done
)
echo Waiting for API... ^(%WAIT_COUNT%/%MAX_WAIT%^)
goto :wait_loop

:done
endlocal
