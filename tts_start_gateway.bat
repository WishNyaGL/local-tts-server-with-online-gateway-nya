@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

title TTS System - Local Server + Gateway

echo ============================================================
echo                 TTS SYSTEM
echo        Local TTS + Online TTS Gateway
echo ============================================================
echo.
echo Directory:
echo %CD%
echo.

set "PYTHON=%~dp0tts-venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python virtual environment not found:
    echo.
    echo %PYTHON%
    echo.
    pause
    exit /b 1
)

echo Python:
"%PYTHON%" --version
echo.

rem ------------------------------------------------------------
rem LOCAL TTS SERVER
rem ------------------------------------------------------------

set "TTS_HOST=127.0.0.1"
set "TTS_PORT=8880"
set "TTS_API_KEY=tts-local"

set "PHONEMIZER_ESPEAK_LIBRARY=%~dp0tts-venv\Lib\site-packages\espeakng_loader\espeak-ng.dll"
set "PHONEMIZER_ESPEAK_DATA_PATH=%~dp0tts-venv\Lib\site-packages\espeakng_loader\espeak-ng-data"

set "PIPER_DIR=%~dp0piper"
set "SILERO_DIR=%~dp0silero"
set "KOKORO_DIR=%~dp0kokoro"

set "TTS_MODEL_RU=%PIPER_DIR%\ru_RU-irina-medium.onnx"
set "TTS_MODEL_EN=%PIPER_DIR%\en_US-libritts-high.onnx"

echo Checking model directories...
echo.

if not exist "%PIPER_DIR%\" mkdir "%PIPER_DIR%"
if not exist "%SILERO_DIR%\" mkdir "%SILERO_DIR%"
if not exist "%KOKORO_DIR%\" mkdir "%KOKORO_DIR%"

echo Piper:  %PIPER_DIR%
echo Silero: %SILERO_DIR%
echo Kokoro: %KOKORO_DIR%
echo.

echo Checking Piper models...
echo.

if not exist "%TTS_MODEL_RU%" (
    echo WARNING: Russian Piper model not found:
    echo %TTS_MODEL_RU%
    echo.
)

if not exist "%TTS_MODEL_EN%" (
    echo WARNING: English Piper model not found:
    echo %TTS_MODEL_EN%
    echo.
)

echo.
echo Checking Kokoro files...
echo.

if not exist "%KOKORO_DIR%\kokoro-config.json" echo WARNING: Kokoro config.json not found
if not exist "%KOKORO_DIR%\ru_g2p.py" echo WARNING: Kokoro ru_g2p.py not found
if not exist "%KOKORO_DIR%\onnx\model.onnx" echo WARNING: Kokoro model.onnx not found
if not exist "%KOKORO_DIR%\onnx\model_dima.onnx" echo WARNING: Kokoro model_dima.onnx not found
if not exist "%KOKORO_DIR%\voices\sveta.bin" echo WARNING: Kokoro sveta.bin not found
if not exist "%KOKORO_DIR%\voices\masha.bin" echo WARNING: Kokoro masha.bin not found
if not exist "%KOKORO_DIR%\voices\dima.bin" echo WARNING: Kokoro dima.bin not found

rem ------------------------------------------------------------
rem ONLINE GATEWAY
rem ------------------------------------------------------------

set "TTS_PROXY_HOST=127.0.0.1"
set "TTS_PROXY_PORT=8890"
set "TTS_LOCAL_URL=http://127.0.0.1:8880/v1/audio/speech"

rem IMPORTANT:
rem Put your API key here, or set it before starting.
rem Do not publish this BAT file with a real key.
set "API_KEY="

echo.
echo ============================================================
echo LOCAL TTS:
echo   http://127.0.0.1:8880/v1/audio/speech
echo.
echo TTS GATEWAY:
echo   http://127.0.0.1:8890/v1/audio/speech
echo   Web UI: http://127.0.0.1:8890/
echo.
echo Online mode is switched in the Gateway Web UI.
echo Default: LOCAL
echo ============================================================
echo.

echo Starting local TTS server...
start "Local TTS Server :8880" cmd /k ""%PYTHON%" "%~dp0tts_server.py""

timeout /t 2 /nobreak >nul

echo Starting TTS Gateway :8890...
start "TTS Gateway :8890" cmd /k ""%PYTHON%" "%~dp0tts_online_proxy.py""

echo.
echo ============================================================
echo Both services started.
echo.
echo Open in browser:
echo   http://127.0.0.1:8890/
echo.
echo IMPORTANT:
echo For the game use:
echo   http://127.0.0.1:8890/v1/audio/speech
echo.
echo The old local server remains available at:
echo   http://127.0.0.1:8880/v1/audio/speech
echo ============================================================
echo.

pause
