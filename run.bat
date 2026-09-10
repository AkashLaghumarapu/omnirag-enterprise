@echo off
echo ================================================================
echo   Launching OmniRAG Enterprise Platform
echo   Agentic CRAG, Hybrid Vector Search, and Automated Benchmarking
echo ================================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking dependencies...
python -m pip install -q -r requirements.txt

echo [2/2] Starting OmniRAG FastAPI Server on http://localhost:8000 ...
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
