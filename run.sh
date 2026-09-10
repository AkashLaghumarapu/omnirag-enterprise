#!/usr/bin/env bash
set -e

echo "================================================================"
echo "  Launching OmniRAG Enterprise Platform"
echo "  Agentic CRAG, Hybrid Vector Search, and Automated Benchmarking"
echo "================================================================"
echo ""

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "[1/2] Checking dependencies..."
python3 -m pip install -q -r requirements.txt

echo "[2/2] Starting OmniRAG FastAPI Server on http://localhost:8000 ..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
