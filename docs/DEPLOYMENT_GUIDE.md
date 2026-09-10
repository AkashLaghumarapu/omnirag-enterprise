# OmniRAG Enterprise: Deployment & Hosting Guide

This guide details how to run OmniRAG Enterprise locally or deploy it to free/production cloud platforms (Hugging Face Spaces, Render, Railway, Docker).

---

## 1. Local One-Click Execution

### Windows:
Double-click `run.bat` or run in PowerShell:
```powershell
.\run.bat
```

### macOS / Linux:
```bash
chmod +x run.sh
./run.sh
```

### Manual Command:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, open your browser to **`http://localhost:8000`**.

---

## 2. Free Cloud Deployment Option A: Hugging Face Spaces (Recommended)

Hugging Face Spaces offers **free CPU hosting** with persistent public HTTPS links, ideal for adding directly to your resume or portfolio.

### Step-by-Step Instructions:
1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New Space** -> Choose **Docker** as the Space SDK -> Select **Blank**.
3. Set Space hardware to **Free CPU (2 vCPU, 16GB RAM)**.
4. Clone your new Space repo or upload the project files:
   - Upload: `Dockerfile`, `requirements.txt`, `backend/`, `frontend/`, `README.md`.
5. Under Space **Settings** -> **Repository Secrets**, optionally add:
   - `GEMINI_API_KEY`: your Google AI Studio API key (free tier available at [aistudio.google.com](https://aistudio.google.com/)).
6. Hugging Face will automatically build the Docker image and launch the live web application with a permanent URL:
   `https://huggingface.co/spaces/<your-username>/omnirag-enterprise`
7. Add this link directly to your resume!

---

## 3. Cloud Deployment Option B: Render (Free Web Service)

Render provides a free web service tier with automated GitHub deployment.

### Step-by-Step:
1. Push this project to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy OmniRAG Enterprise"
   git branch -M main
   git remote add origin https://github.com/<your-username>/omnirag-enterprise.git
   git push -u origin main
   ```
2. Log in to [render.com](https://render.com/) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Settings:
   - **Environment**: `Docker` (or `Python 3`)
   - **Build Command**: `pip install -r requirements.txt` (if Python)
   - **Start Command**: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Add Environment Variables:
   - `GEMINI_API_KEY` (optional, for Gemini 1.5 Flash cloud generation)
6. Click **Deploy Web Service**. You will get a live URL:
   `https://omnirag-enterprise.onrender.com`

---

## 4. Cloud Deployment Option C: Railway

1. Install Railway CLI or connect via [railway.app](https://railway.app/).
2. Run:
   ```bash
   railway up
   ```
3. In Railway settings, click **Generate Domain** to get a public URL.

---

## 5. Docker Deployment

### Build the Docker Image:
```bash
docker build -t omnirag-enterprise .
```

### Run Container:
```bash
docker run -p 8000:8000 --name omnirag -e GEMINI_API_KEY="your-api-key" omnirag-enterprise
```

### Docker Compose:
```bash
docker-compose up -d
```
The application will be running on `http://localhost:8000`.

---

## 6. Environment Variables Reference

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | Optional | `""` | Enables Google Gemini 1.5 Flash generation and text-embedding-004. If omitted, the system seamlessly uses the deterministic offline semantic vectorizer. |
| `GROQ_API_KEY` | Optional | `""` | Enables Groq Llama-3.3 70B ultra-fast inference. |
| `OPENAI_API_KEY` | Optional | `""` | Enables OpenAI API endpoints. |
| `PORT` | Optional | `8000` | Port for the FastAPI HTTP server. |
| `HOST` | Optional | `0.0.0.0` | Bind address. |
