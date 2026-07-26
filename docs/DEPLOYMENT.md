# Live Deployment Guide

Deploy **Advanced AI Medical Intelligence Platform** for free using **Render** (recommended) or **Hugging Face Spaces**.

---

## Option A: Render (Recommended)

### Step 1 — Push code to GitHub

```bash
cd ~/Documents/medical-ai-platform

# first time only - create repo at https://github.com/new (name: medical-ai-platform)
git add -A
git commit -m "MediScan AI - medical intelligence platform with Grad-CAM and LLM reports"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medical-ai-platform.git
git push -u origin main
```

> **Note:** The trained model (`models/chest_xray_model.pth`, ~43MB) is included in the repo. The full dataset (`data/chest_xray/`) is excluded — not needed for deployment.

### Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) → Sign up (free)
2. Click **New +** → **Blueprint**
3. Connect your GitHub account
4. Select the `medical-ai-platform` repository
5. Render reads `render.yaml` automatically
6. Click **Apply** → wait 10–15 min for Docker build

### Step 3 — Get your live URL

After deploy succeeds you'll get a URL like:

```
https://medical-ai-platform-xxxx.onrender.com
```

Add this to your README and project report under **Live Deployment Link**.

### Optional — OpenAI reports on Render

In Render dashboard → your service → **Environment** → add:

| Key | Value |
|-----|-------|
| `OPENAI_API_KEY` | your OpenAI key |

Without this, template-based reports still work.

---

## Option B: Hugging Face Spaces (Docker)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name: `medical-ai-platform`
3. SDK: **Docker**
4. Create space
5. Push your code:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/medical-ai-platform
git push hf main
```

HF Spaces gives more RAM (better for PyTorch). Good backup if Render free tier runs out of memory.

---

## Verify deployment

```bash
curl https://YOUR-APP-URL.onrender.com/api/v1/health
```

Expected:
```json
{"status":"ok","model_loaded":true,"device":"cpu"}
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails (out of memory) | Use Hugging Face Spaces instead |
| `model_loaded: false` | Ensure `models/chest_xray_model.pth` is in GitHub repo |
| App sleeps on free tier | First request after idle takes ~30s (Render free tier spins down) |
| History not persisting | SQLite resets on redeploy (free tier) — mention in report as limitation |

---

## Local Docker test (before deploying)

```bash
docker build -t mediscan .
docker run -p 8000:8000 mediscan
```

Open http://localhost:8000
