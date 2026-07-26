# Advanced AI Medical Intelligence Platform

> Chest X-Ray Disease Detection using Deep Learning

A web-based application that analyzes chest X-ray images to detect **Normal** vs **Pneumonia** using a CNN (ResNet18), explains predictions with **Grad-CAM**, generates medical reports using **LLM**, and stores history in **SQLite**.

---

## project live URL
https://medical-ai-platform-0ohx.onrender.com/

------

## Features

- Deep Learning based disease prediction (ResNet18)
- Explainable AI with Grad-CAM heatmaps
- AI-assisted medical report generation (OpenAI / template fallback)
- REST API built with FastAPI
- SQLite database for prediction history
- Simple Bootstrap web UI
- Docker support

---

## Project Structure

```
medical-ai-platform/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── api/routes.py        # REST endpoints
│   ├── models/              # CNN model & predictor
│   ├── services/            # GradCAM, LLM, Database
│   └── static/              # CSS, JS
├── ml/
│   ├── train.py             # Model training script
│   └── create_sample_data.py
├── models/
│   └── chest_xray_model.pth # Trained weights
├── templates/               # HTML pages
├── data/
├── docs/                    # Project report
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/medical-ai-platform.git
cd medical-ai-platform

python -m venv venv
source venv/bin/activate   # on windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset & Train the Model

**Download** (automatic — uses Hugging Face if Kaggle key not set):

```bash
python ml/download_dataset.py
```

**Train** on full dataset (~15 min on Apple Silicon):

```bash
python ml/train.py --epochs 15 --batch-size 32
```

**Option A - Full dataset (recommended for best accuracy)**

Dataset is auto-downloaded to `data/chest_xray/` (5,216 train images). Or manually download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

```bash
python ml/train.py --epochs 15 --batch-size 32
```

**Option B - Demo mode (quick start)**

```bash
python ml/create_sample_data.py
python ml/train.py --demo --epochs 5
```

### 3. Run the Application

```bash
uvicorn app.main:app --reload --port 8000
```

Open browser: **http://localhost:8000**

### 4. (Optional) LLM Reports

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Without API key, the app uses a template-based report (still works fine for demo).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict` | Upload X-ray & get prediction |
| GET | `/api/v1/history` | List past predictions |
| GET | `/api/v1/history/{id}` | Get single prediction detail |

### Example - Predict via curl

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@your_xray.jpg" \
  -F "patient_notes=fever and cough"
```

---

## Docker Deployment

```bash
docker build -t mediscan .
docker run -p 8000:8000 mediscan
```

## Live Deployment (Render)

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for step-by-step guide.

Quick steps:
1. Push repo to GitHub
2. Connect to [Render](https://render.com) → New Blueprint → select repo
3. Live URL: https://medical-ai-platform-0ohx.onrender.com/


---

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18 (transfer learning) |
| Input Size | 224 x 224 RGB |
| Classes | Normal, Pneumonia |
| Framework | PyTorch |

Grad-CAM is applied on the last convolutional block (`layer4`) to highlight regions influencing the prediction.

---

## Tech Stack

- **Backend:** Python, FastAPI, SQLAlchemy
- **ML:** PyTorch, Torchvision, scikit-learn
- **XAI:** Grad-CAM (custom implementation)
- **LLM:** OpenAI GPT-3.5 (optional)
- **Frontend:** HTML, Bootstrap 5, JavaScript
- **Database:** SQLite
- **Deployment:** Docker, Uvicorn

---

## Limitations & Disclaimer

⚠️ This project is for **educational and research purposes only**.  
It is NOT a substitute for professional medical diagnosis. Always consult a qualified doctor.

---

## Author

**Atharva Chavhan**  
Sipna college of engineering Amravati 
atharvachavhan18@gmail.com

---

## References

1. Selvaraju et al. - Grad-CAM: Visual Explanations from Deep Networks
2. Chest X-Ray Images (Pneumonia) Dataset - Kaggle
3. PyTorch Transfer Learning Tutorial
4. FastAPI Documentation
