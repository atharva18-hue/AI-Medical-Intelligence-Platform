"""
Advanced AI Medical Intelligence Platform - Main FastAPI Application
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.models.predictor import ChestXrayPredictor
from app.services.database import init_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "chest_xray_model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup - load model and init db
    print("Starting Advanced AI Medical Intelligence Platform...")
    init_db()

    if os.path.exists(MODEL_PATH):
        routes.predictor = ChestXrayPredictor(MODEL_PATH)
        routes.model = routes.predictor.model
        routes.device = routes.predictor.device
        print(f"Model loaded on {routes.device}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("Run: python ml/train.py to train the model first")

    yield
    print("Shutting down...")


app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="Chest X-ray disease detection with Explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

# cors for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routes
app.include_router(routes.router, prefix="/api/v1", tags=["predictions"])

# static files and templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(BASE_DIR, "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/history-page")
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})
