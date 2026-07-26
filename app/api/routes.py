import os
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import PredictionResponse, HistoryItem, HealthResponse
from app.services.database import get_db, save_prediction, get_all_predictions, get_prediction_by_id
from app.services.gradcam import generate_gradcam_image
from app.services.llm_report import generate_medical_report

router = APIRouter()

# these get set from main.py on startup
predictor = None
model = None
device = "cpu"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=device,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(...),
    patient_notes: str = Form(default=""),
    db: Session = Depends(get_db),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # basic validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10mb limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    try:
        result, original_img = predictor.predict(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {str(e)}")

    # generate gradcam heatmap
    gradcam_b64 = None
    try:
        tensor, _ = predictor.preprocess(contents)
        target_layer = model.layer4[-1]  # last conv block in resnet
        gradcam_b64 = generate_gradcam_image(model, tensor, original_img, target_layer)
    except Exception as e:
        print(f"GradCAM failed: {e}")  # dont fail whole request

    # generate report
    report = await generate_medical_report(result, patient_notes)

    # save uploaded file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(save_path, "wb") as f:
        f.write(contents)

    record = save_prediction(
        db,
        filename=saved_name,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        report_text=report,
        gradcam_path=saved_name,
    )

    return PredictionResponse(
        id=record.id,
        filename=saved_name,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        report=report,
        gradcam_image=gradcam_b64,
        created_at=record.created_at,
    )


@router.get("/history", response_model=list[HistoryItem])
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    records = get_all_predictions(db, limit=limit)
    return records


@router.get("/history/{pred_id}")
def get_history_detail(pred_id: int, db: Session = Depends(get_db)):
    record = get_prediction_by_id(db, pred_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {
        "id": record.id,
        "filename": record.filename,
        "predicted_class": record.predicted_class,
        "confidence": record.confidence,
        "report": record.report_text,
        "created_at": record.created_at,
    }
