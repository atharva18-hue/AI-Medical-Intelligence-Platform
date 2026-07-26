"""
SQLite database for storing prediction history.
Chose SQLite because easy to setup, no separate db server needed.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "predictions.db")
DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    predicted_class = Column(String(50))
    confidence = Column(Float)
    report_text = Column(Text, nullable=True)
    gradcam_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db, filename, predicted_class, confidence, report_text=None, gradcam_path=None):
    record = PredictionRecord(
        filename=filename,
        predicted_class=predicted_class,
        confidence=confidence,
        report_text=report_text,
        gradcam_path=gradcam_path,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_all_predictions(db, limit=50):
    return db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit).all()


def get_prediction_by_id(db, pred_id):
    return db.query(PredictionRecord).filter(PredictionRecord.id == pred_id).first()
