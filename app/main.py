from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from app.config import settings
from app.pipeline import deep_clean_and_extract_features, BeritaCheckInferenceEngine, scrape_malaysian_news_article
from app.database import init_db, get_db, SavedArticle
from app.tasks import background_process_news_url
from sqlalchemy.orm import Session
from fastapi import Depends
import os

app = FastAPI(
    title="BeritaCheck Backend NLP API Engine",
    version="1.0.0",
    description="Production-ready classification engine for Bias and Sentiment models."
)

# Global engine container singleton
engine = None

@app.on_event("startup")
def bootstrap_models():
    global engine
    if not os.path.exists(settings.BIAS_MODEL_DIR) or not os.path.exists(settings.SENTIMENT_MODEL_DIR):
        print("Warning: Pretrained model weights folders are missing from the workspace.")
    else:
        engine = BeritaCheckInferenceEngine(
            bias_dir=settings.BIAS_MODEL_DIR,
            sentiment_dir=settings.SENTIMENT_MODEL_DIR
        )
        print(" Neural networks loaded into memory successfully.")
    init_db()
class AnalysisRequest(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: str

@app.post("/api/v1/analyze-text")
async def analyze_raw_text(payload: AnalysisRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Provided text string cannot be blank.")
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engines are offline or uninitialized.")

    # 1. Extract notebook engineering metrics
    metrics = deep_clean_and_extract_features(payload.text)
    
    # 2. Model evaluations
    predictions = engine.infer(metrics["polished_text"])
    
    return {
        "status": "success",
        "linguistic_features": {
            "subjectivity_score": metrics["subjectivity_score"],
            "political_density_score": metrics["political_density_score"],
            "lemmatized_sample": metrics["lemmatized_text"][:200]
        },
        "model_outputs": predictions
    }

@app.post("/api/v1/analyze-url")
async def analyze_news_url(payload: ScrapeRequest):
    # 1. Execute live scraping routine
    scraped_text = scrape_malaysian_news_article(str(payload.url))
    if not scraped_text:
        raise HTTPException(status_code=422, detail="Failed to scrape text contents from the target URL.")
        
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference models are uninitialized.")

    metrics = deep_clean_and_extract_features(scraped_text)
    predictions = engine.infer(metrics["polished_text"])

    return {
        "status": "success",
        "source_url": payload.url,
        "extracted_text_snippet": metrics["polished_text"][:300],
        "linguistic_features": {
            "subjectivity_score": metrics["subjectivity_score"],
            "political_density_score": metrics["political_density_score"]
        },
        "model_outputs": predictions
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "models_loaded": engine is not None
    }

@app.post("/api/v1/analyze-url-async")
async def queue_news_url_analysis(payload: ScrapeRequest):
    # Offloads the entire scraping & machine learning pipeline to the background
    task = background_process_news_url.delay(str(payload.url))
    return {
        "status": "queued",
        "task_id": task.id,
        "message": "The article is being processed in the background."
    }

@app.get("/api/v1/articles")
def list_analyzed_articles(db: Session = Depends(get_db)):
    articles = db.query(SavedArticle).order_by(SavedArticle.created_at.desc()).all()
    return {"count": len(articles), "data": articles}