# app/tasks.py
import os
from celery import Celery
from app.database import SessionLocal, SavedArticle
from app.pipeline import scrape_malaysian_news_article, deep_clean_and_extract_features, BeritaCheckInferenceEngine

# Configure Celery to use Redis as the temporary message storage manager.
# Defaulting to localhost for local testing.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("beritacheck_workers", broker=REDIS_URL, backend=REDIS_URL)

# Initialize the engine once globally for the worker environment thread
_worker_inference_engine = None

def get_inference_engine():
    """Lazy initializer to make sure models load once inside the background worker container."""
    global _worker_inference_engine
    if _worker_inference_engine is None:
        _worker_inference_engine = BeritaCheckInferenceEngine(
            bias_dir="./models/beritacheck_bias_model",
            sentiment_dir="./models/beritacheck_roberta_sentiment_model"
        )
    return _worker_inference_engine

@celery_app.task(name="tasks.background_process_news_url")
def background_process_news_url(url: str):
    """
    Worker task: takes a URL, scrapes it, runs the pipeline,
    and commits the analysis output array to the database.
    """
    db = SessionLocal()
    try:
        # 1. Prevent duplicate work if the URL already exists in database
        existing = db.query(SavedArticle).filter(SavedArticle.url == url).first()
        if existing:
            return f"Skipped: URL {url} already fully analyzed in database records."

        # 2. Scrape raw page asset text from pipeline module
        raw_text = scrape_malaysian_news_article(url)
        if not raw_text.strip():
            return f"Failed: Could not read text elements from {url}."

        # 3. Clean and map custom linguistic features from preprocessor
        metrics = deep_clean_and_extract_features(raw_text)
        
        # 4. Trigger Neural Network Predictions via Engine Singelton
        engine = get_inference_engine()
        predictions = engine.infer(metrics["polished_text"])

        # 5. Build and commit the new entry into SQLite database tracking tables
        new_article = SavedArticle(
            url=url,
            raw_text=raw_text,
            polished_text=metrics["polished_text"],
            subjectivity_score=metrics["subjectivity_score"],
            political_density_score=metrics["political_density_score"],
            bias_prediction=predictions["bias_analysis"]["label"],
            bias_confidence=predictions["bias_analysis"]["confidence"],
            sentiment_prediction=predictions["sentiment_analysis"]["label"],
            sentiment_confidence=predictions["sentiment_analysis"]["confidence"]
        )
        
        db.add(new_article)
        db.commit()
        return f"Success: Core data elements parsed and saved for {url}."

    except Exception as e:
        db.rollback()
        return f"Error encountered while running task pipeline background jobs: {str(e)}"
    finally:
        db.close()