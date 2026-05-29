import os

class Settings:
    BIAS_MODEL_DIR: str = os.getenv("BIAS_MODEL_DIR", "./models/beritacheck_bias_model")
    SENTIMENT_MODEL_DIR: str = os.getenv("SENTIMENT_MODEL_DIR", "./models/beritacheck_roberta_sentiment_model")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))

settings = Settings()