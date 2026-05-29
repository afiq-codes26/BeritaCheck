# app/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Using a local SQLite file database for effortless setup. 
# Switch this string out for a PostgreSQL URL when pushing to production.
DATABASE_URL = "sqlite:///./beritacheck.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SavedArticle(Base):
    """The database schema mapping directly to your notebook data outputs."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=True)
    raw_text = Column(Text, nullable=False)
    polished_text = Column(Text, nullable=False)
    
    # Custom Notebook Features
    subjectivity_score = Column(Float, nullable=False)
    political_density_score = Column(Float, nullable=False)
    
    # Deep Learning Model Outputs
    bias_prediction = Column(String, nullable=False)
    bias_confidence = Column(Float, nullable=False)
    sentiment_prediction = Column(String, nullable=False)
    sentiment_confidence = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Database dependency worker to safely open and close ports on web requests
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auto-create the table schema when the application spins up
def init_db():
    Base.metadata.create_all(bind=engine)