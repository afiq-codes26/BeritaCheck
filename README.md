# 🔍 BeritaCheck: Malaysian Political Bias & Sentiment Analyser

**BeritaCheck** is a production-ready, full-stack machine learning framework designed to detect political framing, subjective slant, and linguistic sentiment in Malaysian news articles. By combining custom automated web-scraping pipelines with dual fine-tuned Deep Learning models (**BERT** and **RoBERTa**), BeritaCheck parses raw news URLs to deliver real-time transparency into media neutrality.

---

## 🚀 Core Features

* **Automated Web Scraping:** Extracts clean article body text from major Malaysian news URLs while filtering out boilerplate HTML, ads, and navigational noise.
* **Linguistic Slant Engineering:** Computes custom statistical indicators like **Political Density Score** (via targeted keyword-lexicon mapping) and **Subjectivity Metrics**.
* **Dual-Transformer Inference Engine:**
* **BERT (Bidirectional Encoder Representations from Transformers):** Fine-tuned specifically for binary classification to identify political framing vs. objective reporting.
* **RoBERTa (A Robustly Optimized BERT Approach):** Fine-tuned for three-way classification to monitor underlying linguistic sentiment (*Positive*, *Neutral*, *Negative*).


* **Asynchronous Queue Management:** Uses **Celery** and **Redis** to offload heavy NLP model inference blocks away from the user interface thread, preventing server lag.
* **Streamlit Dashboard:** An interactive, single-page web user interface providing immediate visualization metrics and historical analysis tracking.

---

## 🏗️ System Architecture

The codebase separates concerns into three distinct layers to ensure clean maintenance and deployment scalability:

1. **The Machine Learning Pipeline (`beritacheck_datapreprocessing_nlp.py`):** The isolated training codebase responsible for data scraping, feature tokenization, validation tracking, and compiling the weights.
2. **The Backend REST API (`/app`):** Driven by **FastAPI**, handling routing definitions, structural endpoints, background tasks, and **SQLAlchemy/SQLite** model database mapping.
3. **The Frontend (`beritacheck_app-7 (1).py`):** An intuitive dashboard built in **Streamlit** that queries the backend API asynchronously.

---

## 🛠️ Quick Start Guide

### Prerequisites

Ensure your local environment has Python 3.10+, SQLite, and Redis installed.

### 1. Model Training & Export

Set up your model weights folders by executing the machine learning pipeline script:

```bash
# Download the English vocabulary dictionary for spaCy text-processing
python -m spacy download en_core_web_sm

# Place your dataset 'news_links.csv' in the folder and run training
python beritacheck_datapreprocessing_nlp.py

```

*This dumps the fine-tuned weights inside `./models/beritacheck_bias_model/` and `./models/beritacheck_roberta_sentiment_model/`.*

### 2. Start the Backend API

Navigate to the root directory and boot up the FastAPI application loop along with the background task worker:

```bash
# Terminal 1: Run the web server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Run the Celery background task engine
celery -A app.tasks.celery_app worker --loglevel=info

```

> **API Docs Preview:** Access `http://localhost:8000/docs` to test endpoint arrays interactively via Swagger UI.

### 3. Launch the Dashboard Frontend

Run the user interface in a separate terminal wrapper:

```bash
streamlit run "frontend.py"

```

---

## 👥 The BeritaCheck Engine Team

* **Lead Developer:** Eireen Syafeeya — *Architecture, backend implementation, and Streamlit integration.*
* **NLP Engineer:** Afiq — *Feature optimization and keyword lexicon design.*
* **Data Engineer:** Nia — *Data pipeline harvesting and corpus classification.*
* **Data Analyst:** Fatnin — *Statistical distributions and bias margin tracking.*
* **ML Engineer:** Fatimah — *Transformer model training and evaluation metrics.*
* **UI/UX & Tester:** Husna — *Interface design and semantic validation testing.*
