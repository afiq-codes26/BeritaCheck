import re
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

POLITICAL_KEYWORDS = [
    'minister', 'government', 'anwar', 'ibrahim', 'dzulkefly', 'bersatu', 'loke siew fook',
    'macc', 'corruption', 'subsidies', 'najib', 'razak', 'court', 'amend', 'state speaker',
    'assemblyman', 'parti', 'dap', 'pkr', 'pas', 'perikatan', 'nasional', 'barisan', 'umno',
    'jurisdiction', 'legal', 'execution', 'allocation', 'procurement', 'policy', 'ft'
]

def polish_malaysian_text(text: str) -> str:
    """Advanced Malaysian Text Polishing Function from Notebook Cell 1."""
    text = str(text)

    ad_phrases = [
        r"Subscribe to our FREE\s?Newsletter",
        r"Telegram and WhatsApp channels",
        r"Click on the points to redeem your rewards",
        r"Stay signed in to save the points"
    ]
    for phrase in ad_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)

    # Strip Malaysian media datelines (KUALA LUMPUR, Bernama, The Edge styles)
    text = re.sub(r"^[A-Z\s]+,\s[A-Z][a-z]+\s\d+\s\([^)]+\)\s?[-–—]+", "", text)
    text = re.sub(r"^[A-Z\s]+\s\([^)]+\):\s?", "", text)
    text = re.sub(r"^[A-Z\s]+,\s[A-Z][a-z]+\s\d+\s?[-–—]+", "", text)

    return " ".join(text.split())

def deep_clean_and_extract_features(raw_text: str) -> dict:
    """Processes linguistic properties, calculated density and engineering indices."""
    polished = polish_malaysian_text(raw_text)
    
    # TextBlob subjectivity feature extraction
    subjectivity_score = TextBlob(polished).sentiment.subjectivity
    
    # Deep clean text routine mapping
    clean_text = polished.lower()
    clean_text = re.sub(r'http\S+|www\.\S+', '', clean_text)
    clean_text = re.sub(r'[^a-z0-9\s]', '', clean_text)
    clean_text = " ".join(clean_text.split())
    
    # Stopword processing alignment
    processed_words = [w for w in clean_text.split() if w not in stop_words]
    processed_text = " ".join(processed_words)
    
    # Keyword political alignment density
    political_density = 0.0
    if processed_words:
        match_count = sum(1 for word in processed_words if any(kw in word for kw in POLITICAL_KEYWORDS))
        political_density = match_count / len(processed_words)
        
    # Lemmatized text transformation logic
    doc = nlp(polished if polished.strip() else " ")
    lemmatized_text = " ".join([word.lemma_.lower() for word in doc if not word.is_punct and not word.is_space and not word.is_stop])
    
    return {
        "polished_text": polished,
        "processed_text": processed_text,
        "lemmatized_text": lemmatized_text,
        "subjectivity_score": float(subjectivity_score),
        "political_density_score": float(political_density)
    }