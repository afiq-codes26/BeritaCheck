import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_malaysian_news_article(url: str) -> str:
    """Raw article text parser from Notebook Cell 0."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        if pd.isna(url) or str(url).strip() == "" or "http" not in str(url):
            return ""

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')

        clean_paragraphs = [p.text.strip() for p in paragraphs if len(p.text.strip()) > 20]
        return " ".join(clean_paragraphs)
    except Exception:
        return ""