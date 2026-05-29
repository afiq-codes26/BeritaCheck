import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time
import datetime
import json
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# ── BACKEND ROUTING CONFIGURATION ─────────────────────────────────────────────
# Change this URL string to match your production domain when you deploy to Render/Railway
BACKEND_API_URL = "http://localhost:8000/api/v1"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BeritaCheck",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_nlp_resources():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

stop_words = load_nlp_resources()

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("history", []), ("active_tab", "home"),
                 ("url_input", ""), ("result", None), ("theme_mode", "Dark")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Dynamic Theme CSS Injector ────────────────────────────────────────────────
if st.session_state.theme_mode == "Dark":
    bg, text, accent, card_bg, border = "#0e1117", "#ffffff", "#ff4b4b", "#1e222b", "#31353f"
    history_lbl = "#a0a0a0"
else:
    bg, text, accent, card_bg, border = "#ffffff", "#1f2937", "#ff4b4b", "#f3f4f6", "#e5e7eb"
    history_lbl = "#4b5563"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    h1, h2, h3 {{ color: {text} !important; font-weight: 700 !important; }}
    div[data-testid="stSidebar"] {{ background-color: {card_bg}; border-right: 1px solid {border}; }}
    .stButton>button {{
        background-color: {accent} !important; color: white !important;
        font-weight: 600; border-radius: 8px; border: none; width: 100%; transition: 0.3s;
    }}
    .stButton>button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
    .clear-btn>button {{
        background-color: transparent !important; color: {text} !important;
        border: 1px solid {border} !important;
    }}
    .clear-btn>button:hover {{ background-color: {card_bg} !important; }}
    
    /* Result styling cards */
    .verdict-card {{
        background-color: {card_bg}; padding: 24px; border-radius: 12px;
        margin: 20px 0; border: 1px solid {border}; border-left: 5px solid {accent};
    }}
    .verdict-header {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 4px; }}
    .verdict-title {{ font-size: 28px; font-weight: 800; color: {text}; margin-bottom: 12px; }}
    
    /* Metrics grids layout */
    .metric-row {{ display: flex; gap: 16px; margin-bottom: 16px; }}
    .metric-box {{
        background: {card_bg}; padding: 16px; border-radius: 8px; flex: 1;
        border: 1px solid {border}; text-align: center;
    }}
    .metric-val {{ font-size: 20px; font-weight: 700; color: {accent}; }}
    .metric-lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
    
    /* Team layout elements */
    .team-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-top: 30px; }}
    .team-card {{
        background: {card_bg}; border: 1px solid {border}; border-radius: 12px;
        padding: 24px; text-align: center; transition: 0.3s;
    }}
    .team-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
    .team-emoji {{ font-size: 40px; display: block; margin-bottom: 12px; }}
    .team-name {{ font-size: 18px; font-weight: 700; display: block; color: {text}; }}
    .team-role {{ font-size: 12px; font-weight: 600; color: {accent}; text-transform: uppercase; display: block; margin-bottom: 8px; }}
    .team-desc {{ font-size: 13px; color: #888; }}
    
    /* Footer and links style formatting */
    .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; margin-top: 4px; }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Control Center ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
        
    # Theme toggles configuration
    theme_label = "☀️ Light Mode" if st.session_state.theme_mode == "Dark" else "🌙 Dark Mode"
    if st.button(theme_label):
        st.session_state.theme_mode = "Light" if st.session_state.theme_mode == "Dark" else "Dark"
        st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(f"### 📍 Navigation")
    if st.button("🏠 Home"): st.session_state.active_tab = "home"
    if st.button("📊 History"): st.session_state.active_tab = "dashboard"
    if st.button("👥 Team"): st.session_state.active_tab = "team"

# ── Navigation Router Layouts ─────────────────────────────────────────────────
if st.session_state.active_tab == "home":
    st.markdown("<h1 style='text-align: center;'>🔍 BeritaCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px; margin-top:-10px;'>Malaysian Political Bias Analyser</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # User URL Input Form Setup Elements
    url_input = st.text_input("News Article URL", placeholder="https://www.malaymail.com/news/...", value=st.session_state.url_input)

    c1, c2 = st.columns([4, 1])
    with c1:
        analyse_clicked = st.button("Analyse Article →")
    with c2:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("✕ Clear") or (st.session_state.url_input and not url_input):
            st.session_state.url_input = ""
            st.session_state.result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if analyse_clicked:
        if not url_input.strip():
            st.error("Please enter a valid news URL.")
        else:
            st.session_state.url_input = url_input
            with st.spinner("Processing article through ML Pipeline..."):
                try:
                    # Forward the URL directly to the production API endpoint
                    response = requests.post(
                        f"{BACKEND_API_URL}/analyze-url",
                        json={"url": url_input},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        api_data = response.json()
                        
                        if api_data.get("status") == "success":
                            model_outputs = api_data["model_outputs"]
                            linguistic_features = api_data["linguistic_features"]
                            
                            # Map properties from backend schema into UI elements
                            st.session_state.result = {
                                "url": url_input,
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "verdict": model_outputs["bias_analysis"]["label"],
                                "confidence": f"{model_outputs['bias_analysis']['confidence'] * 100:.1f}%",
                                "density": f"{linguistic_features['political_density_score'] * 100:.1f}%",
                                "subjectivity": f"{linguistic_features['subjectivity_score'] * 100:.1f}%",
                                "sentiment": model_outputs["sentiment_analysis"]["label"],
                                "text": api_data.get("extracted_text_snippet", "") + "..."
                            }
                            
                            # Save record to tracking history
                            st.session_state.history.insert(0, st.session_state.result)
                        else:
                            st.error("Unable to scrape or read this article URL path.")
                    elif response.status_code == 422:
                        st.error("Unable to scrape or read this article URL path.")
                    else:
                        st.error(f"Backend Error: HTTP {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the Machine Learning backend server.")
                except Exception as e:
                    st.error(f"Unexpected Pipeline Error: {str(e)}")

    # Display processing results maps if active state exists
    if st.session_state.result:
        res = st.session_state.result
        
        st.markdown(f"""
        <div class="verdict-card">
            <div class="verdict-header">Bias Verdict</div>
            <div class="verdict-title">{res['verdict']}</div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{res['confidence']}</div>
                    <div class="metric-lbl">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{res['sentiment']}</div>
                    <div class="metric-lbl">Linguistic Sentiment</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{res['density']}</div>
                    <div class="metric-lbl">Political Density</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{res['subjectivity']}</div>
                    <div class="metric-lbl">Subjectivity</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Cleaned Article Content"):
            st.write(res["text"])

elif st.session_state.active_tab == "dashboard":
    st.markdown(f"## 📊 Recent Analyses History")
    st.markdown("<hr style='margin-top:0;'>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No records evaluated yet. Submit a URL on the home tab to display tracking history.")
    else:
        for idx, item in enumerate(st.session_state.history):
            st.markdown(f"""
            <div style="background:{card_bg}; padding:16px; border-radius:8px; border:1px solid {border}; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; color:{history_lbl};">
                    <span>⏱️ {item['timestamp']}</span>
                    <strong>{item['verdict']} ({item['confidence']})</strong>
                </div>
                <div style="margin-top:6px; font-size:14px; font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    🔗 <a href="{item['url']}" target="_blank" style="color:{accent}; text-decoration:none;">{item['url']}</a>
                </div>
                <div style="margin-top:8px; display:flex; gap:16px; font-size:12px; color:{history_lbl};">
                    <span>📊 Sentiment: <b>{item['sentiment']}</b></span>
                    <span>📈 Political Density: <b>{item['density']}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.active_tab == "team":
    st.markdown(f"<h2 style='text-align:center;'>Meet the BeritaCheck Engine Team</h2>", unsafe_allow_html=True)
    st.markdown('<span class="team-desc" style="text-align:center; display:block;">The engineers, analysts, and developers behind the fine-tuned Malaysian bias tracking engine.</span>', unsafe_allow_html=True)

    members = [
        ("👨‍💻", "Afiq",           "NLP Engineer",     "Feature optimisation and keyword lexicon design."),
        ("👩‍💻", "Nia",            "Data Engineer",    "Data pipeline harvesting and corpus classification."),
        ("📊",  "Fatnin",         "Data Analyst",     "Statistical distributions and bias margin tracking."),
        ("👩‍💻", "Eireen Syafeeya","Lead Developer",   "Architecture, backend implementation, and Streamlit frontend."),
        ("🤖",  "Fatimah",        "ML Engineer",      "Transformer model training and evaluation metrics."),
        ("🎨",  "Husna",          "UI/UX & Tester",   "Interface design and semantic validation testing."),
    ]

    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    for emoji, name, role, desc in members:
        st.markdown(f"""
        <div class="team-card">
            <span class="team-emoji">{emoji}</span>
            <span class="team-name">{name}</span>
            <span class="team-role">{role}</span>
            <span class="team-desc">{desc}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div class="footer">BeritaCheck · Malaysian Political Bias Analyser</div>
""", unsafe_allow_html=True)