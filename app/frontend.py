import streamlit as st
import requests
import re
import datetime
import json
import nltk
from nltk.corpus import stopwords

# ── BACKEND ROUTING CONFIGURATION ─────────────────────────────────────────────
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

# ── Dynamic Theme Palette Configuration ──────────────────────────────────────
THEME_PALETTES = {
    "Dark": {
        "bg": "#0d0d0d", "text": "#e8e0d0", "sidebar_bg": "#0a0a0a",
        "border": "#2a2a2a", "sidebar_border": "#1e1e1e", "card_bg": "#141414",
        "accent": "#c8a84b", "accent_hover": "#d9b95c", "input_bg": "#141414",
        "text_muted": "#666", "text_light_muted": "#888", "hr": "#2a2a2a",
        "alert_bg": "#1a0f0f", "alert_border": "#5c2222", "kw_border": "#3a2a10"
    },
    "Light": {
        "bg": "#f9f8f6", "text": "#1a1a1a", "sidebar_bg": "#f0ede9",
        "border": "#dcd8d0", "sidebar_border": "#c8c2b7", "card_bg": "#ffffff",
        "accent": "#b08d2b", "accent_hover": "#94741f", "input_bg": "#ffffff",
        "text_muted": "#888888", "text_light_muted": "#555555", "hr": "#dcd8d0",
        "alert_bg": "#fdf2f2", "alert_border": "#f5c6cb", "kw_border": "#d0c090"
    }
}

# ── CSS Generation with Theme Switcher Support ────────────────────────────────
def inject_css():
    c = THEME_PALETTES.get(st.session_state.theme_mode, THEME_PALETTES["Dark"])
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&display=swap');

/* ─ Global & Reset ─ */
html, body, [data-testid="stAppViewContainer"] {{
    background: {c['bg']} !important;
    color: {c['text']} !important;
}}
[data-testid="stAppViewContainer"] > .main {{ padding-top: 1.5rem; }}
section.main > div {{ max-width: 780px; margin: auto; }}

/* ─ Typography ─ */
h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; color: {c['text']} !important; }}
p, li, div, label {{ font-family: 'Source Serif 4', Georgia, serif !important; }}
code, pre, .mono, [data-testid="stMarkdown"] pre code {{ font-family: 'IBM Plex Mono', monospace !important; }}

/* ─ Header Area Setup ─ */
[data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: none !important;
}}
[data-testid="stDecoration"] {{ display: none !important; }}

/* ─ Sidebar Structural Rules ─ */
[data-testid="stSidebar"] {{
    background: {c['sidebar_bg']} !important;
    border-right: 1px solid {c['sidebar_border']} !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding: 1.5rem 1rem !important; }}
[data-testid="stSidebar"] * {{ color: {c['text']} !important; }}

[data-testid="stSidebarCollapsedControl"] {{
    display: flex !important;
    background: {c['sidebar_bg']} !important;
    border-right: 1px solid {c['sidebar_border']} !important;
}}
[data-testid="stSidebarCollapsedControl"] button {{
    color: {c['accent']} !important;
    background: {c['sidebar_bg']} !important;
    border: none !important;
}}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="stSidebarCollapsedControl"] button svg path,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg path {{
    fill: {c['accent']} !important;
    stroke: {c['accent']} !important;
    color: {c['accent']} !important;
}}
[data-testid="stHeader"] button svg,
[data-testid="stHeader"] button svg path {{
    fill: {c['accent']} !important;
    stroke: {c['accent']} !important;
}}

/* ─ Sidebar Navigation Buttons ─ */
[data-testid="stSidebar"] [data-testid="stButton"] > button {{
    background: transparent !important;
    border: none !important;
    color: {c['text_light_muted']} !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.5rem 0.6rem !important;
    height: auto !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: color 0.15s, border-left 0.15s !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {{
    background: {c['bg']} !important;
    color: {c['accent']} !important;
    border: none !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ─ Sidebar Selectboxes ─ */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {{
    background: {c['card_bg']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: 2px !important;
    color: {c['text']} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {{ fill: {c['text_light_muted']} !important; }}

/* ─ Sidebar Logo ─ */
.sidebar-logo {{
    text-align: center;
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid {c['sidebar_border']};
    margin-bottom: 1rem;
}}
.sidebar-logo .s-logo {{
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: -1px;
    color: {c['text']};
    line-height: 1;
}}
.sidebar-logo .s-logo span {{ color: {c['accent']}; }}
.sidebar-logo .s-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.18em;
    color: {c['text_muted']};
    text-transform: uppercase;
    margin-top: 0.3rem;
    display: block;
}}

.nav-active > button {{
    color: {c['accent']} !important;
    border-left: 2px solid {c['accent']} !important;
    padding-left: calc(0.6rem - 2px) !important;
}}

/* ─ Main Header ─ */
.bc-header {{
    text-align: center;
    padding: 2rem 1rem 1.25rem;
    border-bottom: 1px solid {c['border']};
    margin-bottom: 2rem;
}}
.bc-header .logo {{
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -2px;
    color: {c['text']};
    line-height: 1;
}}
.bc-header .logo span {{ color: {c['accent']}; }}
.bc-header .tagline {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: {c['text_muted']};
    text-transform: uppercase;
    margin-top: 0.4rem;
}}

/* ─ Text Inputs ─ */
[data-testid="stTextInput"] input {{
    background: {c['input_bg']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: 2px !important;
    color: {c['text']} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.85rem !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {c['accent']} !important;
    box-shadow: 0 0 0 2px rgba(200,168,75,0.15) !important;
}}
[data-testid="stTextInput"] label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {c['text_light_muted']} !important;
}}

/* ─ Main Action Buttons ─ */
[data-testid="stButton"] > button {{
    background: {c['accent']} !important;
    color: {c['bg']} !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stButton"] > button:hover {{
    background: {c['accent_hover']} !important;
    transform: translateY(-1px) !important;
}}

/* ─ Download Buttons ─ */
[data-testid="stDownloadButton"] > button {{
    background: {c['card_bg']} !important;
    color: {c['accent']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: {c['bg']} !important;
    border-color: {c['accent']} !important;
    transform: translateY(-1px) !important;
}}

/* ─ Result Cards ─ */
.result-card {{
    background: {c['card_bg']};
    border: 1px solid {c['border']};
    border-radius: 2px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
}}
.result-card.verdict-political {{ border-left: 4px solid #c0392b; }}
.result-card.verdict-objective {{ border-left: 4px solid #27ae60; }}
.card-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {c['text_muted']};
    margin-bottom: 0.4rem;
}}
.card-value {{
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
}}
.verdict-political .card-value {{ color: #e74c3c; }}
.verdict-objective .card-value {{ color: #2ecc71; }}

/* ─ Metric Grid ─ */
.metric-row {{
    display: flex;
    gap: 1px;
    background: {c['border']};
    border-radius: 2px;
    overflow: hidden;
    margin: 1rem 0;
}}
.metric-cell {{
    flex: 1;
    background: {c['card_bg']};
    padding: 1rem 1.2rem;
    text-align: center;
}}
.metric-cell .m-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {c['text_muted']};
    margin-bottom: 0.3rem;
}}
.metric-cell .m-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: {c['accent']};
}}

/* ─ Score Bars ─ */
.score-bar-bg {{
    height: 6px;
    background: {c['border']};
    border-radius: 3px;
    overflow: hidden;
    margin: 0.4rem 0 0.8rem;
}}
.score-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.6s ease; }}
.bar-political {{ background: #c0392b; }}
.bar-objective {{ background: #27ae60; }}
.bar-subjectivity {{ background: {c['accent']}; }}
.bar-political-light {{ background: {c['accent']}; }}
.bar-density {{ background: {c['accent']}; }}

/* ─ Summary Block ─ */
.summary-block {{
    background: {c['card_bg']};
    border: 1px solid {c['border']};
    border-radius: 2px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
}}
.summary-block p {{
    font-size: 0.97rem;
    line-height: 1.75;
    color: {c['text']};
    margin: 0;
}}

/* ─ Keyword Pills ─ */
.kw-list {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }}
.kw-pill {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    background: {c['bg']};
    border: 1px solid {c['kw_border']};
    color: {c['accent']};
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
}}

/* ─ Expanders ─ */
[data-testid="stExpander"] {{
    background: {c['card_bg']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: 2px !important;
    overflow: visible !important;
    margin-bottom: 0.5rem !important;
}}
[data-testid="stExpander"] summary {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {c['text_light_muted']} !important;
    padding: 0.75rem 1rem !important;
}}
[data-testid="stExpander"] summary:hover {{
    color: {c['accent']} !important;
}}
[data-testid="stExpander"] > div > div,
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    padding: 0.25rem 1rem 1rem 1rem !important;
    background: {c['card_bg']} !important;
}}
[data-testid="stExpander"] p {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 1.0rem !important;
    line-height: 1.75 !important;
    color: {c['text']} !important;
}}

/* ─ Section Headers ─ */
.section-hdr {{
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: {c['text']};
    letter-spacing: -0.5px;
    border-bottom: 1px solid {c['border']};
    padding-bottom: 0.5rem;
    margin-bottom: 1.25rem;
    display: block;
}}

/* ─ History Cards ─ */
.hist-card {{
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    background: {c['card_bg']};
    border: 1px solid {c['border']};
    border-radius: 2px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}}
.hist-dot-pol {{ width: 8px; height: 8px; min-width: 8px; border-radius: 50%; background: #c0392b; margin-top: 6px; flex-shrink: 0; }}
.hist-dot-obj {{ width: 8px; height: 8px; min-width: 8px; border-radius: 50%; background: #27ae60; margin-top: 6px; flex-shrink: 0; }}
.hist-title {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.9rem;
    font-weight: 400;
    color: {c['text']};
    display: block;
    margin-bottom: 0.2rem;
    line-height: 1.4;
    word-break: break-word;
}}
.hist-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: {c['text_light_muted']};
    display: block;
    line-height: 1.6;
    letter-spacing: 0.06em;
}}

/* ─ Stats Row ─ */
.stats-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: {c['border']};
    border-radius: 2px;
    overflow: hidden;
    margin: 1.25rem 0;
}}
.stat-box {{
    background: {c['card_bg']};
    padding: 1rem 0.75rem;
    text-align: center;
}}
.stat-n {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: {c['accent']};
    display: block;
    line-height: 1;
}}
.stat-l {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {c['text_muted']};
    margin-top: 0.3rem;
    display: block;
}}

/* ─ Team Grid ─ */
.team-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: {c['border']}; border-radius: 2px; overflow: hidden; }}
.team-card {{
    background: {c['card_bg']};
    padding: 1.25rem 1rem;
    text-align: center;
}}
.team-emoji {{ font-size: 1.8rem; display: block; margin-bottom: 0.5rem; line-height: 1; }}
.team-name {{
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: {c['text']};
    display: block;
    margin-bottom: 0.15rem;
}}
.team-role {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {c['accent']};
    display: block;
    margin-bottom: 0.4rem;
}}
.team-desc {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.78rem;
    color: {c['text_muted']};
    line-height: 1.5;
    display: block;
}}

/* ─ About ─ */
.about-body {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: {c['text']};
    background: {c['card_bg']};
    border: 1px solid {c['border']};
    border-radius: 2px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
    display: block;
}}
.sub-hdr {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {c['text_light_muted']};
    margin: 1.25rem 0 0.6rem 0;
    display: block;
    border-bottom: 1px solid {c['border']};
    padding-bottom: 0.4rem;
}}
.step-row {{
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 0.7rem 0;
    border-bottom: 1px solid {c['border']};
}}
.step-row:last-child {{ border-bottom: none; }}
.step-num {{
    width: 22px; height: 22px; min-width: 22px;
    border-radius: 50%;
    background: {c['accent']};
    color: {c['bg']};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}}
.step-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: {c['text']};
    display: block; margin-bottom: 0.15rem;
}}
.step-desc {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.97rem;
    color: {c['text_muted']};
    line-height: 1.6;
    display: block;
}}

/* ─ Alerts ─ */
[data-testid="stAlert"] {{
    background: {c['alert_bg']} !important;
    border: 1px solid {c['alert_border']} !important;
    border-radius: 2px !important;
}}
[data-testid="stAlert"] * {{ color: {c['text']} !important; }}

/* ─ Progress Bars ─ */
[data-testid="stProgress"] > div > div {{ background: {c['accent']} !important; }}

/* ─ Dividers ─ */
hr {{ border-color: {c['border']} !important; margin: 1.5rem 0 !important; }}

/* ─ Footer ─ */
.footer {{
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {c['text_muted']};
    padding: 0.5rem 0 2rem;
}}
</style>
""", unsafe_allow_html=True)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="s-logo">Berita<span>Check</span></div>
        <span class="s-tag">News Bias Analyser · 🇲🇾</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<label style='font-family:\"IBM Plex Mono\",monospace; font-size:0.65rem; color:inherit; text-transform:uppercase; display:block;'>🎨 Interface Theme</label>", unsafe_allow_html=True)
    theme_opts = ["Dark", "Light"]
    theme_icons = {"Dark": "🌙  Dark Mode", "Light": "☀️  Light Mode"}
    sel_theme = st.selectbox(
        "theme", theme_opts,
        index=theme_opts.index(st.session_state.theme_mode),
        format_func=lambda x: theme_icons[x],
        key="theme_sel", label_visibility="collapsed"
    )
    if sel_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = sel_theme
        st.rerun()

    st.markdown("<hr style='margin:0.75rem 0;'>", unsafe_allow_html=True)

    nav_items = [
        ("home",      "🏠  Home"),
        ("history",   "📋  History"),
        ("faq",       "❓  FAQ"),
        ("about",     "ℹ️  About"),
        ("team",      "👥  Team"),
    ]
    for key, label in nav_items:
        is_active = st.session_state.active_tab == key
        if is_active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_tab = key
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.75rem 0;'>", unsafe_allow_html=True)

    hist_len = len(st.session_state.history)
    pol_count = sum(1 for h in st.session_state.history if h.get("is_political"))
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:inherit; opacity:0.7; letter-spacing:0.1em; text-transform:uppercase; line-height:1.8;">
        <div>📊 {hist_len} articles checked</div>
        <div style="color:#c0392b;">⚠ {pol_count} political-framed</div>
        <div style="color:#27ae60;">✓ {hist_len - pol_count} objective</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: HOME
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.active_tab == "home":

    st.markdown("""
    <div class="bc-header">
        <div class="logo">Berita<span>Check</span></div>
        <div class="tagline">Malaysian Political Bias Analyser</div>
    </div>
    """, unsafe_allow_html=True)

    pol_count = sum(1 for h in st.session_state.history if h.get("is_political"))
    obj_count = len(st.session_state.history) - pol_count
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box">
            <span class="stat-n">{len(st.session_state.history)}</span>
            <span class="stat-l">Articles Checked</span>
        </div>
        <div class="stat-box">
            <span class="stat-n" style="color:#c0392b;">{pol_count}</span>
            <span class="stat-l">Political-Framed</span>
        </div>
        <div class="stat-box">
            <span class="stat-n" style="color:#27ae60;">{obj_count}</span>
            <span class="stat-l">Objective</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    url_val = st.text_input(
        "News Article URL",
        value=st.session_state.url_input,
        placeholder="https://www.malaymail.com/news/...",
        key="url_field",
    )
    st.session_state.url_input = url_val

    analyse_btn = st.button("Analyse Article →", key="analyse_btn")

    if analyse_btn:
        url = st.session_state.url_input.strip()
        if not url:
            st.error("Please enter a valid news article URL.")
        elif not url.startswith("http"):
            st.error("URL must start with http:// or https://")
        else:
            with st.spinner("Processing article through ML pipeline…"):
                try:
                    response = requests.post(
                        f"{BACKEND_API_URL}/analyze-url",
                        json={"url": url},
                        timeout=30
                    )

                    if response.status_code == 200:
                        api_data = response.json()
                        mo = api_data.get("model_outputs", {})

                        # 1. Capture the original Machine Learning Model prediction
                        ml_verdict_label = mo.get("bias_analysis", {}).get("label", "Non-Political/Objective")
                        is_ml_political = "political" in ml_verdict_label.lower() or "bias" in ml_verdict_label.lower()

                        # 2. Extract the Lexicon Density score from linguistic_features
                        features = api_data.get("linguistic_features", {})
                        density_score = features.get("political_density_score", 0.0)
                        density_percentage = density_score * 100
                        is_density_political = density_percentage >= 3.5

                        # 3. Formulate the Hybrid Verdict
                        # Both signals must agree to flag as political, OR density must be
                        # very high (>= 7%) on its own as a strong standalone signal.
                        # This prevents a noisy/undertrained BERT model from flagging every
                        # article, and stops borderline keyword density from overriding a
                        # clean ML result.
                        is_high_density = density_percentage >= 7.0
                        ml_confidence = mo.get("bias_analysis", {}).get("confidence", 0.0)
                        is_high_conf_ml = is_ml_political and ml_confidence >= 0.80

                        if is_density_political and is_ml_political:
                            verdict_label = "Political-Framed (High Density & Context Verified)"
                            is_political = True
                        elif is_high_density and not is_ml_political:
                            verdict_label = "Political-Framed (Very High Keyword Density)"
                            is_political = True
                        elif is_high_conf_ml and not is_density_political:
                            verdict_label = "Political-Framed (High-Confidence Semantic Bias)"
                            is_political = True
                        elif is_ml_political and not is_density_political:
                            verdict_label = "Borderline — Possible Political Framing (Low Density)"
                            is_political = False  # Soft signal only; don't hard-flag
                        elif is_density_political and not is_ml_political:
                            verdict_label = "Borderline — Elevated Keyword Density (Unconfirmed by Model)"
                            is_political = False  # Soft signal only; don't hard-flag
                        else:
                            verdict_label = "Non-Political / Objective"
                            is_political = False

                        # 4. Render the Core UI Elements
                        if is_political:
                            st.error(f"⚠️ {verdict_label}")
                        elif "Borderline" in verdict_label:
                            st.warning(f"⚠️ {verdict_label}")
                        else:
                            st.success(f"✅ {verdict_label}")

                        # 5. Render individual diagnostics directly beneath it
                        st.markdown("### 🛠️ Hybrid Engine Diagnostics")
                        col_ml, col_density = st.columns(2)

                        with col_ml:
                            if is_ml_political:
                                st.markdown("🔴 **BERT Context Engine:** Detected Political Framing")
                            else:
                                st.markdown("🟢 **BERT Context Engine:** Clean / Objective")

                        with col_density:
                            if is_density_political:
                                st.markdown(f"🔴 **Lexicon Keyword Engine:** {density_percentage:.1f}% (Crossed 3.5% Threshold)")
                            else:
                                st.markdown(f"🟢 **Lexicon Keyword Engine:** {density_percentage:.1f}% (Below 3.5% Threshold)")

                        # 6. Safely populate results context using mapped variables
                        st.session_state.result = {
                            "url":          url,
                            "title":        api_data.get("title", url),
                            "timestamp":    datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"),
                            "verdict":      verdict_label,
                            "is_political": is_political,
                            "confidence":   round(mo.get("bias_analysis", {}).get("confidence", 0.0) * 100, 1),
                            "density":      round(density_percentage, 1),
                            "subjectivity": round(features.get("subjectivity_score", 0.0) * 100, 1),
                            "sentiment":    mo.get("sentiment_analysis", {}).get("label", "Neutral"),
                            "text_snippet": api_data.get("extracted_text_snippet", ""),
                        }

                        if not any(h["url"] == url for h in st.session_state.history):
                            st.session_state.history.insert(0, st.session_state.result)
                        st.rerun()

                    elif response.status_code == 422:
                        st.error("Unable to scrape or read this article URL.")
                    else:
                        st.error(f"Backend error: HTTP {response.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Please check the service is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. The backend may be slow or unavailable.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ── Results rendering ────────────────────────────────────────────────────
    if st.session_state.result:
        r = st.session_state.result
        is_pol        = r.get("is_political", False)
        verdict_class = "verdict-political" if is_pol else "verdict-objective"
        verdict_icon  = "⚠" if is_pol else "✓"

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card {verdict_class}">
            <div class="card-label">Bias Verdict</div>
            <div class="card-value">{verdict_icon} {r['verdict']}</div>
            <div style="margin-top:0.5rem; font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:inherit; opacity:0.6; letter-spacing:0.1em;">
                Confidence: {r['confidence']}% &nbsp;·&nbsp; Threshold: 3.5% political keyword density
            </div>
        </div>
        """, unsafe_allow_html=True)

        display_title = r.get("title", r["url"])
        st.markdown(f"""
        <div style="margin:0.5rem 0 1.2rem;">
            <div class="card-label">Article</div>
            <div style="font-family:'Source Serif 4',serif; font-size:1.05rem; line-height:1.4;">
                {display_title[:140]}{'…' if len(display_title) > 140 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        sent_icons = {"Positive": "↑", "Negative": "↓", "Neutral": "—"}
        si = sent_icons.get(r["sentiment"], "")

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-cell">
                <div class="m-label">Political Density</div>
                <div class="m-value">{r['density']}%</div>
            </div>
            <div class="metric-cell">
                <div class="m-label">Subjectivity</div>
                <div class="m-value">{r['subjectivity']}%</div>
            </div>
            <div class="metric-cell">
                <div class="m-label">Sentiment</div>
                <div class="m-value">{si} {r['sentiment']}</div>
            </div>
            <div class="metric-cell">
                <div class="m-label">Confidence</div>
                <div class="m-value">{r['confidence']}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c = THEME_PALETTES.get(st.session_state.theme_mode, THEME_PALETTES["Dark"])
        density_pct  = min(100, r["density"] * 10)
        density_colour  = "#c0392b" if is_pol else c["accent"]
        subj_colour     = c["accent"]
        st.markdown(f"""
        <div style="margin:0.5rem 0 1rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; opacity:0.7; margin-bottom:0.2rem;">
                Political Density — {r['density']}%
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{density_pct}%; background:{density_colour};"></div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; opacity:0.7; margin-bottom:0.2rem;">
                Subjectivity — {r['subjectivity']}%
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{r['subjectivity']}%; background:{subj_colour};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="card-label" style="margin-bottom:0.6rem;">Download Report</div>', unsafe_allow_html=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        safe_title = re.sub(r"[^\w\s-]", "", r.get("title", "article")[:40]).strip().replace(" ", "_")

        report_lines = [
            "=" * 60,
            "  BERITACHECK — ANALYSIS REPORT",
            "  Malaysian Political Bias Analyser",
            "=" * 60,
            "",
            f"  Article : {r.get('title', r['url'])}",
            f"  URL     : {r['url']}",
            f"  Analysed: {r['timestamp']}",
            "",
            "-" * 60,
            "  VERDICT",
            "-" * 60,
            f"  {r['verdict']}",
            f"  Confidence : {r['confidence']}%",
            "  Threshold  : 3.5% political keyword density",
            "",
            "-" * 60,
            "  METRICS",
            "-" * 60,
            f"  Political Density : {r['density']}%",
            f"  Subjectivity      : {r['subjectivity']}%",
            f"  Sentiment         : {r['sentiment']}",
            "",
            "=" * 60,
            "  BeritaCheck · NLP-Powered · 🇲🇾",
            "=" * 60,
        ]
        txt_report = "\n".join(report_lines)

        json_report = json.dumps({
            "beritacheck_report": {
                "generated": r["timestamp"],
                "article": {"title": r.get("title", r["url"]), "url": r["url"]},
                "verdict": {"label": r["verdict"], "confidence_pct": r["confidence"]},
                "metrics": {
                    "political_density_pct": r["density"],
                    "subjectivity_pct": r["subjectivity"],
                    "sentiment": r["sentiment"],
                },
            }
        }, indent=2, ensure_ascii=False)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="↓ Download Report (.txt)",
                data=txt_report,
                file_name=f"BeritaCheck_{safe_title}_{ts}.txt",
                mime="text/plain",
                key="dl_txt",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="↓ Download Report (.json)",
                data=json_report,
                file_name=f"BeritaCheck_{safe_title}_{ts}.json",
                mime="application/json",
                key="dl_json",
                use_container_width=True,
            )

        if r.get("text_snippet"):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Extracted Article Snippet"):
                st.markdown(f'<p>{r["text_snippet"]}…</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("How does BeritaCheck work?"):
            st.markdown("""
**BeritaCheck** uses the same NLP pipeline developed in the research project:

1. **Web Scraping** — Article text is extracted from the provided URL, stripping ads, navigation, and boilerplate.

2. **Text Polishing** — Malaysian media datelines, subscription notices, and noise patterns are removed.

3. **Political Density Score** — The processed text is checked against a curated lexicon of ~50 Malaysian political keywords. Density = matched keywords ÷ total words. A score above **3.5%** triggers a "Political-Framed" verdict.

4. **Subjectivity Score** — Measures how opinion-based vs. factual the language is (0% = purely objective, 100% = purely subjective).

5. **Sentiment Analysis** — Determines whether the overall tone is Positive, Negative, or Neutral.
            """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "history":
    st.markdown('<span class="section-hdr">Analysis History</span>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analyses yet. Go to Home to analyse articles.")
    else:
        _, col_clr = st.columns([3, 1.2])
        with col_clr:
            if st.button("✕ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.result = None
                st.rerun()

        for h in st.session_state.history:
            is_pol  = h.get("is_political", False)
            dot     = "hist-dot-pol" if is_pol else "hist-dot-obj"
            icon    = "⚠" if is_pol else "✓"
            verdict = h.get("verdict", "Unknown")
            title_display = h.get("title", h["url"])
            title_trunc   = title_display[:90] + ("…" if len(title_display) > 90 else "")
            url_trunc     = h["url"][:60] + "…" if len(h["url"]) > 60 else h["url"]

            st.markdown(f"""
            <div class="hist-card">
                <div class="{dot}"></div>
                <div style="flex:1; min-width:0;">
                    <span class="hist-title">{title_trunc}</span>
                    <span class="hist-meta">{verdict} · Density: {h['density']}% · Sentiment: {h['sentiment']} · Analysed: {h['timestamp']}</span>
                    <span class="hist-meta"><a href="{h['url']}" target="_blank" style="color:inherit; text-decoration:underline;">{url_trunc}</a></span>
                </div>
                <span style="color:{'#c0392b' if is_pol else '#27ae60'}; margin-left:auto; font-size:1rem;">{icon}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: FAQ
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "faq":
    st.markdown('<span class="section-hdr">Frequently Asked Questions</span>', unsafe_allow_html=True)

    faqs = [
        ("What is BeritaCheck?",
         "BeritaCheck is an NLP-powered tool that analyses Malaysian news articles to detect signs of political bias or framing using keyword density analysis, subjectivity scoring, and sentiment analysis."),
        ("How accurate is it?",
         "BeritaCheck uses rule-based NLP developed from a dataset of Malaysian news outlets including Bernama, Malay Mail, and Free Malaysia Today. Results should be considered alongside your own critical reading."),
        ("What does 'Political-Framed' mean?",
         "An article is tagged Political-Framed when more than 3.5% of its processed vocabulary consists of political keywords common in Malaysian reporting."),
        ("What does the Confidence score mean?",
         "Confidence reflects how far the article's political density sits from the 3.5% threshold. A higher score means the signal is clearer in either direction."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(f'<p>{a}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "about":
    st.markdown('<span class="section-hdr">About BeritaCheck</span>', unsafe_allow_html=True)

    st.markdown('<span class="about-body"><strong>BeritaCheck</strong> is a Malaysian news political bias detection system built using Natural Language Processing (NLP), developed as part of an academic research project.</span>', unsafe_allow_html=True)

    st.markdown('<span class="sub-hdr">The NLP Pipeline</span>', unsafe_allow_html=True)
    pipeline = [
        ("Data Harvesting",      "Raw articles were collected from 7 Malaysian news outlets using a custom web scraping pipeline."),
        ("Text Preprocessing",   "Articles were cleaned — removing datelines, ad copy, subscription banners, and duplicate content — then normalised for NLP input."),
        ("Feature Engineering",  "Three key features were engineered: Political Keyword Density, TextBlob Subjectivity Score, and TF-IDF Framing Distance from topic consensus vectors."),
        ("Bias Labelling",       "Articles with >3.5% political keyword density were labelled Political-Framed; others were labelled Non-Political/Objective."),
        ("Model Training",       "A fine-tuned BERT model was trained for bias detection and a RoBERTa model for sentiment classification. The web app uses the rule-based pipeline for instant results."),
    ]
    for i, (title, desc) in enumerate(pipeline, 1):
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{i}</div>
            <div style="flex:1;">
                <span class="step-title">{title}</span>
                <span class="step-desc">{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<span class="sub-hdr">News Outlets Studied</span>', unsafe_allow_html=True)
    outlets = ["Bernama", "Malay Mail", "The Star", "The Edge Malaysia",
               "Free Malaysia Today", "The Rakyat Post", "The Straits Times"]
    pills = "".join([f'<span class="kw-pill">{o}</span>' for o in outlets])
    st.markdown(f'<div class="kw-list" style="margin-bottom:1.5rem;">{pills}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: TEAM
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "team":
    st.markdown('<span class="section-hdr">Our Team</span>', unsafe_allow_html=True)
    st.markdown('<span class="about-body" style="text-align:center; display:block;">BeritaCheck was developed by a team of students passionate about media literacy, data science, and the Malaysian information landscape. 🇲🇾</span>', unsafe_allow_html=True)

    members = [
        ("👨‍💻", "Afiq",            "NLP Engineer",    "Feature optimisation and keyword lexicon design."),
        ("👩‍💻", "Nia",             "Data Engineer",   "Data pipeline harvesting and corpus classification."),
        ("📊",  "Fatnin",          "Data Analyst",    "Statistical distributions and bias margin tracking."),
        ("👩‍💻", "Eireen Syafeeya", "Lead Developer",  "Architecture, backend implementation, and Streamlit frontend."),
        ("🤖",  "Fatimah",         "ML Engineer",     "Transformer model training and evaluation metrics."),
        ("🎨",  "Husna",           "UI/UX & Tester",  "Interface design and semantic validation testing."),
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
st.markdown("""
<div class="footer">BeritaCheck · Malaysian Political Bias Analyser · NLP-Powered · 🇲🇾</div>
""", unsafe_allow_html=True)