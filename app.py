"""
South Sudan Flood Early Warning System
Professional Streamlit Dashboard
Author: Chol Monykuch
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── MUST be first ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SSD Flood Early Warning",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="auto",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
PRIMARY   = "#0D1F3C"
ACCENT    = "#2563EB"
ACCENT2   = "#0EA5E9"
SUCCESS   = "#059669"
WARNING   = "#D97706"
DANGER    = "#DC2626"
CRITICAL  = "#7C3AED"
MUTED     = "#64748B"
BG_PAGE   = "#F1F5F9"
BG_CARD   = "#FFFFFF"
TEXT_DARK = "#0F172A"
TEXT_MUT  = "#64748B"
BORDER    = "#E2E8F0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
.stApp {{ background-color: {BG_PAGE}; }}
.block-container {{ padding: 1.5rem 2.5rem 4rem !important; max-width: 1440px; }}

/* ── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {{
    background: #0D1F3C;
    padding-top: 0;
}}

/* ── Sidebar text colours ─────────────────────────── */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {{ color: #CBD5E1 !important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.1) !important; margin: 0.7rem 0 !important; }}
section[data-testid="stSidebar"] .stRadio > label {{ color: #CBD5E1 !important; }}

/* ── Selectbox ── */
section[data-testid="stSidebar"] .stSelectbox label {{ color: #94A3B8 !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:.06em; }}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stSelectbox * {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] .stSelectbox > label {{ color: #94A3B8 !important; }}
section[data-testid="stSidebar"] .stSelectbox svg {{ fill: #64A8D8 !important; }}

/* ── Slider ── */
section[data-testid="stSidebar"] .stSlider > label {{
    color: #94A3B8 !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: .06em;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {{
    background: rgba(255,255,255,0.12) !important; height: 4px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {{
    background: #2563EB !important;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background: #FFFFFF !important;
    border: 2px solid #38BDF8 !important;
    box-shadow: 0 2px 8px rgba(56,189,248,0.4) !important;
    width: 18px !important; height: 18px !important;
    cursor: grab !important; top: -7px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:active {{
    cursor: grabbing !important;
    box-shadow: 0 0 0 5px rgba(56,189,248,0.25) !important;
}}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
    color: #475569 !important; font-size: 0.68rem !important;
}}

/* ── Tab bar ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {BG_CARD}; border-radius: 12px;
    border: 1px solid {BORDER};
    padding: 5px; gap: 3px; margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; padding: 0.5rem 1.5rem;
    font-weight: 500; font-size: 0.87rem; color: {MUTED};
    border: none !important; outline: none !important;
    transition: all 0.15s;
}}
.stTabs [aria-selected="true"] {{
    background: {PRIMARY} !important;
    color: white !important; font-weight: 600;
    box-shadow: 0 2px 8px rgba(13,31,60,.25) !important;
}}

/* ── Cards ───────────────────────────────────────── */
.card {{
    background: {BG_CARD}; border-radius: 14px;
    border: 1px solid {BORDER};
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.06);
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    transition: box-shadow 0.2s;
}}
.card:hover {{
    box-shadow: 0 4px 6px rgba(0,0,0,.05), 0 10px 30px rgba(0,0,0,.09);
}}
.card-title {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: {MUTED};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.6rem; margin-bottom: 1rem;
}}

/* ── KPI strip ───────────────────────────────────── */
.kpi {{
    background: {BG_CARD}; border-radius: 14px;
    border: 1px solid {BORDER};
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.06);
    padding: 1.1rem 1.3rem; text-align: center;
    position: relative; overflow: hidden;
}}
.kpi::before {{
    content: ''; position: absolute; top:0; left:0; right:0; height:3px;
    background: var(--kpi-color, {ACCENT});
}}
.kpi-val {{ font-size: 2rem; font-weight: 800; color: {TEXT_DARK}; line-height:1.1; }}
.kpi-lbl {{ font-size: 0.74rem; color: {TEXT_MUT}; margin-top: 5px; font-weight: 600;
            text-transform:uppercase; letter-spacing:.05em; }}
.kpi-sub {{ font-size: 0.71rem; color: {MUTED}; margin-top: 3px; }}

/* ── Verdict banner ──────────────────────────────── */
.verdict-wrap {{ border-radius: 12px; padding: 1.6rem 1.8rem; text-align:center; position:relative; overflow:hidden; }}
.verdict-low      {{ background:#ECFDF5; border:1.5px solid #059669; }}
.verdict-moderate {{ background:#FFFBEB; border:1.5px solid #D97706; }}
.verdict-high     {{ background:#FFF1F2; border:1.5px solid #DC2626; }}
.verdict-critical {{ background:#F5F3FF; border:1.5px solid #7C3AED; }}
.v-label {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:.1em;
            font-weight:700; opacity:.65; margin-bottom:8px; }}
.v-prob  {{ font-size:3.4rem; font-weight:800; line-height:1; letter-spacing:-.02em; }}
.v-risk  {{ font-size:1.05rem; font-weight:700; margin-top:8px; }}
.v-dec   {{ font-size:0.76rem; margin-top:10px; opacity:.6; }}

/* ── Alert / info boxes ──────────────────────────── */
.abox {{
    border-radius: 10px; padding: 0.85rem 1.1rem;
    font-size: 0.85rem; line-height: 1.6; margin: 0.4rem 0;
}}
.abox-info     {{ background:#EFF6FF; border-left:4px solid {ACCENT}; color:#1E3A8A; }}
.abox-success  {{ background:#ECFDF5; border-left:4px solid {SUCCESS}; color:#064E3B; }}
.abox-warn     {{ background:#FFFBEB; border-left:4px solid {WARNING}; color:#78350F; }}
.abox-danger   {{ background:#FFF1F2; border-left:4px solid {DANGER}; color:#7F1D1D; }}
.abox-critical {{ background:#F5F3FF; border-left:4px solid {CRITICAL}; color:#3B0764; }}

/* ── Data table ──────────────────────────────────── */
.stDataFrame {{ border-radius: 10px; overflow:hidden; border: 1px solid {BORDER} !important; }}
.stDataFrame thead tr th {{
    background: #F8FAFC !important; font-weight: 700 !important;
    font-size: 0.76rem !important; text-transform: uppercase;
    letter-spacing: .05em; color: {MUTED} !important;
}}

/* ── Sidebar section label ───────────────────────── */
.sb-section {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: .12em;
    color: #475569 !important; font-weight: 700;
    margin: 1.1rem 0 0.45rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,.08);
}}

/* ── Page header ─────────────────────────────────── */
.page-header {{
    background: {PRIMARY};
    border-radius: 16px; padding: 1.8rem 2.2rem 1.5rem;
    margin-bottom: 1.4rem; color: white;
    box-shadow: 0 4px 24px rgba(13,31,60,.3);
}}
.ph-inner {{
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 1rem;
}}
.ph-left  {{ flex: 1 1 auto; min-width: 0; }}
.ph-right {{ flex: 0 0 auto; text-align: right; min-width: 130px; }}
.ph-badges {{ margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.3rem; }}
.ph-eyebrow {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: .1em;
    color: #93C5FD; font-weight: 700; margin-bottom: 6px;
    white-space: nowrap; overflow: visible;
}}
.page-header h1 {{ margin:0; font-size:1.6rem; font-weight:800; color:white;
                   letter-spacing:-.02em; }}
.page-header p  {{ margin:0.3rem 0 0; font-size:0.84rem; color:#93C5FD; }}
.ph-badge {{
    display:inline-flex; align-items:center;
    background:rgba(255,255,255,.12); backdrop-filter:blur(4px);
    border:1px solid rgba(255,255,255,.2); color:white;
    border-radius:20px; padding:3px 12px; font-size:0.72rem;
    font-weight:600; letter-spacing:.02em;
}}

/* ── Model result cards ──────────────────────────── */
.model-card {{
    border-radius: 12px; padding: 1.1rem 1.2rem;
    border: 1.5px solid {BORDER};
    background: {BG_CARD};
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 14px rgba(0,0,0,.06);
    height: 100%;
}}
.model-card-best {{
    border-color: {SUCCESS};
    box-shadow: 0 0 0 3px rgba(5,150,105,.1), 0 4px 20px rgba(5,150,105,.15);
}}
.model-auc {{
    font-size: 2.1rem; font-weight: 800; line-height: 1.1;
    letter-spacing: -.02em;
}}
.model-metric-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding: 3px 0; border-bottom: 1px solid {BORDER}; font-size: 0.8rem;
}}
.model-metric-row:last-child {{ border-bottom: none; }}

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer {{ visibility: hidden; }}

/* ══════════════════════════════════════════════════
   RESPONSIVE — tablet  (≤ 1024 px)
══════════════════════════════════════════════════ */
@media (max-width: 1024px) {{
    .block-container {{ padding: 1.2rem 1.5rem 3rem !important; }}
    .page-header {{ padding: 1.4rem 1.5rem 1.2rem; }}
    .page-header h1 {{ font-size: 1.3rem; }}
    .kpi-val {{ font-size: 1.65rem; }}
    .v-prob  {{ font-size: 2.6rem; }}
    .model-auc {{ font-size: 1.7rem; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.45rem 1rem; font-size: 0.82rem; }}
}}

/* ══════════════════════════════════════════════════
   RESPONSIVE — mobile  (≤ 768 px)
══════════════════════════════════════════════════ */
@media (max-width: 768px) {{
    /* Container — enough top padding to clear Streamlit's toolbar */
    .block-container {{ padding: 4rem 0.8rem 3rem !important; }}

    /* Page header — stack vertically */
    .page-header {{
        padding: 1rem 1.1rem 0.9rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }}
    .ph-inner  {{ flex-direction: column; gap: 0.75rem; }}
    .ph-right  {{
        display: flex; flex-direction: row; align-items: center;
        text-align: left; min-width: unset; gap: 1rem;
        background: rgba(255,255,255,.07); border-radius: 10px;
        padding: 0.6rem 1rem; width: 100%;
    }}
    .ph-right > div:first-child {{ flex: 1; }}
    .ph-left   {{ width: 100%; }}
    .ph-badges {{ gap: 0.25rem; margin-top: 0.6rem; }}
    .ph-eyebrow {{
        font-size: 0.78rem; letter-spacing: .04em;
        color: white; white-space: normal;
    }}
    .page-header h1 {{ font-size: 1.05rem; letter-spacing: -.01em; }}
    .page-header p  {{ font-size: 0.74rem; }}
    .ph-badge {{ font-size: 0.65rem; padding: 2px 8px; }}
    .ph-right div[style*="2.6rem"] {{ font-size: 2rem !important; }}

    /* Tabs — horizontal scroll, no wrap */
    .stTabs [data-baseweb="tab-list"] {{
        padding: 3px; gap: 2px; overflow-x: auto;
        flex-wrap: nowrap; border-radius: 10px;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }}
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.38rem 0.7rem; font-size: 0.74rem;
        white-space: nowrap; flex-shrink: 0;
    }}

    /* KPI cards — 2×2 grid on mobile */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        max-width: calc(50% - 0.5rem) !important;
        width: calc(50% - 0.5rem) !important;
    }}

    /* Cards */
    .card {{ padding: 0.9rem 1rem; border-radius: 10px; }}
    .card-title {{ font-size: 0.66rem; }}

    /* KPI strip */
    .kpi {{ padding: 0.8rem 0.75rem; border-radius: 10px; }}
    .kpi-val {{ font-size: 1.35rem; }}
    .kpi-lbl {{ font-size: 0.65rem; }}
    .kpi-sub {{ font-size: 0.62rem; }}

    /* Verdict */
    .verdict-wrap {{ padding: 1.2rem 1rem; border-radius: 10px; }}
    .v-prob {{ font-size: 2.1rem; }}
    .v-risk {{ font-size: 0.9rem; }}
    .v-label {{ font-size: 0.64rem; }}
    .v-dec  {{ font-size: 0.69rem; }}

    /* Alert boxes */
    .abox {{ font-size: 0.79rem; padding: 0.65rem 0.85rem; }}

    /* Model cards */
    .model-card {{ padding: 0.85rem 0.9rem; border-radius: 10px; }}
    .model-auc  {{ font-size: 1.45rem; }}
    .model-metric-row {{ font-size: 0.73rem; }}

    /* Data tables — horizontal scroll */
    .stDataFrame {{ overflow-x: auto !important; }}
    .stDataFrame table {{ min-width: 480px; }}
}}

/* ══════════════════════════════════════════════════
   RESPONSIVE — small mobile  (≤ 480 px)
══════════════════════════════════════════════════ */
@media (max-width: 480px) {{
    .block-container {{ padding: 4rem 0.5rem 3rem !important; }}
    .page-header h1 {{ font-size: 0.98rem; }}
    .page-header {{ padding: 0.85rem 0.9rem 0.8rem; }}
    .kpi-val {{ font-size: 1.25rem; }}
    .v-prob  {{ font-size: 1.9rem; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.35rem 0.6rem; font-size: 0.72rem; }}
    .model-auc {{ font-size: 1.3rem; }}
    .card {{ padding: 0.8rem 0.75rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COUNTY COORDINATES  (all 79 verified)
# ══════════════════════════════════════════════════════════════════════════════
# Coordinates verified against South Sudan administrative geography.
# Where counties are geographically adjacent, dots may overlap at national zoom —
# hover over any dot to see its name and data.
COUNTY_COORDS = {
    # ── Upper Nile State ──────────────────────────────────────────────────────
    "Renk":           (11.79, 32.79),
    "Melut":          (10.45, 32.18),
    "Baliet":         (10.05, 32.05),
    "Manyo":          (10.38, 31.70),
    "Fashoda":        (9.88,  31.70),
    "Panyikang":      (9.68,  31.05),
    "Malakal":        (9.53,  31.66),
    "Longochuk":      (10.18, 34.00),
    "Maban":          (10.12, 33.45),
    "Maiwut":         (9.48,  33.82),
    "Ulang":          (9.82,  33.45),
    "Luakpiny/Nasir": (8.58,  33.08),
    # ── Jonglei State ─────────────────────────────────────────────────────────
    "Fangak":         (9.08,  30.88),
    "Ayod":           (8.65,  31.42),
    "Nyirol":         (8.40,  31.08),
    "Canal/Pigi":     (7.42,  31.02),
    "Duk":            (7.58,  31.75),
    "Twic East":      (7.60,  32.38),
    "Bor South":      (6.22,  31.56),
    "Awerial":        (6.72,  31.48),
    "Uror":           (8.20,  33.18),
    "Akobo":          (7.80,  33.00),
    "Pibor":          (6.82,  33.12),
    "Pochalla":       (7.38,  34.10),
    # ── Unity State ───────────────────────────────────────────────────────────
    "Abiemnhom":      (9.78,  28.95),
    "Rubkona":        (9.12,  29.78),
    "Guit":           (9.50,  30.08),
    "Koch":           (8.92,  29.42),
    "Mayom":          (9.35,  28.55),
    "Pariang":        (9.58,  29.52),
    "Leer":           (8.28,  30.12),
    "Mayendit":       (8.45,  30.68),
    "Panyijiar":      (7.85,  30.38),
    # ── Warrap State ──────────────────────────────────────────────────────────
    "Twic":           (8.78,  28.48),
    "Gogrial East":   (8.52,  28.22),
    "Gogrial West":   (8.28,  27.62),
    "Tonj North":     (7.88,  28.30),
    "Tonj East":      (7.60,  28.78),
    "Tonj South":     (7.18,  29.05),
    "Cueibet":        (7.10,  29.42),
    # ── Northern Bahr el Ghazal ───────────────────────────────────────────────
    "Aweil Centre":   (8.72,  27.38),
    "Aweil East":     (8.78,  27.98),
    "Aweil North":    (9.08,  27.20),
    "Aweil South":    (8.42,  27.48),
    "Aweil West":     (8.90,  26.82),
    # ── Western Bahr el Ghazal ────────────────────────────────────────────────
    "Wau":            (7.70,  28.00),
    "Jur River":      (8.10,  27.12),
    "Raga":           (8.48,  25.68),
    # ── Lakes State ───────────────────────────────────────────────────────────
    "Rumbek Centre":  (6.80,  29.68),
    "Rumbek East":    (6.52,  30.18),
    "Rumbek North":   (7.18,  29.95),
    "Wulu":           (6.48,  29.15),
    "Yirol East":     (6.68,  31.10),
    "Yirol West":     (6.40,  30.55),
    # ── Western Equatoria ─────────────────────────────────────────────────────
    "Tambura":        (5.60,  27.50),
    "Nagero":         (5.88,  27.70),
    "Ezo":            (4.80,  27.15),
    "Ibba":           (4.82,  29.08),
    "Nzara":          (4.68,  28.25),
    "Yambio":         (4.58,  28.42),
    "Maridi":         (4.92,  29.48),
    "Mvolo":          (5.92,  29.65),
    "Mundri East":    (5.20,  30.38),
    "Mundri West":    (5.55,  29.72),
    # ── Central Equatoria ─────────────────────────────────────────────────────
    "Juba":           (4.85,  31.60),
    "Terekeka":       (5.55,  31.78),
    "Lainya":         (4.28,  30.98),
    "Morobo":         (3.98,  30.72),
    "Yei":            (4.09,  30.68),
    "Kajo-keji":      (3.88,  31.68),
    "Magwi":          (3.85,  32.18),
    # ── Eastern Equatoria ─────────────────────────────────────────────────────
    "Torit":          (4.42,  32.58),
    "Lafon":          (5.65,  32.82),
    "Kapoeta East":   (4.78,  33.62),
    "Kapoeta North":  (5.22,  33.55),
    "Kapoeta South":  (4.18,  33.78),
    "Ikotos":         (4.32,  33.28),
    "Budi":           (4.22,  33.88),
    # ── Abyei ─────────────────────────────────────────────────────────────────
    "Abyei":          (9.60,  28.42),
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Sidebar slider helper — shows label + live value badge ────────────────────
def sb_slider(label, unit, min_v, max_v, default, step, fmt, help_text, key):
    """
    Renders a sidebar slider with a clearly visible current-value badge.
    Uses a county-scoped session-state key so value resets when county changes.
    """
    cur = st.session_state.get(key, default)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-top:10px;margin-bottom:2px">'
        f'<span style="color:#D6EAF8;font-size:0.83rem;font-weight:500">{label}</span>'
        f'<span style="background:rgba(255,255,255,.14);color:#F9E79F;font-size:0.85rem;'
        f'font-weight:700;padding:1px 9px;border-radius:5px;'
        f'border:1px solid rgba(255,255,255,.2)">{fmt.format(cur)}&thinsp;{unit}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    return st.slider(
        label, min_v, max_v, default, step,
        label_visibility="collapsed",
        help=help_text,
        key=key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open("model/best_model.pkl","rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_artifacts():
    with open("model/metadata.json")       as f: meta      = json.load(f)
    with open("model/counties.json")       as f: counties  = json.load(f)
    with open("model/feature_stats.json")  as f: fstats    = json.load(f)
    with open("model/county_defaults.json") as f: cdefaults = json.load(f)
    hist    = pd.read_csv("model/county_flood_history.csv")
    monthly = pd.read_csv("model/monthly_flood_data.csv")
    return meta, counties, fstats, cdefaults, hist, monthly

model                                                   = load_model()
meta, counties, fstats, county_defaults, hist_df, monthly_df = load_artifacts()
FEATURES  = meta["features"]
THRESHOLD = meta["threshold"]

# ── Session state for prediction (only updates on button click) ───────────────
if "pred_prob"    not in st.session_state: st.session_state.pred_prob    = None
if "pred_county"  not in st.session_state: st.session_state.pred_county  = None
if "pred_month"   not in st.session_state: st.session_state.pred_month   = None
if "pred_inputs"  not in st.session_state: st.session_state.pred_inputs  = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction(raw: dict) -> float:
    r = raw.copy()
    r["temp_range"]    = r["max_temperature_celsius"] - r["min_temperature_celsius"]
    r["wetness_index"] = (r["rainfall_mm"] * r["soil_moisture_mm"]) / 1000.0
    r["rain_wetland"]  = r["rainfall_mm"] * r["wetland_fraction"]
    r["month_sin"]     = np.sin(2 * np.pi * r["month"] / 12)
    r["month_cos"]     = np.cos(2 * np.pi * r["month"] / 12)
    X = np.array([[r[f] for f in FEATURES]])
    return float(model.predict_proba(X)[0, 1])


def risk_tier(p: float):
    """Returns (label, hex_color, css_class, alert_class, icon, action_text)"""
    if p < 0.25:
        return ("Low Risk",      "#1E8449", "verdict-low",
                "abox-success",  "✓",
                "No immediate action required. Maintain routine monitoring protocols.")
    elif p < 0.50:
        return ("Moderate Risk", "#B7950B", "verdict-moderate",
                "abox-warn",     "⚠",
                "Alert county officials. Pre-position emergency supplies and review evacuation routes.")
    elif p < 0.75:
        return ("High Risk",     "#922B21", "verdict-high",
                "abox-danger",   "!",
                "Issue public warning. Activate emergency response teams and prepare for evacuations.")
    else:
        return ("Critical Risk", "#6C3483", "verdict-critical",
                "abox-critical", "!!",
                "IMMEDIATE ACTION. Initiate evacuations. Notify national emergency management.")


def risk_color_hex(rate: float) -> str:
    if rate < 0.03:  return "#1E8449"
    if rate < 0.06:  return "#B7950B"
    if rate < 0.12:  return "#CA6F1E"
    return "#922B21"


def build_map_df() -> pd.DataFrame:
    hr = hist_df.set_index("county")["flood_rate"].to_dict()
    rows = []
    for c in counties:
        if c not in COUNTY_COORDS:
            continue
        lat, lon = COUNTY_COORDS[c]
        rate = hr.get(c, 0.0)
        rows.append({
            "county": c, "lat": lat, "lon": lon,
            "flood_rate": rate,
            "flood_pct":  round(rate * 100, 2),
            "flood_events": int(hist_df[hist_df["county"]==c]["flood_events"].iloc[0])
                            if c in hist_df["county"].values else 0,
            "risk_tier": ("Low" if rate < 0.03 else
                          "Moderate" if rate < 0.06 else
                          "High" if rate < 0.12 else "Critical"),
        })
    return pd.DataFrame(rows)


MAP_DF = build_map_df()

CHART_TEMPLATE = dict(
    font=dict(family="Inter, sans-serif", size=11, color=TEXT_DARK),
    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=True, gridcolor="#F2F3F4", zeroline=False,
               linecolor=BORDER, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="#F2F3F4", zeroline=False,
               linecolor=BORDER, tickfont=dict(size=10)),
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo area
    st.markdown("""
    <div style="background:rgba(255,255,255,.05);
                border:1px solid rgba(255,255,255,.1);border-radius:14px;
                padding:1.2rem 1.2rem 1rem;margin-bottom:1.1rem;text-align:center">
      <div style="color:white;font-weight:800;font-size:1.05rem;
                  letter-spacing:-.01em">SSD Flood EWS</div>
      <div style="color:#64A8D8;font-size:0.7rem;margin-top:3px;
                  text-transform:uppercase;letter-spacing:.1em">
        Early Warning System</div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.08);
                  font-size:0.68rem;color:#475569">
        79 counties · 2011–2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── HOW TO USE ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(255,255,255,.08);border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:0.8rem;font-size:0.78rem;
                color:#AED6F1;line-height:1.6">
      <b style="color:white">How to use:</b><br>
      1. Select a <b>county</b> and <b>month</b> below<br>
      2. Adjust the climate &amp; terrain sliders to match current or forecast conditions<br>
      3. The flood prediction updates instantly across all tabs
    </div>
    """, unsafe_allow_html=True)

    # ── LOCATION & TIME ───────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Step 1 — Location &amp; Time</p>',
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.4rem">'
                'Which county and month are you assessing?</div>',
                unsafe_allow_html=True)
    sel_county = st.selectbox("County", counties, index=counties.index("Malakal"),
                              help="Select the South Sudan county to assess flood risk for.")
    sel_month  = st.selectbox("Forecast Month", range(1,13),
                              format_func=lambda m: MONTH_NAMES[m-1],
                              index=7,
                              help="The month you want to predict. Rainy season is June–October.")

    # ── Pull ALL defaults from dataset medians for selected county ────────────
    crow     = hist_df[hist_df["county"] == sel_county]
    hist_rate_sb   = float(crow["flood_rate"].iloc[0])   if not crow.empty else 0.0
    hist_events_sb = int(crow["flood_events"].iloc[0])   if not crow.empty else 0

    cd = county_defaults.get(sel_county, {})
    def_rain   = float(min(cd.get("rainfall_mm",            83.0),  336.5))
    def_sm     = float(min(cd.get("soil_moisture_mm",       29.0),  224.9))
    def_maxt   = float(min(max(cd.get("max_temperature_celsius", 35.0), 26.2), 43.1))
    def_mint   = float(min(max(cd.get("min_temperature_celsius", 21.0), 14.7), 27.9))
    def_vpd    = float(min(max(cd.get("vapor_pressure_deficit_kPa", 2.3), 0.4), 5.0))
    def_wet    = float(min(cd.get("wetland_fraction",       0.10),   0.92))
    def_elev   = float(min(max(cd.get("elevation_m",        513.0), 392.0), 1145.0))
    def_slope  = float(min(max(cd.get("slope_deg",           1.7),   0.9),   8.3))
    def_ndvi   = float(min(max(cd.get("ndvi",                0.55),  0.19),  0.85))

    # County quick-stats
    rank_df_sb = hist_df.sort_values("flood_rate", ascending=False).reset_index(drop=True)
    crank = (rank_df_sb[rank_df_sb["county"]==sel_county].index[0]+1
             if sel_county in rank_df_sb["county"].values else "—")
    st.markdown(f"""
    <div style="background:rgba(0,0,0,.18);border-radius:7px;padding:0.6rem 0.9rem;
                margin:0.4rem 0 0.8rem;font-size:0.76rem;color:#D6EAF8;line-height:1.8">
      <b style="color:white">{sel_county}</b><br>
      Historical flood rate: <b style="color:#F9E79F">{hist_rate_sb*100:.1f}%</b>
      &nbsp;({hist_events_sb} recorded events)<br>
      County rank: <b style="color:#F9E79F">#{crank} of 79</b><br>
      <span style="font-size:0.69rem;opacity:.8">All slider values below auto-loaded
      from this county's dataset medians.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── CLIMATE ───────────────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Step 2 — Climate Conditions</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.3rem">'
        'Pre-filled from this county\'s dataset median. '
        'Adjust to match current or forecast conditions. '
        '<b style="color:#F9E79F">Current value shown in yellow.</b></div>',
        unsafe_allow_html=True)

    ckey = sel_county.replace(" ", "_").replace("/", "_")   # county-scoped key prefix

    rainfall   = sb_slider("Rainfall", "mm",  0.0, 336.5, def_rain,  1.0,
                           "{:.0f}",
                           "Total monthly rainfall. Rainy season (Jun–Oct) averages 80–200 mm. "
                           "Higher → more runoff → higher flood risk.",
                           f"rain_{ckey}")
    soil_moist = sb_slider("Soil Moisture", "mm", 0.8, 224.9, def_sm, 0.5,
                           "{:.1f}",
                           "How saturated the soil is. Higher = ground already wet, "
                           "less capacity to absorb rain → higher flood risk.",
                           f"sm_{ckey}")
    max_temp   = sb_slider("Max Temperature", "°C", 26.2, 43.1, def_maxt, 0.1,
                           "{:.1f}",
                           "Daily high temperature. Drives evaporation and water cycle.",
                           f"maxt_{ckey}")
    min_temp   = sb_slider("Min Temperature", "°C", 14.7, 27.9, def_mint, 0.1,
                           "{:.1f}",
                           "Daily low temperature. The gap (max − min) is used as a "
                           "temperature-range feature in the model.",
                           f"mint_{ckey}")
    vpd        = sb_slider("Vapour Pressure Deficit", "kPa", 0.4, 5.0, def_vpd, 0.05,
                           "{:.2f}",
                           "Atmospheric dryness. Low VPD (< 1.5 kPa) = humid air = "
                           "higher flood risk. High VPD = dry conditions.",
                           f"vpd_{ckey}")

    # ── TERRAIN ───────────────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Step 3 — Terrain &amp; Land Cover</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.3rem">'
        'Geographic properties of the county — pre-filled from the dataset. '
        'Terrain is relatively fixed; only adjust if you have updated measurements.</div>',
        unsafe_allow_html=True)

    wetland   = sb_slider("Wetland Fraction", "/ 1.0", 0.0, 0.92, def_wet, 0.01,
                          "{:.2f}",
                          "Share of the county covered by wetland/floodplain. "
                          "Higher = inherently more flood-prone. 0 = none, 0.92 = mostly wetland.",
                          f"wet_{ckey}")
    elevation = sb_slider("Elevation", "m", 392.0, 1145.0, def_elev, 1.0,
                          "{:.0f}",
                          "Average elevation above sea level. "
                          "Lower-lying counties (< 430 m) are most at risk.",
                          f"elev_{ckey}")
    slope     = sb_slider("Slope", "°", 0.9, 8.3, def_slope, 0.1,
                          "{:.1f}",
                          "Average terrain steepness. Flat land (< 1°) drains slowly → "
                          "water pools. Steeper terrain sheds water faster.",
                          f"slope_{ckey}")
    ndvi_val  = sb_slider("NDVI (Vegetation)", "", 0.19, 0.85, def_ndvi, 0.01,
                          "{:.2f}",
                          "Normalised Difference Vegetation Index. "
                          "Higher = denser vegetation cover (values > 0.6 = good cover).",
                          f"ndvi_{ckey}")

    # ── PRIOR MONTH ───────────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Step 4 — Prior Month Status</p>',
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.4rem">'
                'Was there active flooding in this county last month? '
                'This is the single strongest predictor in the model.</div>',
                unsafe_allow_html=True)
    flood_prev = 1 if st.toggle("Flooding occurred last month", value=False,
                                help="Toggle ON if the county experienced flooding in the previous month. This dramatically increases predicted risk.") else 0
    if flood_prev:
        st.markdown('<div style="font-size:0.74rem;color:#F1948A;margin-top:0.3rem">'
                    '⚠ Prior flood toggled ON — this significantly raises predicted probability.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#7FB3D3;text-align:center;line-height:1.7">
      Model: <b style="color:white">{meta['best_model_name']}</b><br>
      AUC-ROC: <b style="color:#F9E79F">{meta['test_metrics'][meta['best_model_name']]['auc_roc']}</b><br>
      Decision threshold: {THRESHOLD*100:.0f}%<br>
      <span style="font-size:0.68rem;opacity:.7">Trained on 14,220 obs · 79 counties · 2011–2025</span>
    </div>
    """, unsafe_allow_html=True)


# ── Run prediction (live, every interaction) ──────────────────────────────────
raw_inputs = dict(
    rainfall_mm=rainfall, soil_moisture_mm=soil_moist,
    max_temperature_celsius=max_temp, min_temperature_celsius=min_temp,
    vapor_pressure_deficit_kPa=vpd, wetland_fraction=wetland,
    elevation_m=elevation, slope_deg=slope, ndvi=ndvi_val,
    flood_prev_month=flood_prev, month=sel_month,
)
prob                                        = run_prediction(raw_inputs)
r_label, r_col, r_cls, r_alert, r_icon, r_action = risk_tier(prob)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
hist_rate   = float(crow["flood_rate"].iloc[0])   if not crow.empty else 0.0
hist_events = int(crow["flood_events"].iloc[0])   if not crow.empty else 0

_lr_auc_hdr = meta['test_metrics'][meta['best_model_name']]['auc_roc']
st.markdown(f"""
<div class="page-header">
  <div class="ph-inner">
    <div class="ph-left">
      <div class="ph-eyebrow">
        ◆ Early Warning System · South Sudan
      </div>
      <h1>County-Level Flood Prediction</h1>
      <p>Machine learning · 79 counties · 2011–2025 · Viewing:
         <b style="color:white">{sel_county}</b>, {MONTH_NAMES[sel_month-1]}</p>
      <div class="ph-badges">
        <span class="ph-badge">Logistic Regression</span>
        <span class="ph-badge">AUC-ROC {_lr_auc_hdr}</span>
        <span class="ph-badge">79 Counties</span>
        <span class="ph-badge">5-Fold CV</span>
      </div>
    </div>
    <div class="ph-right">
      <div>
        <div style="font-size:0.68rem;color:#93C5FD;text-transform:uppercase;
                    letter-spacing:.08em;font-weight:600">Model AUC-ROC</div>
        <div style="font-size:0.72rem;color:#93C5FD;margin-top:2px">
          Hold-out test 2024–2025
        </div>
      </div>
      <div style="font-size:2.6rem;font-weight:800;color:white;line-height:1;
                  letter-spacing:-.03em">{_lr_auc_hdr}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "  Flood Prediction  ",
    "  County Risk Map   ",
    "  Historical Analysis  ",
    "  Model Performance  ",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1  —  FLOOD PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── TOP ROW: 4 quick KPIs ─────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")
    rank_df = hist_df.sort_values("flood_rate", ascending=False).reset_index(drop=True)
    county_rank = (rank_df[rank_df["county"]==sel_county].index[0]+1
                   if sel_county in rank_df["county"].values else "—")

    for col, val, lbl, sub, top_col in [
        (k1, f"{prob*100:.1f}%", "Flood Probability",
         f"{'⚠ Above' if prob>=THRESHOLD else '✓ Below'} {THRESHOLD*100:.0f}% threshold",
         r_col),
        (k2, r_label,            "Risk Level",
         f"{'FLOOD ALERT' if prob>=THRESHOLD else 'No flood predicted'}", r_col),
        (k3, f"{hist_rate*100:.1f}%", "Historical Rate",
         f"{hist_events} events · 180 months", ACCENT),
        (k4, f"#{county_rank}",  "County Rank",
         "of 79 by flood rate", ACCENT),
    ]:
        col.markdown(f"""
        <div class="kpi" style="--kpi-color:{top_col}">
          <div class="kpi-val" style="color:{top_col}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Context banner below KPIs ─────────────────────────────────────────────
    diff = prob - hist_rate
    diff_str = f"{abs(diff)*100:.1f}% {'higher' if diff > 0 else 'lower'} than the historical average"
    if prob >= THRESHOLD:
        banner_cls = "abox-danger" if prob >= 0.5 else "abox-warn"
        banner_msg = (
            f"The model predicts a <b>{prob*100:.1f}% flood probability</b> for <b>{sel_county}</b> "
            f"in <b>{MONTH_NAMES[sel_month-1]}</b> under the current input conditions. "
            f"This is {diff_str} for this county ({hist_rate*100:.1f}%). "
            f"The prediction crosses the decision threshold ({THRESHOLD*100:.0f}%) — "
            f"<b>flood conditions are indicated.</b>"
        )
    else:
        banner_cls = "abox-success" if prob < 0.15 else "abox-info"
        banner_msg = (
            f"The model predicts a <b>{prob*100:.1f}% flood probability</b> for <b>{sel_county}</b> "
            f"in <b>{MONTH_NAMES[sel_month-1]}</b> under the current input conditions. "
            f"This is {diff_str} for this county ({hist_rate*100:.1f}%). "
            f"The probability is below the decision threshold ({THRESHOLD*100:.0f}%) — "
            f"<b>no flood predicted under these conditions.</b>"
        )
    st.markdown(f'<div class="abox {banner_cls}" style="margin:0.5rem 0 1rem">{banner_msg}</div>',
                unsafe_allow_html=True)

    # ── MAIN ROW ──────────────────────────────────────────────────────────────
    col_left, col_mid, col_right = st.columns([1, 1.1, 1.5], gap="medium")

    # ── Verdict ───────────────────────────────────────────────────────────────
    with col_left:
        decision_label = "FLOOD PREDICTED" if prob >= THRESHOLD else "NO FLOOD"
        decision_bg    = r_col if prob >= THRESHOLD else SUCCESS
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Flood Risk Verdict</div>
          <div class="verdict-wrap {r_cls}">
            <div class="v-label">Flood Probability — {sel_county}, {MONTH_NAMES[sel_month-1]}</div>
            <div class="v-prob" style="color:{r_col}">{prob*100:.1f}%</div>
            <div class="v-risk" style="color:{r_col}">{r_label}</div>
            <div style="margin:10px 0 6px">
              <span style="background:{decision_bg};color:white;border-radius:4px;
                           padding:3px 12px;font-size:0.78rem;font-weight:700;
                           letter-spacing:.05em">{decision_label}</span>
            </div>
            <div class="v-dec">Decision threshold: {THRESHOLD*100:.0f}% &nbsp;|&nbsp;
              Model: {meta['best_model_name']}</div>
          </div>

          <div style="margin:0.8rem 0 0.6rem;font-size:0.78rem;color:{MUTED}">
            <b style="color:{TEXT_DARK}">What does this probability mean?</b><br>
            The model assigns a <b>{prob*100:.1f}%</b> chance that the conditions you entered
            ({rainfall:.0f} mm rainfall, {soil_moist:.0f} mm soil moisture, month: {MONTH_NAMES[sel_month-1]})
            are consistent with a flood month. Values above <b>{THRESHOLD*100:.0f}%</b> trigger a flood alert.
            Historical base rate for this county is <b>{hist_rate*100:.1f}%</b>.
          </div>

          <div class="{r_alert} abox">
            <b>Recommended action</b><br>{r_action}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauge ─────────────────────────────────────────────────────────────────
    with col_mid:
        st.markdown('<div class="card" style="padding-bottom:0.5rem">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Probability Gauge</div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number=dict(suffix="%", font=dict(size=40, color=r_col, family="Inter")),
            gauge=dict(
                axis=dict(range=[0,100], tickwidth=1, tickcolor="#ccc",
                          tickvals=[0,25,50,75,100],
                          ticktext=["0%","25%","50%","75%","100%"]),
                bar=dict(color=r_col, thickness=0.28),
                bgcolor="white", bordercolor=BORDER, borderwidth=1,
                steps=[
                    dict(range=[0,25],   color="#EAFAF1"),
                    dict(range=[25,50],  color="#FEF9E7"),
                    dict(range=[50,75],  color="#FDEDEC"),
                    dict(range=[75,100], color="#F5EEF8"),
                ],
                threshold=dict(
                    line=dict(color=TEXT_DARK, width=2.5),
                    thickness=0.82, value=THRESHOLD*100,
                ),
            ),
        ))
        fig_gauge.update_layout(
            height=290,
            margin=dict(t=30, b=40, l=30, r=30),
            paper_bgcolor=BG_CARD,
            font=dict(family="Inter"),
            annotations=[dict(
                text=f"Decision threshold: {THRESHOLD*100:.0f}%",
                x=0.5, y=-0.12, showarrow=False,
                font=dict(size=10, color=MUTED)
            )],
        )
        st.plotly_chart(fig_gauge, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature contributions ─────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Feature Importance — What Drives This Prediction</div>',
                    unsafe_allow_html=True)

        FEAT_LABEL = {
            "rainfall_mm":"Rainfall (mm)",
            "soil_moisture_mm":"Soil Moisture (mm)",
            "max_temperature_celsius":"Max Temperature (°C)",
            "min_temperature_celsius":"Min Temperature (°C)",
            "vapor_pressure_deficit_kPa":"Vapour Pressure Deficit",
            "wetland_fraction":"Wetland Fraction",
            "elevation_m":"Elevation (m)",
            "slope_deg":"Slope (°)",
            "ndvi":"NDVI",
            "flood_prev_month":"Flood Previous Month ★",
            "temp_range":"Temperature Range",
            "wetness_index":"Wetness Index",
            "rain_wetland":"Rainfall × Wetland",
            "month_sin":"Seasonal Signal (sin)",
            "month_cos":"Seasonal Signal (cos)",
        }
        fi      = meta.get("feature_importance", {})
        fi_sort = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        labels  = [FEAT_LABEL.get(k,k) for k,_ in fi_sort]
        vals    = [v for _,v in fi_sort]
        avg_fi  = np.mean(vals)

        bar_colors = [r_col if v >= avg_fi else "#AEB6BF" for v in vals]
        fig_fi = go.Figure(go.Bar(
            x=vals[::-1], y=labels[::-1],
            orientation="h",
            marker=dict(color=bar_colors[::-1], line=dict(width=0)),
            text=[f"{v*100:.1f}%" for v in vals[::-1]],
            textposition="outside",
            textfont=dict(size=9.5, color=TEXT_MUT),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            cliponaxis=False,
        ))
        fig_fi.add_vline(x=avg_fi, line_dash="dot", line_color="#AEB6BF",
                         line_width=1.5,
                         annotation_text="avg",
                         annotation_font=dict(size=9, color=MUTED))
        fig_fi.update_layout(
            height=380, margin=dict(t=0, b=0, l=10, r=60),
            xaxis=dict(title=None, showgrid=True, gridcolor="#F2F3F4",
                       zeroline=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=False, tickfont=dict(size=10)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_fi, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── BOTTOM ROW: What-if + Input table ─────────────────────────────────────
    b1, b2 = st.columns([1.6, 1], gap="medium")

    with b1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Sensitivity Analysis — How Rainfall Affects Probability</div>',
                    unsafe_allow_html=True)

        rain_x  = np.linspace(0, 336.5, 80)
        probs_r = []
        for rv in rain_x:
            tmp = raw_inputs.copy()
            tmp["rainfall_mm"] = rv
            probs_r.append(run_prediction(tmp))

        fig_wi = go.Figure()
        fig_wi.add_hrect(y0=THRESHOLD*100, y1=100,
                         fillcolor="rgba(146,43,33,.06)", line_width=0)
        fig_wi.add_trace(go.Scatter(
            x=rain_x, y=[p*100 for p in probs_r],
            mode="lines", line=dict(color=ACCENT, width=2.5),
            fill="tozeroy", fillcolor="rgba(46,134,193,.09)",
            hovertemplate="Rainfall: %{x:.0f} mm<br>Probability: %{y:.1f}%<extra></extra>",
        ))
        fig_wi.add_vline(x=rainfall, line_dash="dash",
                         line_color=r_col, line_width=1.8,
                         annotation_text=f"Current: {rainfall:.0f} mm",
                         annotation_position="top right",
                         annotation_font=dict(size=10, color=r_col))
        fig_wi.add_hline(y=THRESHOLD*100,
                         line_dash="dot", line_color=DANGER, line_width=1.5,
                         annotation_text=f"Threshold ({THRESHOLD*100:.0f}%)",
                         annotation_position="bottom right",
                         annotation_font=dict(size=9, color=DANGER))
        fig_wi.update_layout(
            height=230,
            xaxis=dict(title="Rainfall (mm)", showgrid=True, gridcolor="#F2F3F4",
                       range=[0,336.5], tickfont=dict(size=10)),
            yaxis=dict(title="Flood Probability (%)", range=[0,105],
                       showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=10)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"), margin=dict(t=5,b=5,l=5,r=10),
        )
        st.plotly_chart(fig_wi, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Input Values Summary</div>',
                    unsafe_allow_html=True)
        inp = pd.DataFrame({
            "Variable": ["Rainfall","Soil Moisture","Max Temp","Min Temp",
                         "VPD","Wetland Fraction","Elevation",
                         "Slope","NDVI","Flood Prev. Month"],
            "Value":    [f"{rainfall:.1f} mm", f"{soil_moist:.1f} mm",
                         f"{max_temp:.1f} °C",  f"{min_temp:.1f} °C",
                         f"{vpd:.2f} kPa",     f"{wetland:.2f}",
                         f"{elevation:.0f} m", f"{slope:.1f} °",
                         f"{ndvi_val:.2f}",
                         "Yes" if flood_prev else "No"],
        })
        st.dataframe(inp, hide_index=True, use_container_width=True, height=340)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2  —  COUNTY RISK RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### County Flood Risk — All 79 Counties")

    TIER_COLOR = {
        "Low":      "#1E8449",
        "Moderate": "#B7950B",
        "High":     "#CA6F1E",
        "Critical": "#922B21",
    }
    TIER_ORDER = ["Critical", "High", "Moderate", "Low"]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    tc = MAP_DF["risk_tier"].value_counts()
    k1t, k2t, k3t, k4t = st.columns(4, gap="small")
    for col, tier, color in [
        (k1t, "Critical", "#922B21"),
        (k2t, "High",     "#CA6F1E"),
        (k3t, "Moderate", "#B7950B"),
        (k4t, "Low",      "#1E8449"),
    ]:
        n = int(tc.get(tier, 0))
        col.markdown(f"""
        <div class="kpi" style="border-top:3px solid {color}">
          <div class="kpi-val" style="color:{color}">{n}</div>
          <div class="kpi-lbl">{tier} Risk Counties</div>
          <div class="kpi-sub">{n/79*100:.0f}% of 79 counties</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main chart + side panel ────────────────────────────────────────────────
    col_chart, col_panel = st.columns([2.4, 1], gap="medium")

    with col_chart:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">All 79 Counties — Historical Flood Rate, Sorted by Risk</div>',
            unsafe_allow_html=True)

        # Sorted bar chart — all 79 counties, colour by tier, selected highlighted
        chart_df = hist_df.sort_values("flood_rate", ascending=True).copy()
        chart_df["flood_pct"] = chart_df["flood_rate"] * 100
        chart_df["risk_tier"] = chart_df["flood_rate"].apply(
            lambda r: "Critical" if r >= 0.12 else
                      "High"     if r >= 0.06 else
                      "Moderate" if r >= 0.03 else "Low"
        )
        chart_df["bar_color"]  = chart_df["risk_tier"].map(TIER_COLOR)
        chart_df["is_sel"]     = chart_df["county"] == sel_county
        # Selected county gets a brighter outline marker
        chart_df["bar_color"]  = chart_df.apply(
            lambda r: r_col if r["is_sel"] else r["bar_color"], axis=1
        )

        fig_bar = go.Figure(go.Bar(
            x=chart_df["flood_pct"],
            y=chart_df["county"],
            orientation="h",
            marker=dict(
                color=chart_df["bar_color"],
                line=dict(
                    color=chart_df["is_sel"].apply(lambda x: TEXT_DARK if x else "rgba(0,0,0,0)"),
                    width=chart_df["is_sel"].apply(lambda x: 2 if x else 0),
                ),
            ),
            text=chart_df.apply(
                lambda r: f"  {r['flood_pct']:.1f}%{'  ◀ '+sel_county if r['is_sel'] else ''}",
                axis=1
            ),
            textposition="outside",
            textfont=dict(size=chart_df["is_sel"].apply(lambda x: 10 if x else 8.5).tolist(),
                          color=chart_df["is_sel"].apply(
                              lambda x: TEXT_DARK if x else TEXT_MUT).tolist()),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Historical flood rate: %{x:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
        fig_bar.add_vline(x=hist_df["flood_rate"].mean()*100,
                          line_dash="dot", line_color=MUTED, line_width=1.5,
                          annotation_text=f"National avg {hist_df['flood_rate'].mean()*100:.1f}%",
                          annotation_font=dict(size=9, color=MUTED),
                          annotation_position="top right")
        fig_bar.update_layout(
            height=max(600, len(chart_df) * 15),
            margin=dict(t=10, b=10, l=10, r=80),
            xaxis=dict(title="Historical Flood Rate (%)", showgrid=True,
                       gridcolor="#F2F3F4", tickfont=dict(size=9)),
            yaxis=dict(showgrid=False, tickfont=dict(size=8.5)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown(
            '<div style="font-size:0.74rem;color:#717D7E;margin-top:0.3rem">'
            'Flood rate = % of months (2011–2025) with recorded flooding. '
            'The selected county is highlighted. '
            'Colours: green = Low, amber = Moderate, orange = High, red = Critical.</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_panel:
        # Risk donut
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Risk Breakdown</div>', unsafe_allow_html=True)
        tier_counts = MAP_DF["risk_tier"].value_counts().reindex(
            ["Critical","High","Moderate","Low"], fill_value=0)
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts.index.tolist(),
            values=tier_counts.values,
            marker_colors=["#922B21","#CA6F1E","#B7950B","#1E8449"],
            hole=0.52,
            textinfo="label+value",
            textfont=dict(size=10),
            hovertemplate="<b>%{label} Risk</b><br>%{value} counties (%{percent})<extra></extra>",
            sort=False,
            direction="clockwise",
        ))
        fig_donut.update_layout(
            height=230, margin=dict(t=5,b=5,l=5,r=5),
            paper_bgcolor=BG_CARD, showlegend=False,
            font=dict(family="Inter"),
            annotations=[dict(text="79<br>counties", x=0.5, y=0.5,
                              font=dict(size=11, color=TEXT_DARK), showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)

        # Selected county summary
        st.markdown(f"""
        <div class="card" style="border-top:3px solid {r_col}">
          <div class="card-title">Selected County</div>
          <div style="font-size:1.05rem;font-weight:700;color:{PRIMARY}">{sel_county}</div>
          <div style="font-size:0.78rem;color:{MUTED};margin:4px 0 10px">
            Rank #{county_rank} of 79 by historical flood rate</div>
          <table style="width:100%;font-size:0.82rem;border-collapse:collapse">
            <tr><td style="color:{MUTED};padding:3px 0">Predicted prob.</td>
                <td style="font-weight:700;color:{r_col};text-align:right">{prob*100:.1f}%</td></tr>
            <tr><td style="color:{MUTED};padding:3px 0">Historical rate</td>
                <td style="font-weight:600;text-align:right">{hist_rate*100:.1f}%</td></tr>
            <tr><td style="color:{MUTED};padding:3px 0">Recorded events</td>
                <td style="font-weight:600;text-align:right">{hist_events} of 180 months</td></tr>
            <tr><td style="color:{MUTED};padding:3px 0">Risk tier</td>
                <td style="font-weight:700;color:{r_col};text-align:right">{r_label}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Top 10 table
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top 10 Highest Risk Counties</div>',
                    unsafe_allow_html=True)
        top10 = (hist_df.nlargest(10, "flood_rate")
                        [["county","flood_rate","flood_events"]].copy())
        top10.columns = ["County","Rate","Events"]
        top10["Rate"] = top10["Rate"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(top10, hide_index=True, use_container_width=True, height=310)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3  —  HISTORICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    cty_ts = (monthly_df[monthly_df["county"]==sel_county].copy()
              .sort_values(["year","month"]))
    cty_ts["date"] = pd.to_datetime(cty_ts[["year","month"]].assign(day=1))

    st.markdown(f"### Historical Flood Record — {sel_county}")

    # ── Explainer banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="abox abox-info" style="margin-bottom:1rem">
      <b>How to read this page:</b><br>
      The charts below show <b>recorded flood events from the training dataset (2011–2025)</b>.
      A flood rate of <b>0%</b> in a given year means <em>no flooding was recorded in the dataset for that period</em>
      — it does <b>not</b> mean the county is safe or never floods.
      Ground truth in the dataset captures <em>large-scale inundation months</em> only;
      localised or short-duration floods may be underreported.<br><br>
      The <b>model prediction</b> (left sidebar) is independent of this historical rate —
      it uses current climate inputs (rainfall, soil moisture, temperature, etc.) to estimate flood risk
      for the conditions you specify. A county with <b>0% historical rate can still receive a high
      predicted probability</b> if current conditions are dangerous.
    </div>
    """, unsafe_allow_html=True)

    # ── Timeline + Seasonality ────────────────────────────────────────────────
    h1, h2 = st.columns([2.2, 1], gap="medium")

    with h1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Flood Event Timeline (2011–2025)</div>',
                    unsafe_allow_html=True)

        fig_tl = go.Figure()
        # Shade rainy season (Jun–Sep)
        for y in range(2011, 2026):
            fig_tl.add_vrect(
                x0=f"{y}-06-01", x1=f"{y}-10-01",
                fillcolor="rgba(46,134,193,0.06)", line_width=0,
                annotation_text="Rainy season" if y==2011 else "",
                annotation_position="top left",
                annotation_font=dict(size=8.5, color=MUTED),
            )
        # No-flood dots
        nf = cty_ts[cty_ts["flood"]==0]
        fig_tl.add_trace(go.Scatter(
            x=nf["date"], y=nf["flood"], mode="markers", name="No Flood",
            marker=dict(color="#D5D8DC", size=5, opacity=0.7),
        ))
        # Flood events
        fl = cty_ts[cty_ts["flood"]==1]
        fig_tl.add_trace(go.Scatter(
            x=fl["date"], y=fl["flood"], mode="markers", name="Flood Event",
            marker=dict(color=DANGER, size=12, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>Flood Event</b><br>%{x|%B %Y}<extra></extra>",
        ))
        fig_tl.update_layout(
            height=240,
            xaxis=dict(title="Date", showgrid=True, gridcolor="#F2F3F4",
                       tickfont=dict(size=9.5)),
            yaxis=dict(tickvals=[0,1], ticktext=["No Flood","Flood"],
                       range=[-0.3,1.5], tickfont=dict(size=9.5)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"), margin=dict(t=5,b=5,l=5,r=5),
            legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10)),
        )
        st.plotly_chart(fig_tl, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)

    with h2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Monthly Seasonality</div>',
                    unsafe_allow_html=True)
        seas     = cty_ts.groupby("month")["flood"].mean() * 100
        sea_vals = [seas.get(m, 0) for m in range(1,13)]
        avg_sea  = np.mean(sea_vals)

        fig_sea = go.Figure(go.Bar(
            x=MONTH_SHORT, y=sea_vals,
            marker_color=[DANGER if v>avg_sea else ACCENT for v in sea_vals],
            marker_line_width=0,
            text=[f"{v:.0f}%" if v>0 else "" for v in sea_vals],
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{x}<br>Flood rate: %{y:.1f}%<extra></extra>",
        ))
        fig_sea.add_hline(y=avg_sea, line_dash="dot", line_color=MUTED,
                          line_width=1, annotation_text=f"avg {avg_sea:.1f}%",
                          annotation_font=dict(size=8.5, color=MUTED))
        fig_sea.update_layout(
            height=240,
            xaxis=dict(tickfont=dict(size=9.5), showgrid=False),
            yaxis=dict(title="Flood Rate (%)", showgrid=True,
                       gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"), margin=dict(t=5,b=5,l=5,r=5),
        )
        st.plotly_chart(fig_sea, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Annual trend ──────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Annual Flood Rate — County vs National Average</div>',
                unsafe_allow_html=True)

    yr_cty = cty_ts.groupby("year")["flood"].mean() * 100
    yr_nat = monthly_df.groupby("year")["flood"].mean() * 100

    fig_yr = go.Figure()
    fig_yr.add_trace(go.Scatter(
        x=yr_nat.index, y=yr_nat.values, mode="lines",
        name="National average",
        line=dict(color="#AEB6BF", dash="dot", width=2),
        hovertemplate="Year: %{x}<br>National avg: %{y:.1f}%<extra></extra>",
    ))
    fig_yr.add_trace(go.Scatter(
        x=yr_cty.index, y=yr_cty.values, mode="lines+markers",
        name=sel_county,
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=7, color=ACCENT, line=dict(color="white",width=1.5)),
        fill="tozeroy", fillcolor="rgba(46,134,193,0.08)",
        hovertemplate="Year: %{x}<br>Flood rate: %{y:.1f}%<extra></extra>",
    ))
    fig_yr.update_layout(
        height=240,
        xaxis=dict(title="Year", showgrid=True, gridcolor="#F2F3F4",
                   dtick=1, tickangle=-45, tickfont=dict(size=9.5)),
        yaxis=dict(title="Flood Rate (%)", showgrid=True,
                   gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(family="Inter"), margin=dict(t=5,b=5,l=5,r=5),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
    )
    st.plotly_chart(fig_yr, use_container_width=True,
                    config=dict(displayModeBar=False))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Multi-county comparison ────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Multi-County Comparison</div>',
                unsafe_allow_html=True)

    defaults = [sel_county]
    for c in ["Malakal","Bor South","Rubkona","Juba","Renk"]:
        if c != sel_county and c in counties: defaults.append(c)
        if len(defaults) == 5: break

    compare = st.multiselect("Select counties:", counties,
                             default=defaults, key="cmp")
    if compare:
        rows = []
        for c in compare:
            cd = monthly_df[monthly_df["county"]==c].groupby("year")["flood"].mean()*100
            for yr, r in cd.items():
                rows.append({"County":c,"Year":yr,"Flood Rate (%)":round(r,2)})
        cmp_df = pd.DataFrame(rows)
        fig_cmp = px.line(cmp_df, x="Year", y="Flood Rate (%)", color="County",
                          markers=True,
                          color_discrete_sequence=px.colors.qualitative.Safe)
        fig_cmp.update_layout(
            height=300, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"), margin=dict(t=5,b=5,l=5,r=5),
            xaxis=dict(showgrid=True, gridcolor="#F2F3F4", dtick=2,
                       tickfont=dict(size=9.5)),
            yaxis=dict(showgrid=True, gridcolor="#F2F3F4",
                       tickfont=dict(size=9.5)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10)),
        )
        st.plotly_chart(fig_cmp, use_container_width=True,
                        config=dict(displayModeBar=False))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── System KPIs ────────────────────────────────────────────────────────────
    st.markdown("#### System-Wide Statistics")
    sk1,sk2,sk3,sk4,sk5 = st.columns(5,gap="small")
    for col,val,lbl,bdr in [
        (sk1, f"{int(hist_df['flood_events'].sum()):,}", "Total Flood Events",    DANGER),
        (sk2, f"{hist_df['flood_rate'].mean()*100:.1f}%","National Mean Rate",    WARNING),
        (sk3, str((hist_df["flood_rate"]>0.08).sum()),   "Critical Counties (>8%)",CRITICAL),
        (sk4, str((hist_df["flood_rate"]<0.01).sum()),   "Low-Risk Counties (<1%)",SUCCESS),
        (sk5, hist_df.loc[hist_df["flood_rate"].idxmax(),"county"],
                                                          "Highest-Risk County",  PRIMARY),
    ]:
        col.markdown(f"""
        <div class="kpi" style="border-top:3px solid {bdr}">
          <div class="kpi-val" style="font-size:1.5rem">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4  —  MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Model Performance, Validation & Scientific Basis")

    # ── Why we trust these results ────────────────────────────────────────────
    lr_auc   = meta['test_metrics']['Logistic Regression']['auc_roc']
    lr_f1    = meta['test_metrics']['Logistic Regression']['f1']
    lr_prec  = meta['test_metrics']['Logistic Regression']['precision']
    rf_auc   = meta['test_metrics']['Random Forest']['auc_roc']
    rf_prec  = meta['test_metrics']['Random Forest']['precision']
    top_feat = list(meta['feature_importance'].keys())[0]
    top_imp  = list(meta['feature_importance'].values())[0]
    cv_auc   = meta['cv_metrics']['Logistic Regression']['auc_roc_mean']
    cv_std   = meta['cv_metrics']['Logistic Regression']['auc_roc_std']

    st.markdown(f"""
    <div class="abox abox-info" style="margin-bottom:1rem;font-size:0.88rem">
    <b>Deployed model: Logistic Regression — why this choice and why we trust it</b><br><br>
    Four models were evaluated: Logistic Regression, Random Forest, XGBoost, and LightGBM.
    <b>Logistic Regression was selected as the operational model</b> because it achieved
    the highest F1 score ({lr_f1:.4f}) and Precision ({lr_prec:.4f}) on the unseen 2024–2025 test set.
    Precision of {lr_prec:.0%} means {lr_prec:.0%} of flood alerts are genuine — critical for a
    humanitarian early warning system where false alarms trigger costly evacuations and erode
    institutional trust. Random Forest achieved the highest AUC-ROC ({rf_auc:.4f}) but with precision
    of only {rf_prec:.4f}, meaning over half of its alerts would be false positives.
    <br><br>
    Cross-validation (5-fold TimeSeriesSplit, 2011–2023) confirmed AUC =
    {cv_auc:.4f} ± {cv_std:.4f}, showing stable generalisation over time.
    The dominant predictor (<b>{top_feat}</b>, normalised coefficient {top_imp*100:.1f}%)
    reflects the hydrological fact that flood conditions persist month-to-month.
    A leakage audit excluded <code>water_fraction</code> — it encodes the flood label definition itself.
    Climate-only variables still achieve AUC = {meta['ablation'][2]['Test AUC']:.3f},
    confirming the model has learned genuine climate-flood relationships.
    </div>
    """, unsafe_allow_html=True)

    # ── Metric explanation guide ──────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">What Each Metric Means</div>',
                unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4,gap="small")
    for col,metric,defn,good in [
        (mc1,"AUC-ROC","Ability to rank flood months above no-flood months. 1.0=perfect, 0.5=random. Best for imbalanced data.","≥ 0.85 is strong"),
        (mc2,"F1 Score","Harmonic mean of Precision & Recall. Balances false alarms vs missed floods.","≥ 0.50 is solid for 1:20 imbalance"),
        (mc3,"Precision","Of all flood predictions, what fraction were real floods? High = few false alarms.","Higher = fewer wasted alerts"),
        (mc4,"Recall","Of all real floods, what fraction did we catch? High = fewer missed events.","Higher = fewer missed floods"),
    ]:
        col.markdown(f"""
        <div style="background:#F8F9FA;border-radius:8px;padding:0.9rem;
                    border-top:3px solid {ACCENT}">
          <div style="font-weight:700;font-size:0.9rem;color:{PRIMARY}">{metric}</div>
          <div style="font-size:0.8rem;color:{TEXT_MUT};margin:6px 0">{defn}</div>
          <div style="font-size:0.75rem;color:{SUCCESS};font-weight:600">{good}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── All model results ─────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Hold-Out Test Results — 2024–2025 (Never Seen During Training)</div>',
                unsafe_allow_html=True)
    tm = meta["test_metrics"]
    model_names = [n for n in tm.keys()][:4]
    tc1, tc2, tc3, tc4 = st.columns(4, gap="small")
    for col, name in zip([tc1, tc2, tc3, tc4], model_names):
        mt = tm[name]
        is_best = (name == meta["best_model_name"])
        auc_col = SUCCESS if is_best else ACCENT
        card_cls = "model-card model-card-best" if is_best else "model-card"
        best_badge = (
            f'<div style="display:inline-block;background:{SUCCESS};color:white;'
            f'font-size:0.65rem;font-weight:700;border-radius:20px;'
            f'padding:2px 8px;letter-spacing:.04em;margin-bottom:8px">✓ DEPLOYED</div>'
            if is_best else ""
        )
        col.markdown(f"""
        <div class="{card_cls}">
          {best_badge}
          <div style="font-size:0.78rem;font-weight:700;color:{TEXT_DARK};
                      margin-bottom:4px">{name}</div>
          <div class="model-auc" style="color:{auc_col}">{mt['auc_roc']:.4f}</div>
          <div style="font-size:0.68rem;color:{MUTED};margin-bottom:10px;
                      text-transform:uppercase;letter-spacing:.06em">AUC-ROC</div>
          <div style="font-size:0.79rem;">
            <div class="model-metric-row">
              <span style="color:{MUTED}">F1</span>
              <b style="color:{TEXT_DARK}">{mt['f1']:.4f}</b>
            </div>
            <div class="model-metric-row">
              <span style="color:{MUTED}">Precision</span>
              <b style="color:{TEXT_DARK}">{mt['precision']:.4f}</b>
            </div>
            <div class="model-metric-row">
              <span style="color:{MUTED}">Recall</span>
              <b style="color:{TEXT_DARK}">{mt['recall']:.4f}</b>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── CV + Ablation side by side ────────────────────────────────────────────
    mp1, mp2 = st.columns(2, gap="medium")

    with mp1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Cross-Validation AUC-ROC (5-Fold TimeSeriesSplit · 2011–2023)</div>',
                    unsafe_allow_html=True)
        cv_m  = meta["cv_metrics"]
        names = list(cv_m.keys())
        means = [cv_m[n]["auc_roc_mean"] for n in names]
        stds  = [cv_m[n]["auc_roc_std"]  for n in names]
        colors_cv = [SUCCESS if n==meta["best_model_name"] else ACCENT for n in names]

        fig_cv = go.Figure()
        for nm, mn, sd, cc in zip(names, means, stds, colors_cv):
            fig_cv.add_trace(go.Bar(
                name=nm, x=[nm], y=[mn],
                error_y=dict(type="data", array=[sd], visible=True,
                             thickness=1.8, width=8),
                marker_color=cc, marker_line_width=0, width=0.5,
                text=[f"{mn:.4f}"], textposition="outside",
                textfont=dict(size=10, color=TEXT_DARK),
            ))
        fig_cv.add_hline(y=0.9, line_dash="dot", line_color="#AEB6BF",
                         line_width=1,
                         annotation_text="0.90 reference",
                         annotation_font=dict(size=9, color=MUTED))
        fig_cv.update_layout(
            height=290, showlegend=False,
            yaxis=dict(range=[0,1.1], title="AUC-ROC",
                       showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
            xaxis=dict(tickfont=dict(size=9.5), showgrid=False),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="Inter"), margin=dict(t=15,b=0,l=5,r=5),
        )
        st.plotly_chart(fig_cv, use_container_width=True,
                        config=dict(displayModeBar=False))
        st.markdown(f"""
        <div class="abox abox-info" style="font-size:0.8rem">
        The ±std bars show consistency across all 5 time windows.
        {meta['best_model_name']} achieved
        {cv_m[meta['best_model_name']]['auc_roc_mean']:.4f}
        ± {cv_m[meta['best_model_name']]['auc_roc_std']:.4f}.
        Low std = model generalises stably over time.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with mp2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Ablation Study — Isolating Each Feature Group (XGBoost)</div>',
                    unsafe_allow_html=True)
        abl = pd.DataFrame(meta.get("ablation", []))
        if not abl.empty:
            abl_c = ["#1B4F72","#2E86C1","#AEB6BF"]
            fig_abl = make_subplots(rows=1, cols=1)
            for i, row in abl.iterrows():
                fig_abl.add_trace(go.Bar(
                    name=row["Feature Set"],
                    x=["CV AUC","Test AUC"],
                    y=[row["CV AUC"], row["Test AUC"]],
                    marker_color=abl_c[i%3], marker_line_width=0, width=0.3,
                    text=[f"{row['CV AUC']:.3f}", f"{row['Test AUC']:.3f}"],
                    textposition="outside", textfont=dict(size=10),
                ))
            fig_abl.update_layout(
                height=240, barmode="group",
                yaxis=dict(range=[0,1.12], title="AUC-ROC",
                           showgrid=True, gridcolor="#F2F3F4",
                           tickfont=dict(size=9.5)),
                xaxis=dict(tickfont=dict(size=10), showgrid=False),
                paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                font=dict(family="Inter"), margin=dict(t=15,b=0,l=5,r=5),
                legend=dict(orientation="h", y=1.15, font=dict(size=9.5)),
            )
            st.plotly_chart(fig_abl, use_container_width=True,
                            config=dict(displayModeBar=False))

            delta = abl.iloc[0]["Test AUC"] - abl.iloc[2]["Test AUC"]
            st.markdown(f"""
            <div class="abox abox-info" style="font-size:0.8rem">
            <b>Key finding:</b> Removing <code>flood_prev_month</code> drops AUC
            {abl.iloc[0]['Test AUC']:.3f} → {abl.iloc[1]['Test AUC']:.3f} (−{abl.iloc[0]['Test AUC']-abl.iloc[1]['Test AUC']:.3f}).
            Pure climate variables alone still achieve AUC {abl.iloc[2]['Test AUC']:.3f},
            confirming the model captures real climate-flood signals beyond simple persistence.
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── National flood heatmap ────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">National Flood Calendar — Monthly × Annual Flood Rate (All 79 Counties)</div>',
                unsafe_allow_html=True)
    heat = (monthly_df.groupby(["year","month"])["flood"].mean()*100).reset_index()
    pivot = heat.pivot(index="month", columns="year", values="flood")
    pivot.index = [MONTH_SHORT[i-1] for i in pivot.index]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0,"#EAFAF1"],[0.25,"#F9E79F"],
                    [0.6,"#E67E22"],[1.0,"#922B21"]],
        text=np.round(pivot.values,1),
        texttemplate="%{text}%",
        textfont=dict(size=8.5),
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Flood rate: %{z:.1f}%<extra></extra>",
        showscale=True,
        colorbar=dict(title=dict(text="Rate %",side="right"),
                      thickness=12, len=0.9,
                      tickfont=dict(size=9)),
        xgap=2, ygap=2,
    ))
    fig_heat.update_layout(
        height=300, margin=dict(t=5,b=5,l=5,r=60),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9.5),
                   title="Year", side="bottom"),
        yaxis=dict(title="Month", tickfont=dict(size=9.5)),
        paper_bgcolor=BG_CARD, font=dict(family="Inter"),
    )
    st.plotly_chart(fig_heat, use_container_width=True,
                    config=dict(displayModeBar=False))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Methodology table ─────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Methodology & Transparency</div>',
                unsafe_allow_html=True)
    meth = meta.get("methodology", {})
    excl = meta.get("excluded_features", [])

    mm1,mm2,mm3 = st.columns(3, gap="medium")
    with mm1:
        st.markdown(f"""
        <div class="abox abox-info">
        <b>Validation strategy</b><br>
        {meth.get('cv','')}<br><br>
        <b>Test set</b><br>
        {meth.get('test','')}
        </div>""", unsafe_allow_html=True)
    with mm2:
        st.markdown(f"""
        <div class="abox abox-info">
        <b>Class imbalance (1:20)</b><br>
        {meth.get('imbalance','')}<br><br>
        <b>Threshold selection</b><br>
        {meth.get('threshold','')}
        </div>""", unsafe_allow_html=True)
    with mm3:
        st.markdown(f"""
        <div class="abox abox-warn">
        <b>Excluded feature: <code>{', '.join(excl)}</code></b><br>
        {meta.get('exclusion_reason','')}
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:{MUTED};font-size:0.78rem;padding:0.3rem 0 1rem">
  South Sudan Flood Early Warning System &nbsp;·&nbsp;
  {meta['best_model_name']} · AUC-ROC {meta['test_metrics'][meta['best_model_name']]['auc_roc']} · F1 {meta['test_metrics'][meta['best_model_name']]['f1']} · Precision {meta['test_metrics'][meta['best_model_name']]['precision']} · 79 Counties · 2011–2025<br>
  Author: <b>Chol Monykuch</b> &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
