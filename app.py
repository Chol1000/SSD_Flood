"""
South Sudan Flood Early Warning System
Professional Streamlit Dashboard — v2
Author: Chol Monykuch
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, io

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── MUST be first ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SSD Flood Early Warning",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="auto",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
SB        = "#0B1120"      # Sidebar background
BLUE      = "#1D4ED8"      # Primary blue
EMERALD   = "#059669"      # Low risk / success
EM_BRIGHT = "#34D399"      # Emerald accent (tabs, highlights)
AMBER     = "#D97706"      # Moderate
ORANGE    = "#EA580C"      # High
RED       = "#DC2626"      # Danger
VIOLET    = "#7C3AED"      # Critical
MUTED     = "#64748B"
BG_PAGE   = "#EFF2F7"
BG_CARD   = "#FFFFFF"
TEXT_DARK = "#0F172A"
TEXT_MUT  = "#64748B"
BORDER    = "#E2E8F0"

# Legacy aliases (used throughout existing chart/HTML code)
PRIMARY   = SB
ACCENT    = BLUE
ACCENT2   = "#0EA5E9"
SUCCESS   = EMERALD
WARNING   = AMBER
DANGER    = RED
CRITICAL  = VIOLET

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }}
.stApp {{ background-color: {BG_PAGE}; }}
.block-container {{ padding: 1.5rem 2.5rem 4rem !important; max-width: 1440px; }}

/* ── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {{
    background: {SB};
    padding-top: 0;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {{ color: #CBD5E1 !important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.1) !important; margin: 0.7rem 0 !important; }}
section[data-testid="stSidebar"] .stRadio > label {{ color: #CBD5E1 !important; }}

section[data-testid="stSidebar"] .stSelectbox label {{
    color: #94A3B8 !important; font-size:0.75rem !important;
    text-transform:uppercase; letter-spacing:.06em;
}}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stSelectbox * {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] .stSelectbox > label {{ color: #94A3B8 !important; }}
section[data-testid="stSidebar"] .stSelectbox svg {{ fill: {EM_BRIGHT} !important; }}

section[data-testid="stSidebar"] .stSlider > label {{
    color: #94A3B8 !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: .06em;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {{
    background: rgba(255,255,255,0.12) !important; height: 4px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {{
    background: {EM_BRIGHT} !important;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background: #FFFFFF !important;
    border: 2px solid {EM_BRIGHT} !important;
    box-shadow: 0 2px 8px rgba(52,211,153,0.4) !important;
    width: 18px !important; height: 18px !important;
    cursor: grab !important; top: -7px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:active {{
    cursor: grabbing !important;
    box-shadow: 0 0 0 5px rgba(52,211,153,0.25) !important;
}}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
    color: #475569 !important; font-size: 0.68rem !important;
}}

/* ── Underline tab bar ────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 2px solid {BORDER};
    border-radius: 0;
    padding: 0 0.25rem;
    gap: 0;
    box-shadow: none;
    margin-bottom: 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0;
    padding: 0.65rem 1.5rem;
    font-weight: 500;
    font-size: 0.88rem;
    color: {MUTED};
    border: none !important;
    outline: none !important;
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_DARK} !important;
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: {TEXT_DARK} !important;
    font-weight: 700;
    border-bottom: 3px solid {EM_BRIGHT} !important;
}}

/* ── Cards ───────────────────────────────────────── */
.card {{
    background: {BG_CARD}; border-radius: 14px;
    border: 1px solid {BORDER};
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}}
.card-title {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: {MUTED};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.6rem; margin-bottom: 1rem;
}}

/* ── KPI strip — left-border style ──────────────── */
.kpi {{
    background: {BG_CARD};
    border-radius: 12px;
    border: 1px solid {BORDER};
    border-left: 4px solid var(--kpi-color, {BLUE});
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    padding: 1.1rem 1.3rem;
    text-align: left;
}}
.kpi-val {{ font-size: 1.9rem; font-weight: 800; color: {TEXT_DARK}; line-height:1.1; }}
.kpi-lbl {{ font-size: 0.73rem; color: {TEXT_MUT}; margin-top: 5px; font-weight: 600;
            text-transform:uppercase; letter-spacing:.05em; }}
.kpi-sub {{ font-size: 0.70rem; color: {MUTED}; margin-top: 3px; }}

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
.abox-info     {{ background:#EFF6FF; border-left:4px solid {BLUE}; color:#1E3A8A; }}
.abox-success  {{ background:#ECFDF5; border-left:4px solid {EMERALD}; color:#064E3B; }}
.abox-warn     {{ background:#FFFBEB; border-left:4px solid {AMBER}; color:#78350F; }}
.abox-danger   {{ background:#FFF1F2; border-left:4px solid {RED}; color:#7F1D1D; }}
.abox-critical {{ background:#F5F3FF; border-left:4px solid {VIOLET}; color:#3B0764; }}
.abox-onset    {{ background:#F0FDF4; border-left:4px solid {EM_BRIGHT}; color:#14532D; }}

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
    background: linear-gradient(135deg, {SB} 0%, #1a2744 100%);
    border-radius: 16px; padding: 1.8rem 2.2rem 1.5rem;
    margin-bottom: 1.4rem; color: white;
    box-shadow: 0 4px 24px rgba(11,17,32,.35);
}}
.ph-inner {{
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 1rem;
}}
.ph-left  {{ flex: 1 1 auto; min-width: 0; }}
.ph-right {{ flex: 0 0 auto; text-align: right; min-width: 130px; }}
.ph-badges {{ margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.3rem; max-width: 100%; overflow: hidden; }}
.ph-eyebrow {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: .1em;
    color: {EM_BRIGHT}; font-weight: 700; margin-bottom: 6px;
}}
.page-header h1 {{ margin:0; font-size:1.6rem; font-weight:800; color:white; letter-spacing:-.02em; }}
.page-header p  {{ margin:0.3rem 0 0; font-size:0.84rem; color:#93C5FD; }}
.ph-badge {{
    display:inline-flex; align-items:center;
    background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.2); color:white;
    border-radius:20px; padding:3px 12px; font-size:0.72rem;
    font-weight:600; letter-spacing:.02em;
}}

/* ── Model result cards ──────────────────────────── */
.model-card {{
    border-radius: 12px; padding: 1.1rem 1.2rem;
    border: 1.5px solid {BORDER};
    background: {BG_CARD};
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    height: 100%; min-height: 220px;
}}
.model-card-best {{
    border-color: {EMERALD};
    box-shadow: 0 0 0 3px rgba(5,150,105,.1), 0 4px 20px rgba(5,150,105,.15);
}}
.model-auc {{
    font-size: 2.1rem; font-weight: 800; line-height: 1.1; letter-spacing: -.02em;
}}
.model-metric-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding: 3px 0; border-bottom: 1px solid {BORDER}; font-size: 0.8rem;
}}
.model-metric-row:last-child {{ border-bottom: none; }}

/* ── Scanner probability bar ─────────────────────── */
.prob-bar-wrap {{
    background: #F1F5F9; border-radius: 6px; height: 8px;
    overflow: hidden; margin-top: 3px;
}}
.prob-bar-fill {{
    height: 8px; border-radius: 6px;
}}

/* ── Prevent horizontal overflow ────────────────── */
.stApp, .block-container, [data-testid="stVerticalBlock"] {{
    max-width: 100% !important;
    overflow-x: hidden !important;
}}

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer {{ visibility: hidden; }}

/* ══ Responsive — tablet ══════════════════════════ */
@media (max-width: 1024px) {{
    .block-container {{ padding: 1.2rem 1.5rem 3rem !important; }}
    .page-header {{ padding: 1.4rem 1.5rem 1.2rem; }}
    .page-header h1 {{ font-size: 1.3rem; }}
    .kpi-val {{ font-size: 1.55rem; }}
    .v-prob  {{ font-size: 2.6rem; }}
    .model-auc {{ font-size: 1.7rem; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 1rem; font-size: 0.82rem; }}
}}

/* ══ Responsive — mobile ══════════════════════════ */
@media (max-width: 768px) {{
    .block-container {{ padding: 4rem 0.8rem 3rem !important; }}
    .page-header {{ padding: 1rem 1.1rem 0.9rem; border-radius: 12px; margin-bottom: 1rem; }}
    .ph-inner  {{ flex-direction: column; gap: 0.75rem; }}
    .ph-right  {{
        display: flex; flex-direction: row; align-items: center;
        text-align: left; min-width: unset; gap: 1rem;
        background: rgba(255,255,255,.07); border-radius: 10px;
        padding: 0.6rem 1rem; width: 100%;
    }}
    .ph-left   {{ width: 100%; }}
    .ph-badges {{ gap: 0.25rem; margin-top: 0.6rem; }}
    .page-header h1 {{ font-size: 1.05rem; }}
    .page-header p  {{ font-size: 0.74rem; }}
    .ph-badge {{ font-size: 0.65rem; padding: 2px 8px; }}
    .stTabs [data-baseweb="tab-list"] {{
        overflow-x: auto; flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch; scrollbar-width: none;
    }}
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.42rem 0.75rem; font-size: 0.76rem; white-space: nowrap; flex-shrink: 0; }}
    [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 0.5rem !important; }}
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
    }}
    .card {{ padding: 0.9rem 1rem; border-radius: 10px; }}
    .kpi {{ padding: 0.8rem 0.9rem; border-radius: 10px; }}
    .kpi-val {{ font-size: 1.3rem; }}
    .verdict-wrap {{ padding: 1.2rem 1rem; }}
    .v-prob {{ font-size: 2.1rem; }}
    .stDataFrame {{ overflow-x: auto !important; }}
}}

@media (max-width: 480px) {{
    .block-container {{ padding: 4rem 0.5rem 3rem !important; }}
    .page-header h1 {{ font-size: 0.98rem; }}
    .kpi-val {{ font-size: 1.2rem; }}
    .v-prob {{ font-size: 1.9rem; }}
    .stTabs [data-baseweb="tab"] {{ padding: 0.38rem 0.6rem; font-size: 0.72rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COUNTY COORDINATES  (all 79 verified)
# ══════════════════════════════════════════════════════════════════════════════
# Centroids computed from geoBoundaries SSD ADM2 shapefile (EPSG:32636 → WGS84)
# Source: geoBoundaries.org — geoBoundaries-SSD-ADM2.geojson (2024)
COUNTY_COORDS = {
    "Abiemnhom":      ( 9.5512,  29.1292),
    "Akobo":          ( 7.7990,  32.8548),
    "Aweil Centre":   ( 8.4180,  26.8951),
    "Aweil East":     ( 9.2428,  27.6131),
    "Aweil North":    ( 9.3349,  26.7293),
    "Aweil South":    ( 8.6680,  27.7360),
    "Aweil West":     ( 8.9306,  26.7233),
    "Awerial":        ( 6.1502,  31.2295),
    "Ayod":           ( 8.2914,  30.9805),
    "Baliet":         ( 9.4904,  32.4017),
    "Bor South":      ( 6.4655,  32.0048),
    "Budi":           ( 4.3603,  33.4464),
    "Canal/Pigi":     ( 9.0837,  31.4421),
    "Cueibet":        ( 7.0822,  29.2249),
    "Duk":            ( 7.6176,  31.1862),
    "Ezo":            ( 5.4590,  27.8969),
    "Fangak":         ( 9.0719,  30.6879),
    "Fashoda":        ( 9.9803,  31.8442),
    "Gogrial East":   ( 8.5971,  28.5886),
    "Gogrial West":   ( 8.5672,  28.1099),
    "Guit":           ( 9.1975,  30.0752),
    "Ibba":           ( 5.1012,  29.0619),
    "Ikotos":         ( 4.0288,  33.1314),
    "Juba":           ( 4.7171,  31.4833),
    "Jur River":      ( 7.5926,  28.0247),
    "Kajo-keji":      ( 3.8955,  31.4608),
    "Kapoeta East":   ( 5.1381,  34.6551),
    "Kapoeta North":  ( 5.3065,  33.4619),
    "Kapoeta South":  ( 4.5634,  33.6803),
    "Koch":           ( 8.6339,  29.8573),
    "Lafon":          ( 5.2492,  32.6714),
    "Lainya":         ( 4.2779,  30.8536),
    "Leer":           ( 8.1742,  30.2282),
    "Longochuk":      ( 9.2559,  33.5274),
    "Luakpiny/Nasir": ( 8.8354,  33.1623),
    "Maban":          (10.0238,  33.4602),
    "Magwi":          ( 3.9083,  32.2137),
    "Maiwut":         ( 8.7461,  33.8596),
    "Malakal":        ( 9.6585,  31.6428),
    "Manyo":          (11.1301,  32.3695),
    "Maridi":         ( 5.0966,  29.6847),
    "Mayendit":       ( 8.1224,  29.9196),
    "Mayom":          ( 9.0946,  29.2165),
    "Melut":          (10.3809,  32.5685),
    "Morobo":         ( 3.7410,  30.8335),
    "Mundri East":    ( 5.3078,  30.6825),
    "Mundri West":    ( 5.2139,  30.2082),
    "Mvolo":          ( 5.9180,  30.0323),
    "Nagero":         ( 6.4714,  27.7377),
    "Nyirol":         ( 8.5831,  32.1197),
    "Nzara":          ( 5.3177,  28.2079),
    "Panyijiar":      ( 7.5245,  30.3367),
    "Panyikang":      ( 9.4437,  31.4524),
    "Pariang":        ( 9.8152,  30.1188),
    "Pibor":          ( 6.3967,  33.5704),
    "Pochalla":       ( 7.1156,  33.8135),
    "Raga":           ( 8.6169,  25.5895),
    "Renk":           (11.3620,  32.9570),
    "Rubkona":        ( 9.2684,  29.6357),
    "Rumbek Centre":  ( 7.0184,  29.7839),
    "Rumbek East":    ( 6.7186,  30.0137),
    "Rumbek North":   ( 7.5246,  29.7042),
    "Tambura":        ( 6.0456,  27.1441),
    "Terekeka":       ( 5.6628,  31.3257),
    "Tonj East":      ( 7.8472,  29.2847),
    "Tonj North":     ( 8.2461,  28.9159),
    "Tonj South":     ( 7.0804,  28.7347),
    "Torit":          ( 4.3724,  32.5671),
    "Twic":           ( 9.1360,  28.3918),
    "Twic East":      ( 7.1008,  31.3227),
    "Ulang":          ( 8.5787,  32.8278),
    "Uror":           ( 7.6252,  32.1114),
    "Wau":            ( 7.2846,  27.2976),
    "Wulu":           ( 6.2002,  29.2086),
    "Yambio":         ( 5.1296,  28.5462),
    "Yei":            ( 4.2492,  30.3460),
    "Yirol East":     ( 6.7933,  30.8025),
    "Yirol West":     ( 6.3953,  30.4869),
    # Abyei — special administrative area, not in standard ADM2 dataset
    "Abyei":          ( 9.6000,  28.4200),
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Sidebar slider helper ─────────────────────────────────────────────────────
def sb_slider(label, unit, min_v, max_v, default, step, fmt, help_text, key):
    cur = st.session_state.get(key, default)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-top:10px;margin-bottom:2px">'
        f'<span style="color:#D6EAF8;font-size:0.83rem;font-weight:500">{label}</span>'
        f'<span style="background:rgba(52,211,153,.18);color:{EM_BRIGHT};font-size:0.85rem;'
        f'font-weight:700;padding:1px 9px;border-radius:5px;'
        f'border:1px solid rgba(52,211,153,.3)">{fmt.format(cur)}&thinsp;{unit}</span>'
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
    with open("model/metadata.json")        as f: meta      = json.load(f)
    with open("model/counties.json")        as f: counties  = json.load(f)
    with open("model/feature_stats.json")   as f: fstats    = json.load(f)
    with open("model/county_defaults.json") as f: cdefaults = json.load(f)
    hist    = pd.read_csv("model/county_flood_history.csv")
    monthly = pd.read_csv("model/monthly_flood_data.csv")
    return meta, counties, fstats, cdefaults, hist, monthly

model                                                    = load_model()
meta, counties, fstats, county_defaults, hist_df, monthly_df = load_artifacts()
FEATURES  = meta["features"]
THRESHOLD = meta["threshold"]

if "pred_prob"   not in st.session_state: st.session_state.pred_prob   = None
if "pred_county" not in st.session_state: st.session_state.pred_county = None
if "pred_month"  not in st.session_state: st.session_state.pred_month  = None
if "pred_inputs" not in st.session_state: st.session_state.pred_inputs = None


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
            "Risk Level": ("Critical Risk" if rate >= 0.12 else
                           "High Risk"     if rate >= 0.06 else
                           "Moderate Risk" if rate >= 0.03 else "Low Risk"),
            "risk_tier":  ("Critical" if rate >= 0.12 else
                           "High"     if rate >= 0.06 else
                           "Moderate" if rate >= 0.03 else "Low"),
            "bubble_size": max(8, rate * 200),
        })
    return pd.DataFrame(rows)


MAP_DF = build_map_df()

CHART_TEMPLATE = dict(
    font=dict(family="DM Sans, sans-serif", size=11, color=TEXT_DARK),
    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=True, gridcolor="#F2F3F4", zeroline=False,
               linecolor=BORDER, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="#F2F3F4", zeroline=False,
               linecolor=BORDER, tickfont=dict(size=10)),
)


@st.cache_data(show_spinner=False)
def scan_all_counties(rainfall, soil_moist, max_temp, min_temp, vpd, flood_prev, sel_month):
    """Predict flood probability for all 79 counties using sidebar climate + each county's terrain."""
    results = []
    for c in counties:
        cd = county_defaults.get(c, {})
        inputs = {
            "rainfall_mm":               rainfall,
            "soil_moisture_mm":          soil_moist,
            "max_temperature_celsius":   max_temp,
            "min_temperature_celsius":   min_temp,
            "vapor_pressure_deficit_kPa": vpd,
            "wetland_fraction": float(min(cd.get("wetland_fraction", 0.10), 0.92)),
            "elevation_m":      float(min(max(cd.get("elevation_m",   513.0), 392.0), 1145.0)),
            "slope_deg":        float(min(max(cd.get("slope_deg",       1.7),   0.9),   8.3)),
            "ndvi":             float(min(max(cd.get("ndvi",             0.55),  0.19),  0.85)),
            "flood_prev_month": flood_prev,
            "month":            sel_month,
        }
        p = run_prediction(inputs)
        tier, color, _, _, _, _ = risk_tier(p)
        results.append({
            "County": c,
            "Prob %": round(p * 100, 1),
            "_prob":  p,
            "Risk":   tier,
            "_color": color,
        })
    return pd.DataFrame(results).sort_values("Prob %", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"""
    <div style="background:rgba(52,211,153,.06);
                border:1px solid rgba(52,211,153,.15);border-radius:14px;
                padding:1.1rem 1.2rem 0.9rem;margin-bottom:1.1rem;text-align:center">
      <div style="color:white;font-weight:800;font-size:1.1rem;
                  letter-spacing:-.01em">SSD Flood EWS</div>
      <div style="color:{EM_BRIGHT};font-size:0.68rem;margin-top:3px;
                  text-transform:uppercase;letter-spacing:.12em">
        Early Warning System</div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.08);
                  font-size:0.68rem;color:#475569">
        79 counties · 2011–2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,.06);border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:0.8rem;font-size:0.78rem;
                color:#AED6F1;line-height:1.6">
      <b style="color:white">How to use:</b><br>
      1. Select a <b>county</b> and <b>month</b><br>
      2. Adjust climate &amp; terrain sliders<br>
      3. Predictions update live across all tabs
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sb-section">Step 1 — Location &amp; Time</p>', unsafe_allow_html=True)
    sel_county = st.selectbox("County", counties, index=counties.index("Malakal"),
                              help="Select the South Sudan county to assess flood risk for.")
    sel_month  = st.selectbox("Forecast Month", range(1,13),
                              format_func=lambda m: MONTH_NAMES[m-1], index=7,
                              help="The month to predict. Rainy season is June–October.")

    crow           = hist_df[hist_df["county"] == sel_county]
    hist_rate_sb   = float(crow["flood_rate"].iloc[0])   if not crow.empty else 0.0
    hist_events_sb = int(crow["flood_events"].iloc[0])   if not crow.empty else 0

    cd        = county_defaults.get(sel_county, {})
    # Round each default to the slider's step to avoid "values conflict with step" warning
    def_rain  = round(float(min(cd.get("rainfall_mm",              83.0), 336.5)),  0)
    def_sm    = round(round(float(min(cd.get("soil_moisture_mm",   29.0), 224.9)) / 0.5) * 0.5, 1)
    def_maxt  = round(float(min(max(cd.get("max_temperature_celsius", 35.0), 26.2), 43.1)), 1)
    def_mint  = round(float(min(max(cd.get("min_temperature_celsius", 21.0), 14.7), 27.9)), 1)
    def_vpd   = round(round(float(min(max(cd.get("vapor_pressure_deficit_kPa", 2.3), 0.4), 5.0)) / 0.05) * 0.05, 2)
    def_wet   = round(float(min(cd.get("wetland_fraction",          0.10), 0.92)),  2)
    def_elev  = round(float(min(max(cd.get("elevation_m",          513.0), 392.0), 1145.0)), 0)
    def_slope = round(float(min(max(cd.get("slope_deg",              1.7),   0.9),   8.3)),  1)
    def_ndvi  = round(float(min(max(cd.get("ndvi",                   0.55),  0.19),  0.85)), 2)

    rank_df_sb = hist_df.sort_values("flood_rate", ascending=False).reset_index(drop=True)
    crank = (rank_df_sb[rank_df_sb["county"]==sel_county].index[0]+1
             if sel_county in rank_df_sb["county"].values else "—")
    st.markdown(f"""
    <div style="background:rgba(0,0,0,.2);border-radius:7px;padding:0.6rem 0.9rem;
                margin:0.4rem 0 0.8rem;font-size:0.76rem;color:#D6EAF8;line-height:1.8">
      <b style="color:white">{sel_county}</b><br>
      Historical flood rate: <b style="color:{EM_BRIGHT}">{hist_rate_sb*100:.1f}%</b>
      &nbsp;({hist_events_sb} events)<br>
      County rank: <b style="color:{EM_BRIGHT}">#{crank} of 79</b><br>
      <span style="font-size:0.68rem;opacity:.75">Sliders pre-filled from dataset medians.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sb-section">Step 2 — Climate Conditions</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.3rem">'
        f'Adjust to match current or forecast conditions. '
        f'<b style="color:{EM_BRIGHT}">Current value in green.</b></div>',
        unsafe_allow_html=True)

    ckey = sel_county.replace(" ", "_").replace("/", "_")

    rainfall   = sb_slider("Rainfall",              "mm",   0.0, 336.5, def_rain,  1.0,
                           "{:.0f}", "Total monthly rainfall. Higher → more runoff → higher flood risk.",
                           f"rain_{ckey}")
    soil_moist = sb_slider("Soil Moisture",         "mm",   0.8, 224.9, def_sm,    0.5,
                           "{:.1f}", "Saturated soil has less capacity to absorb rain → higher flood risk.",
                           f"sm_{ckey}")
    max_temp   = sb_slider("Max Temperature",       "°C",  26.2,  43.1, def_maxt,  0.1,
                           "{:.1f}", "Daily high temperature. Drives evaporation and water cycle.",
                           f"maxt_{ckey}")
    min_temp   = sb_slider("Min Temperature",       "°C",  14.7,  27.9, def_mint,  0.1,
                           "{:.1f}", "Daily low temperature. Max−Min range is a model feature.",
                           f"mint_{ckey}")
    vpd        = sb_slider("Vapour Pressure Deficit","kPa",  0.4,   5.0, def_vpd,  0.05,
                           "{:.2f}", "Low VPD (< 1.5 kPa) = humid air = higher flood risk.",
                           f"vpd_{ckey}")

    st.markdown('<p class="sb-section">Step 3 — Terrain &amp; Land Cover</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.3rem">'
        'Geographic properties — relatively fixed. Adjust only with updated measurements.</div>',
        unsafe_allow_html=True)

    wetland   = sb_slider("Wetland Fraction",  "/ 1.0", 0.0,   0.92, def_wet,   0.01,
                          "{:.2f}", "Share of county covered by wetland. 0.92 = mostly wetland.",
                          f"wet_{ckey}")
    elevation = sb_slider("Elevation",         "m",   392.0, 1145.0, def_elev,   1.0,
                          "{:.0f}", "Counties below 430 m are most at risk.",
                          f"elev_{ckey}")
    slope     = sb_slider("Slope",             "°",     0.9,    8.3, def_slope,  0.1,
                          "{:.1f}", "Flat land drains slowly → water pools. Steep terrain sheds faster.",
                          f"slope_{ckey}")
    ndvi_val  = sb_slider("NDVI (Vegetation)", "",      0.19,  0.85, def_ndvi,  0.01,
                          "{:.2f}", "Vegetation index. > 0.6 = dense cover.",
                          f"ndvi_{ckey}")

    st.markdown('<p class="sb-section">Step 4 — Prior Month Status</p>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.74rem;color:#7FB3D3;margin-bottom:0.4rem">'
                'Was there active flooding last month? Strongest single predictor.</div>',
                unsafe_allow_html=True)
    flood_prev = 1 if st.toggle("Flooding occurred last month", value=False,
                                help="Dramatically increases predicted risk when ON.") else 0
    if flood_prev:
        st.markdown(f'<div style="font-size:0.74rem;color:#F1948A;margin-top:0.3rem">'
                    f'⚠ Prior flood ON — probability significantly elevated.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#7FB3D3;text-align:center;line-height:1.7">
      Model: <b style="color:white">{meta['best_model_name']}</b><br>
      AUC-ROC: <b style="color:{EM_BRIGHT}">{meta['test_metrics'][meta['best_model_name']]['auc_roc']}</b><br>
      Decision threshold: {THRESHOLD*100:.0f}%<br>
      <span style="font-size:0.68rem;opacity:.7">Trained on 14,220 obs · 79 counties · 2011–2025</span>
    </div>
    """, unsafe_allow_html=True)


# ── Build current prediction ──────────────────────────────────────────────────
raw_inputs = dict(
    rainfall_mm=rainfall, soil_moisture_mm=soil_moist,
    max_temperature_celsius=max_temp, min_temperature_celsius=min_temp,
    vapor_pressure_deficit_kPa=vpd, wetland_fraction=wetland,
    elevation_m=elevation, slope_deg=slope, ndvi=ndvi_val,
    flood_prev_month=flood_prev, month=sel_month,
)
prob                                          = run_prediction(raw_inputs)
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
      <div class="ph-eyebrow">Early Warning System · South Sudan</div>
      <h1>County-Level Flood Prediction</h1>
      <p>Machine learning · 79 counties · 2011–2025 · Viewing:
         <b style="color:white">{sel_county}</b>, {MONTH_NAMES[sel_month-1]}</p>
      <div class="ph-badges">
        <span class="ph-badge">Logistic Regression</span>
        <span class="ph-badge">AUC-ROC {_lr_auc_hdr}</span>
        <span class="ph-badge">79 Counties</span>
        <span class="ph-badge">2011–2025</span>
      </div>
    </div>
    <div class="ph-right">
      <div style="font-size:0.68rem;color:{EM_BRIGHT};text-transform:uppercase;
                  letter-spacing:.08em;font-weight:600">Model AUC-ROC</div>
      <div style="font-size:2.6rem;font-weight:800;color:white;line-height:1;
                  letter-spacing:-.03em">{_lr_auc_hdr}</div>
      <div style="font-size:0.7rem;color:#93C5FD;margin-top:3px">
        Hold-out test 2024–2025
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS  (5 tabs)
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Flood Prediction  ",
    "  Geographic Map  ",
    "  County Scanner  ",
    "  Historical Analysis  ",
    "  Model Science  ",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1  —  FLOOD PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── 4 KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")
    rank_df = hist_df.sort_values("flood_rate", ascending=False).reset_index(drop=True)
    county_rank = (rank_df[rank_df["county"]==sel_county].index[0]+1
                   if sel_county in rank_df["county"].values else "—")

    for col, val, lbl, sub, kpi_col in [
        (k1, f"{prob*100:.1f}%", "Flood Probability",
         f"{'⚠ Above' if prob>=THRESHOLD else '✓ Below'} {THRESHOLD*100:.0f}% threshold", r_col),
        (k2, r_label, "Risk Level",
         f"{'FLOOD ALERT' if prob>=THRESHOLD else 'No flood predicted'}", r_col),
        (k3, f"{hist_rate*100:.1f}%", "Historical Rate",
         f"{hist_events} events · 180 months", BLUE),
        (k4, f"#{county_rank}", "County Rank",
         "of 79 by historical flood rate", BLUE),
    ]:
        col.markdown(f"""
        <div class="kpi" style="--kpi-color:{kpi_col}">
          <div class="kpi-val" style="color:{kpi_col}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Status card ───────────────────────────────────────────────────────────
    diff      = prob - hist_rate
    above_thr = prob >= THRESHOLD
    diff_sign = "+" if diff >= 0 else "−"
    diff_abs  = abs(diff) * 100
    status_bg   = "#FFF1F2" if above_thr else ("#ECFDF5" if prob < 0.15 else "#EFF6FF")
    status_bdr  = RED if above_thr else (EMERALD if prob < 0.15 else BLUE)
    status_icon = "⚑ FLOOD PREDICTED" if above_thr else ("✓ No Flood Predicted" if prob < 0.50 else "⚠ Elevated Risk — Below Threshold")
    status_col  = RED if above_thr else (EMERALD if prob < 0.15 else BLUE)

    st.markdown(f"""
    <div style="background:{status_bg};border:1.5px solid {status_bdr};
                border-radius:12px;padding:1rem 1.4rem;margin:0.5rem 0 0.8rem;
                display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
      <div style="flex:0 0 auto">
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:.1em;
                    font-weight:700;color:{status_col};opacity:.75">Decision</div>
        <div style="font-size:1.05rem;font-weight:800;color:{status_col};
                    white-space:nowrap">{status_icon}</div>
      </div>
      <div style="width:1px;height:40px;background:{status_bdr};opacity:.3;flex:0 0 auto"></div>
      <div style="flex:1;min-width:180px">
        <div style="font-size:0.8rem;color:{TEXT_DARK};line-height:1.6">
          <b>{sel_county}</b> · {MONTH_NAMES[sel_month-1]} · {prob*100:.1f}% probability
        </div>
        <div style="font-size:0.76rem;color:{MUTED};margin-top:2px">
          {diff_sign}{diff_abs:.1f}% vs county average ({hist_rate*100:.1f}%) &nbsp;·&nbsp;
          Threshold: {THRESHOLD*100:.0f}%
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Onset detection banner (only when flood_prev == 0) ───────────────────
    if flood_prev == 0:
        onset_inputs = raw_inputs.copy()
        onset_inputs["flood_prev_month"] = 1
        onset_prob = run_prediction(onset_inputs)
        delta_pct  = (onset_prob - prob) * 100
        st.markdown(
            f'<div style="background:#F0FDF4;border:1px solid {EM_BRIGHT};border-radius:10px;'
            f'padding:0.65rem 1.1rem;margin-bottom:0.8rem;font-size:0.82rem;color:#14532D">'
            f'<b>Onset sensitivity:</b> If last month had flooding, probability rises '
            f'<b>{prob*100:.1f}% → {onset_prob*100:.1f}%</b> (+{delta_pct:.1f} pp). '
            f'Toggle <em>"Flooding occurred last month"</em> in the sidebar to model that scenario.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── RESULT HERO: full-width card split into left info + right gauge ───────
    decision_label = "FLOOD PREDICTED" if prob >= THRESHOLD else "NO FLOOD PREDICTED"
    decision_bg    = r_col if prob >= THRESHOLD else EMERALD
    diff_sign      = "+" if prob >= hist_rate else "−"
    diff_abs_hero  = abs(prob - hist_rate) * 100

    hero_left, hero_right = st.columns([1.35, 1], gap="medium")

    with hero_left:
        _action_bg = "#ECFDF5" if prob < 0.25 else "#FFFBEB" if prob < 0.50 else "#FFF1F2" if prob < 0.75 else "#F5F3FF"
        _delta_col = "#922B21" if prob > hist_rate else "#1E8449"
        st.markdown(
            f'<div style="background:{BG_CARD};border-radius:16px;border:1.5px solid {r_col}44;'
            f'box-shadow:0 2px 6px rgba(0,0,0,.05),0 8px 32px {r_col}14;padding:0;overflow:hidden">'
            f'<div style="background:{r_col};padding:0.6rem 1.5rem;display:flex;align-items:center;justify-content:space-between">'
            f'<div style="color:white;font-size:0.8rem;font-weight:700;letter-spacing:.04em">{sel_county} &nbsp;·&nbsp; {MONTH_NAMES[sel_month-1]}</div>'
            f'<span style="background:rgba(255,255,255,.2);color:white;border-radius:20px;padding:2px 12px;font-size:0.7rem;font-weight:700;letter-spacing:.06em">{r_label.upper()}</span>'
            f'</div>'
            f'<div style="padding:1.4rem 1.5rem">'
            f'<div style="display:flex;align-items:flex-end;gap:0.8rem;margin-bottom:0.6rem">'
            f'<div style="font-size:4.2rem;font-weight:800;line-height:1;letter-spacing:-.04em;color:{r_col}">{prob*100:.1f}%</div>'
            f'<div style="padding-bottom:0.4rem">'
            f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:{MUTED};margin-bottom:5px">Flood Probability</div>'
            f'<span style="background:{decision_bg};color:white;border-radius:6px;padding:4px 14px;font-size:0.78rem;font-weight:800;letter-spacing:.05em;display:inline-block">{decision_label}</span>'
            f'</div>'
            f'</div>'
            f'<div style="font-size:0.84rem;color:{MUTED};line-height:1.65;border-top:1px solid {BORDER};padding-top:0.85rem;margin-bottom:1rem">'
            f'The model assigns a <b style="color:{TEXT_DARK}">{prob*100:.1f}%</b> chance that conditions in '
            f'<b style="color:{TEXT_DARK}">{sel_county}</b> during <b style="color:{TEXT_DARK}">{MONTH_NAMES[sel_month-1]}</b> '
            f'are consistent with a flood month — based on {rainfall:.0f}\u202fmm rainfall, {soil_moist:.0f}\u202fmm soil moisture, '
            f'and this county\'s terrain. A probability above <b style="color:{TEXT_DARK}">{THRESHOLD*100:.0f}%</b> triggers a flood alert.'
            f'</div>'
            f'<div style="display:flex;gap:0.55rem;flex-wrap:wrap;margin-bottom:1rem">'
            f'<div style="flex:1;min-width:85px;background:#F8FAFC;border-radius:10px;padding:0.6rem 0.85rem;border:1px solid {BORDER}">'
            f'<div style="font-size:0.64rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:{MUTED};margin-bottom:3px">Historical Rate</div>'
            f'<div style="font-size:1.2rem;font-weight:800;color:{TEXT_DARK}">{hist_rate*100:.1f}%</div>'
            f'<div style="font-size:0.68rem;color:{MUTED}">{hist_events} of 180 months</div>'
            f'</div>'
            f'<div style="flex:1;min-width:85px;background:#F8FAFC;border-radius:10px;padding:0.6rem 0.85rem;border:1px solid {BORDER}">'
            f'<div style="font-size:0.64rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:{MUTED};margin-bottom:3px">vs County Avg</div>'
            f'<div style="font-size:1.2rem;font-weight:800;color:{_delta_col}">{diff_sign}{diff_abs_hero:.1f}pp</div>'
            f'<div style="font-size:0.68rem;color:{MUTED}">percentage points</div>'
            f'</div>'
            f'<div style="flex:1;min-width:85px;background:#F8FAFC;border-radius:10px;padding:0.6rem 0.85rem;border:1px solid {BORDER}">'
            f'<div style="font-size:0.64rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:{MUTED};margin-bottom:3px">County Rank</div>'
            f'<div style="font-size:1.2rem;font-weight:800;color:{TEXT_DARK}">#{county_rank}</div>'
            f'<div style="font-size:0.68rem;color:{MUTED}">of 79 by flood rate</div>'
            f'</div>'
            f'<div style="flex:1;min-width:85px;background:#F8FAFC;border-radius:10px;padding:0.6rem 0.85rem;border:1px solid {BORDER}">'
            f'<div style="font-size:0.64rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:{MUTED};margin-bottom:3px">Alert Threshold</div>'
            f'<div style="font-size:1.2rem;font-weight:800;color:{TEXT_DARK}">{THRESHOLD*100:.0f}%</div>'
            f'<div style="font-size:0.68rem;color:{MUTED}">decision boundary</div>'
            f'</div>'
            f'</div>'
            f'<div style="background:{_action_bg};border-left:4px solid {r_col};border-radius:8px;padding:0.75rem 1rem;font-size:0.83rem;color:{TEXT_DARK};line-height:1.6">'
            f'<div style="font-size:0.67rem;text-transform:uppercase;letter-spacing:.09em;font-weight:700;color:{r_col};margin-bottom:4px">Recommended Action</div>'
            f'{r_action}'
            f'</div>'
            f'<div style="margin-top:0.65rem;font-size:0.71rem;color:{MUTED}">'
            f'Model: {meta["best_model_name"]} &nbsp;·&nbsp; Threshold: {THRESHOLD*100:.0f}% &nbsp;·&nbsp; 14,220 observations · 2011–2025'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(f"""
        <div style="background:{BG_CARD};border-radius:16px;border:1px solid {BORDER};
                    box-shadow:0 2px 6px rgba(0,0,0,.05);padding:1.2rem 1rem 0.5rem;
                    height:100%">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:.08em;
                      text-transform:uppercase;color:{MUTED};border-bottom:1px solid {BORDER};
                      padding-bottom:0.55rem;margin-bottom:0.2rem">Probability Gauge</div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number=dict(suffix="%", font=dict(size=44, color=r_col, family="DM Sans")),
            gauge=dict(
                axis=dict(range=[0,100], tickwidth=1, tickcolor="#CBD5E1",
                          tickvals=[0,25,50,75,100],
                          ticktext=["0","25","50","75","100"]),
                bar=dict(color=r_col, thickness=0.3),
                bgcolor="white", bordercolor=BORDER, borderwidth=1,
                steps=[
                    dict(range=[0,25],   color="#ECFDF5"),
                    dict(range=[25,50],  color="#FEF9E7"),
                    dict(range=[50,75],  color="#FFF1F2"),
                    dict(range=[75,100], color="#F5F3FF"),
                ],
                threshold=dict(
                    line=dict(color=TEXT_DARK, width=3),
                    thickness=0.85, value=THRESHOLD*100,
                ),
            ),
        ))
        fig_gauge.update_layout(
            height=310, margin=dict(t=20, b=50, l=20, r=20),
            paper_bgcolor=BG_CARD, font=dict(family="DM Sans"),
            annotations=[
                dict(text=f"Alert threshold: {THRESHOLD*100:.0f}%",
                     x=0.5, y=-0.18, showarrow=False,
                     font=dict(size=11, color=MUTED)),
                dict(text=r_label,
                     x=0.5, y=0.28, showarrow=False,
                     font=dict(size=16, color=r_col, family="DM Sans")),
            ],
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))

        # Risk scale legend
        st.markdown(f"""
          <div style="display:flex;gap:4px;justify-content:center;
                      padding:0 0.5rem 0.8rem;flex-wrap:wrap">
            {"".join([
              f'<div style="flex:1;text-align:center;padding:5px 4px;border-radius:7px;'
              f'background:{bg};border:1px solid {bdr}">'
              f'<div style="font-size:0.65rem;font-weight:700;color:{tc}">{lbl}</div>'
              f'<div style="font-size:0.62rem;color:{tc};opacity:.75">{rng}</div></div>'
              for lbl, rng, bg, bdr, tc in [
                ("Low",      "0–25%",   "#ECFDF5", "#059669", "#064E3B"),
                ("Moderate", "25–50%",  "#FEF9E7", "#D97706", "#78350F"),
                ("High",     "50–75%",  "#FFF1F2", "#DC2626", "#7F1D1D"),
                ("Critical", "75–100%", "#F5F3FF", "#7C3AED", "#3B0764"),
              ]
            ])}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature Importance — full width ──────────────────────────────────────
    st.markdown('<div class="card" style="margin-top:0.5rem">', unsafe_allow_html=True)
    fi_left, fi_right = st.columns([1.8, 1], gap="small")

    with fi_left:
        st.markdown('<div class="card-title">What Drives This Prediction — Model Feature Importance</div>',
                    unsafe_allow_html=True)
        FEAT_LABEL = {
            "rainfall_mm":               "Rainfall (mm)",
            "soil_moisture_mm":          "Soil Moisture (mm)",
            "max_temperature_celsius":   "Max Temperature (°C)",
            "min_temperature_celsius":   "Min Temperature (°C)",
            "vapor_pressure_deficit_kPa":"Vapour Pressure Deficit (kPa)",
            "wetland_fraction":          "Wetland Fraction",
            "elevation_m":               "Elevation (m)",
            "slope_deg":                 "Slope (°)",
            "ndvi":                      "NDVI (Vegetation)",
            "flood_prev_month":          "Flood Previous Month ★",
            "temp_range":                "Temperature Range (°C)",
            "wetness_index":             "Wetness Index",
            "rain_wetland":              "Rainfall × Wetland",
            "month_sin":                 "Seasonal Signal (sin)",
            "month_cos":                 "Seasonal Signal (cos)",
        }
        fi      = meta.get("feature_importance", {})
        fi_sort = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        labels  = [FEAT_LABEL.get(k, k) for k, _ in fi_sort]
        vals    = [v for _, v in fi_sort]
        avg_fi  = np.mean(vals)
        bar_colors = [r_col if v >= avg_fi else "#CBD5E1" for v in vals]

        fig_fi = go.Figure(go.Bar(
            x=vals[::-1], y=labels[::-1], orientation="h",
            marker=dict(color=bar_colors[::-1], line=dict(width=0)),
            text=[f"{v*100:.1f}%" for v in vals[::-1]],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_MUT),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f} (%{text})<extra></extra>",
            cliponaxis=False,
        ))
        fig_fi.add_vline(x=avg_fi, line_dash="dot", line_color="#94A3B8", line_width=1.5,
                         annotation_text="avg", annotation_position="top",
                         annotation_font=dict(size=9, color=MUTED))
        fig_fi.update_layout(
            height=420, margin=dict(t=5, b=5, l=5, r=70),
            xaxis=dict(title=None, showgrid=True, gridcolor="#F1F5F9",
                       zeroline=False, tickfont=dict(size=9.5)),
            yaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_fi, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))

    with fi_right:
        st.markdown('<div class="card-title">Top Features</div>', unsafe_allow_html=True)
        for i, (feat, imp) in enumerate(fi_sort[:5]):
            lbl  = FEAT_LABEL.get(feat, feat)
            pct  = imp * 100
            col_ = r_col if imp >= avg_fi else "#64748B"
            bar_w = int(pct / fi_sort[0][1] * 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.55rem;'
                f'padding:0.55rem 0.8rem;background:#F8FAFC;border-radius:9px;border:1px solid {BORDER};'
                f'box-sizing:border-box;width:100%">'
                f'<div style="font-size:0.95rem;font-weight:800;color:{col_};min-width:24px;text-align:center">#{i+1}</div>'
                f'<div style="flex:1;min-width:0;overflow:hidden">'
                f'<div style="font-size:0.78rem;font-weight:600;color:{TEXT_DARK};line-height:1.3;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{lbl}</div>'
                f'<div style="height:4px;background:#E2E8F0;border-radius:3px;margin-top:5px">'
                f'<div style="height:4px;width:{bar_w}%;background:{col_};border-radius:3px"></div>'
                f'</div>'
                f'</div>'
                f'<div style="font-size:0.85rem;font-weight:700;color:{col_};white-space:nowrap;padding-left:4px">{pct:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="font-size:0.77rem;color:{MUTED};line-height:1.6;margin-top:0.7rem;padding:0.8rem;'
            f'background:#F8FAFC;border-radius:9px;border:1px solid {BORDER}">'
            f'<b style="color:{TEXT_DARK}">What the bars show:</b> Normalised absolute coefficients '
            f'from the Logistic Regression. Higher = stronger contribution to flood/no-flood '
            f'discrimination across all 14,220 training months.<br><br>'
            f'<span style="color:{r_col}">&#9632;</span> <b>Highlighted</b> = above average importance. '
            f'&#9733; <b>Flood Prev Month</b> = strongest temporal signal.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sensitivity grid (2 × 2) ─────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Sensitivity Analysis — How Each Variable Affects Flood Probability</div>',
                unsafe_allow_html=True)

    sens_specs = [
        ("Rainfall (mm)",              "rainfall_mm",               0.0,   336.5, rainfall,   "#1D4ED8"),
        ("Soil Moisture (mm)",         "soil_moisture_mm",          0.8,   224.9, soil_moist, "#0EA5E9"),
        ("Vapour Pressure Deficit (kPa)", "vapor_pressure_deficit_kPa", 0.4, 5.0, vpd,       "#7C3AED"),
        ("Wetland Fraction",           "wetland_fraction",          0.0,   0.92,  wetland,    "#059669"),
    ]

    fig_sens = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.12, vertical_spacing=0.18,
        subplot_titles=[s[0] for s in sens_specs],
    )
    for idx, (xlabel, feat_key, xmin, xmax, cur_val, line_col) in enumerate(sens_specs):
        row, col = divmod(idx, 2)
        xs    = np.linspace(xmin, xmax, 60)
        probs = []
        for xv in xs:
            tmp = raw_inputs.copy(); tmp[feat_key] = xv
            probs.append(run_prediction(tmp))

        fig_sens.add_hrect(
            y0=THRESHOLD*100, y1=100,
            fillcolor="rgba(220,38,38,.05)", line_width=0,
            row=row+1, col=col+1,
        )
        fig_sens.add_trace(go.Scatter(
            x=xs, y=[p*100 for p in probs],
            mode="lines", line=dict(color=line_col, width=2.2),
            fill="tozeroy", fillcolor=f"rgba({int(line_col[1:3],16)},{int(line_col[3:5],16)},{int(line_col[5:7],16)},0.07)",
            showlegend=False,
            hovertemplate=f"{xlabel}: %{{x:.2f}}<br>Probability: %{{y:.1f}}%<extra></extra>",
        ), row=row+1, col=col+1)
        # Current value marker
        cur_prob = run_prediction({**raw_inputs, feat_key: cur_val}) * 100
        fig_sens.add_trace(go.Scatter(
            x=[cur_val], y=[cur_prob],
            mode="markers", marker=dict(color=line_col, size=9,
                                         line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=f"Current: {cur_val:.2f}<br>Prob: {cur_prob:.1f}%<extra></extra>",
        ), row=row+1, col=col+1)
        # Threshold line
        fig_sens.add_hline(y=THRESHOLD*100, line_dash="dot",
                           line_color=RED, line_width=1.2, row=row+1, col=col+1)

    fig_sens.update_layout(
        height=420, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(family="DM Sans", size=10),
        margin=dict(t=35, b=10, l=10, r=10),
    )
    fig_sens.update_yaxes(range=[0, 105], ticksuffix="%",
                          showgrid=True, gridcolor="#F2F3F4",
                          tickfont=dict(size=9))
    fig_sens.update_xaxes(showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9))
    st.plotly_chart(fig_sens, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
    st.markdown(
        f'<div style="font-size:0.75rem;color:{MUTED};margin-top:-0.3rem">'
        f'Each chart sweeps one variable while holding all others at their current sidebar values. '
        f'Red dots mark your current settings. The dashed red line is the flood decision threshold '
        f'({THRESHOLD*100:.0f}%).</div>',
        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input summary + CSV download ─────────────────────────────────────────
    b1, b2 = st.columns([1.4, 1], gap="medium")
    with b1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Input Values Summary</div>', unsafe_allow_html=True)
        inp = pd.DataFrame({
            "Variable": ["Rainfall","Soil Moisture","Max Temp","Min Temp",
                         "Vapour Pressure Deficit","Wetland Fraction","Elevation",
                         "Slope","NDVI","Flood Prev. Month"],
            "Value":    [f"{rainfall:.1f} mm", f"{soil_moist:.1f} mm",
                         f"{max_temp:.1f} °C",  f"{min_temp:.1f} °C",
                         f"{vpd:.2f} kPa",      f"{wetland:.2f}",
                         f"{elevation:.0f} m",  f"{slope:.1f} °",
                         f"{ndvi_val:.2f}",
                         "Yes" if flood_prev else "No"],
        })
        st.dataframe(inp, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Export Prediction</div>', unsafe_allow_html=True)
        export_df = pd.DataFrame({
            "Field": ["County","Month","Flood Probability","Risk Level","Flood Predicted",
                      "Decision Threshold","Historical Flood Rate","County Rank",
                      "Rainfall (mm)","Soil Moisture (mm)","Max Temp (°C)","Min Temp (°C)",
                      "VPD (kPa)","Wetland Fraction","Elevation (m)","Slope (°)",
                      "NDVI","Flood Prev Month"],
            "Value": [sel_county, MONTH_NAMES[sel_month-1],
                      f"{prob*100:.2f}%", r_label,
                      "Yes" if prob >= THRESHOLD else "No",
                      f"{THRESHOLD*100:.0f}%", f"{hist_rate*100:.1f}%",
                      f"#{county_rank} of 79",
                      f"{rainfall:.1f}", f"{soil_moist:.1f}",
                      f"{max_temp:.1f}", f"{min_temp:.1f}",
                      f"{vpd:.2f}", f"{wetland:.2f}",
                      f"{elevation:.0f}", f"{slope:.1f}",
                      f"{ndvi_val:.2f}", "Yes" if flood_prev else "No"],
        })
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            label="Download prediction as CSV",
            data=csv_bytes,
            file_name=f"flood_prediction_{sel_county.replace(' ','_')}_{MONTH_NAMES[sel_month-1]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(f"""
        <div style="margin-top:1rem;font-size:0.8rem;color:{MUTED};line-height:1.7">
          <b style="color:{TEXT_DARK}">Export includes:</b><br>
          Predicted probability, risk tier, flood/no-flood decision, all input
          parameter values, and county metadata. Useful for logging predictions
          or sharing with field teams.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2  —  GEOGRAPHIC MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Geographic Flood Risk Map — All 79 Counties")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    tc = MAP_DF["risk_tier"].value_counts()
    mk1, mk2, mk3, mk4 = st.columns(4, gap="small")
    for col, tier, color in [
        (mk1, "Critical", RED),
        (mk2, "High",     ORANGE),
        (mk3, "Moderate", AMBER),
        (mk4, "Low",      EMERALD),
    ]:
        n = int(tc.get(tier, 0))
        col.markdown(f"""
        <div class="kpi" style="--kpi-color:{color}">
          <div class="kpi-val" style="color:{color}">{n}</div>
          <div class="kpi-lbl">{tier} Risk Counties</div>
          <div class="kpi-sub">{n/79*100:.0f}% of 79 counties</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive scatter_mapbox ─────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">Interactive Map — Historical Flood Rate by County (hover for details)</div>',
        unsafe_allow_html=True)

    RISK_COLOR_MAP = {
        "Critical Risk": "#DC2626",
        "High Risk":     "#EA580C",
        "Moderate Risk": "#D97706",
        "Low Risk":      "#059669",
    }

    # Highlight selected county
    map_df2 = MAP_DF.copy()
    map_df2["Selected"] = map_df2["county"] == sel_county
    map_df2["hover_text"] = map_df2.apply(
        lambda r: (
            f"<b>{r['county']}</b><br>"
            f"Historical flood rate: {r['flood_pct']:.1f}%<br>"
            f"Flood events: {r['flood_events']}/180 months<br>"
            f"Risk tier: {r['risk_tier']}"
        ), axis=1
    )

    fig_map = px.scatter_map(
        map_df2,
        lat="lat", lon="lon",
        color="Risk Level",
        size="bubble_size",
        size_max=28,
        color_discrete_map=RISK_COLOR_MAP,
        hover_name="county",
        hover_data={"lat": False, "lon": False, "bubble_size": False,
                    "flood_pct": ":.1f", "flood_events": True, "risk_tier": True,
                    "Risk Level": False},
        map_style="carto-positron",
        zoom=5.0,
        center={"lat": 7.2, "lon": 30.5},
    )
    # Add selected county as highlighted ring (outer glow + inner dot)
    if sel_county in COUNTY_COORDS:
        sel_lat, sel_lon = COUNTY_COORDS[sel_county]
        fig_map.add_trace(go.Scattermap(
            lat=[sel_lat], lon=[sel_lon],
            mode="markers",
            marker=dict(size=36, color=EM_BRIGHT, opacity=0.30),
            showlegend=False,
            hovertemplate=f"<b>{sel_county}</b> (selected)<extra></extra>",
        ))
        fig_map.add_trace(go.Scattermap(
            lat=[sel_lat], lon=[sel_lon],
            mode="markers",
            marker=dict(size=18, color=EM_BRIGHT, opacity=0.9),
            name=f"▶ {sel_county} (selected)",
            hovertemplate=f"<b>{sel_county}</b> (selected)<extra></extra>",
        ))

    fig_map.update_layout(
        height=520,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor=BG_CARD,
        legend=dict(
            orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1,
            font=dict(family="DM Sans", size=11),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config=dict(displayModeBar=True))
    st.markdown(
        f'<div style="font-size:0.74rem;color:{MUTED};margin-top:0.3rem">'
        f'Bubble size is proportional to historical flood rate. The green ring marks your selected county '
        f'(<b>{sel_county}</b>). Hover over any dot for county details. '
        f'Colours: green = Low (&lt;3%), amber = Moderate (3–6%), orange = High (6–12%), red = Critical (&gt;12%).</div>',
        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Side-by-side: donut + top 15 ─────────────────────────────────────────
    mp1, mp2 = st.columns([1, 1.8], gap="medium")

    with mp1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Risk Breakdown</div>', unsafe_allow_html=True)
        tier_counts = MAP_DF["risk_tier"].value_counts().reindex(
            ["Critical","High","Moderate","Low"], fill_value=0)
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts.index.tolist(),
            values=tier_counts.values,
            marker_colors=[RED, ORANGE, AMBER, EMERALD],
            hole=0.54,
            textinfo="label+value",
            textfont=dict(size=10),
            hovertemplate="<b>%{label} Risk</b><br>%{value} counties (%{percent})<extra></extra>",
            sort=False, direction="clockwise",
        ))
        fig_donut.update_layout(
            height=230, margin=dict(t=5,b=5,l=5,r=5),
            paper_bgcolor=BG_CARD, showlegend=False,
            font=dict(family="DM Sans"),
            annotations=[dict(text="79<br>counties", x=0.5, y=0.5,
                              font=dict(size=11, color=TEXT_DARK), showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
        st.markdown("</div>", unsafe_allow_html=True)

    with mp2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top 15 Highest-Risk Counties</div>', unsafe_allow_html=True)
        top15 = hist_df.nlargest(15, "flood_rate")[["county","flood_rate","flood_events"]].copy()
        top15["Flood Rate"] = top15["flood_rate"].apply(lambda x: f"{x*100:.1f}%")
        top15["Risk"] = top15["flood_rate"].apply(
            lambda r: "Critical" if r >= 0.12 else "High" if r >= 0.06 else "Moderate")
        top15 = top15.rename(columns={"county":"County","flood_events":"Flood Events"})
        top15 = top15[["County","Flood Rate","Flood Events","Risk"]]
        st.dataframe(top15, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3  —  COUNTY SCANNER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"### Live County Scanner — All 79 Counties · {MONTH_NAMES[sel_month-1]}")

    st.markdown(f"""
    <div class="abox abox-info" style="margin-bottom:1rem">
      <b>How the scanner works:</b> For each of the 79 counties, the model uses the
      <b>climate conditions from the sidebar</b> (rainfall, temperatures, VPD, soil moisture)
      combined with <b>each county's own terrain characteristics</b> (wetland fraction, elevation,
      slope, NDVI — from dataset medians). This shows which counties are most at risk right now
      under the climate conditions you have specified.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Scanning all 79 counties…"):
        scan_df = scan_all_counties(
            rainfall, soil_moist, max_temp, min_temp,
            vpd, flood_prev, sel_month,
        )

    # ── Scanner KPIs ──────────────────────────────────────────────────────────
    n_critical  = int((scan_df["Risk"] == "Critical Risk").sum())
    n_high      = int((scan_df["Risk"] == "High Risk").sum())
    n_moderate  = int((scan_df["Risk"] == "Moderate Risk").sum())
    avg_prob    = scan_df["Prob %"].mean()

    sk1, sk2, sk3, sk4 = st.columns(4, gap="small")
    for col, val, lbl, sub, kc in [
        (sk1, n_critical, "Critical Risk Counties",  "probability ≥ 75%", VIOLET),
        (sk2, n_high,     "High Risk Counties",      "probability 50–75%", ORANGE),
        (sk3, n_moderate, "Moderate Risk Counties",  "probability 25–50%", AMBER),
        (sk4, f"{avg_prob:.1f}%","Mean Probability", "across all 79 counties", BLUE),
    ]:
        col.markdown(f"""
        <div class="kpi" style="--kpi-color:{kc}">
          <div class="kpi-val" style="color:{kc}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scanner chart + table ─────────────────────────────────────────────────
    sc1, sc2 = st.columns([1.6, 1], gap="medium")

    with sc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">All-County Flood Probability — Current Climate Inputs</div>',
                    unsafe_allow_html=True)

        sc_colors = scan_df["_color"].tolist()
        # Highlight selected county
        bar_line_colors = [EM_BRIGHT if c == sel_county else "rgba(0,0,0,0)"
                           for c in scan_df["County"]]
        bar_line_widths = [2.5 if c == sel_county else 0 for c in scan_df["County"]]

        fig_scan = go.Figure(go.Bar(
            x=scan_df["Prob %"],
            y=scan_df["County"],
            orientation="h",
            marker=dict(
                color=sc_colors,
                line=dict(color=bar_line_colors, width=bar_line_widths),
            ),
            text=scan_df.apply(
                lambda r: f"  {r['Prob %']:.1f}%{'  ◀ '+sel_county if r['County']==sel_county else ''}",
                axis=1,
            ),
            textposition="outside",
            textfont=dict(size=8.5, color=TEXT_MUT),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Flood probability: %{x:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
        fig_scan.add_vline(
            x=THRESHOLD * 100, line_dash="dash", line_color=RED, line_width=1.5,
            annotation_text=f"Alert threshold ({THRESHOLD*100:.0f}%)",
            annotation_position="top right",
            annotation_font=dict(size=9, color=RED),
        )
        fig_scan.update_layout(
            height=max(700, len(scan_df) * 15),
            margin=dict(t=10, b=10, l=10, r=80),
            xaxis=dict(title="Flood Probability (%)", range=[0, 110],
                       showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9)),
            yaxis=dict(showgrid=False, tickfont=dict(size=8.5),
                       autorange="reversed"),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="DM Sans"),
            showlegend=False,
        )
        st.plotly_chart(fig_scan, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
        st.markdown("</div>", unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">County-by-County Results</div>', unsafe_allow_html=True)

        # Build HTML table with inline probability bars
        rows_html = ""
        for _, row in scan_df.iterrows():
            is_sel = row["County"] == sel_county
            bg = f"background:rgba(52,211,153,.06);font-weight:700;" if is_sel else ""
            bar_w = min(100, row["Prob %"])
            rows_html += f"""
            <tr style="{bg}border-bottom:1px solid #F1F5F9">
              <td style="padding:5px 6px;font-size:0.79rem;color:{TEXT_DARK}">{row['County']}</td>
              <td style="padding:5px 6px;font-size:0.8rem;font-weight:700;
                         color:{row['_color']};text-align:right;white-space:nowrap">{row['Prob %']:.1f}%</td>
              <td style="padding:5px 6px;min-width:80px">
                <div class="prob-bar-wrap">
                  <div class="prob-bar-fill" style="width:{bar_w}%;background:{row['_color']}"></div>
                </div>
              </td>
              <td style="padding:5px 6px;font-size:0.74rem;color:{row['_color']};white-space:nowrap">{row['Risk']}</td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-y:auto;max-height:680px;border:1px solid {BORDER};border-radius:10px">
          <table style="width:100%;border-collapse:collapse">
            <thead style="position:sticky;top:0;background:#F8FAFC;z-index:1">
              <tr>
                <th style="padding:7px 6px;font-size:0.71rem;font-weight:700;
                           text-transform:uppercase;letter-spacing:.05em;
                           color:{MUTED};text-align:left">County</th>
                <th style="padding:7px 6px;font-size:0.71rem;font-weight:700;
                           text-transform:uppercase;letter-spacing:.05em;
                           color:{MUTED};text-align:right">Prob</th>
                <th style="padding:7px 6px;min-width:80px"></th>
                <th style="padding:7px 6px;font-size:0.71rem;font-weight:700;
                           text-transform:uppercase;letter-spacing:.05em;
                           color:{MUTED}">Risk</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # CSV download for scan results
        scan_export = scan_df[["County","Prob %","Risk"]].copy()
        scan_export["Flood Alert"] = scan_export["Prob %"].apply(
            lambda p: "Yes" if p >= THRESHOLD*100 else "No")
        scan_csv = scan_export.to_csv(index=False).encode()
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="Download all-county scan as CSV",
            data=scan_csv,
            file_name=f"county_scan_{MONTH_NAMES[sel_month-1]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4  —  HISTORICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    cty_ts = (monthly_df[monthly_df["county"]==sel_county].copy()
              .sort_values(["year","month"]))
    cty_ts["date"] = pd.to_datetime(cty_ts[["year","month"]].assign(day=1))

    st.markdown(f"### Historical Flood Record — {sel_county}")

    st.markdown(f"""
    <div class="abox abox-info" style="margin-bottom:1rem">
      <b>How to read this page:</b> The charts below show <b>recorded flood events from the
      training dataset (2011–2025)</b>. A flood rate of <b>0%</b> means no flooding was recorded
      for that period — it does <b>not</b> mean the county never floods. The dataset captures
      large-scale inundation months only; localised or short-duration floods may be underreported.<br><br>
      The <b>model prediction</b> (Tab 1) is independent — it uses the climate inputs you specify,
      not this historical rate. A county with <b>0% historical rate can still receive a high predicted
      probability</b> if current conditions are dangerous.
    </div>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns([2.2, 1], gap="medium")

    with h1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Flood Event Timeline (2011–2025)</div>',
                    unsafe_allow_html=True)
        fig_tl = go.Figure()
        for y in range(2011, 2026):
            fig_tl.add_vrect(
                x0=f"{y}-06-01", x1=f"{y}-10-01",
                fillcolor="rgba(29,78,216,0.05)", line_width=0,
                annotation_text="Rainy season" if y==2011 else "",
                annotation_position="top left",
                annotation_font=dict(size=8.5, color=MUTED),
            )
        nf = cty_ts[cty_ts["flood"]==0]
        fig_tl.add_trace(go.Scatter(
            x=nf["date"], y=nf["flood"], mode="markers", name="No Flood",
            marker=dict(color="#D5D8DC", size=5, opacity=0.7),
        ))
        fl = cty_ts[cty_ts["flood"]==1]
        fig_tl.add_trace(go.Scatter(
            x=fl["date"], y=fl["flood"], mode="markers", name="Flood Event",
            marker=dict(color=RED, size=12, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>Flood Event</b><br>%{x|%B %Y}<extra></extra>",
        ))
        fig_tl.update_layout(
            height=240,
            xaxis=dict(title="Date", showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
            yaxis=dict(tickvals=[0,1], ticktext=["No Flood","Flood"],
                       range=[-0.3,1.5], tickfont=dict(size=9.5)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="DM Sans"), margin=dict(t=5,b=5,l=5,r=5),
            legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10)),
        )
        st.plotly_chart(fig_tl, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
        st.markdown("</div>", unsafe_allow_html=True)

    with h2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Monthly Seasonality</div>', unsafe_allow_html=True)
        seas     = cty_ts.groupby("month")["flood"].mean() * 100
        sea_vals = [seas.get(m, 0) for m in range(1,13)]
        avg_sea  = np.mean(sea_vals)

        fig_sea = go.Figure(go.Bar(
            x=MONTH_SHORT, y=sea_vals,
            marker_color=[RED if v>avg_sea else BLUE for v in sea_vals],
            marker_line_width=0,
            text=[f"{v:.0f}%" if v>0 else "" for v in sea_vals],
            textposition="outside", textfont=dict(size=9),
            hovertemplate="%{x}<br>Flood rate: %{y:.1f}%<extra></extra>",
        ))
        fig_sea.add_hline(y=avg_sea, line_dash="dot", line_color=MUTED, line_width=1,
                          annotation_text=f"avg {avg_sea:.1f}%",
                          annotation_font=dict(size=8.5, color=MUTED))
        fig_sea.update_layout(
            height=240,
            xaxis=dict(tickfont=dict(size=9.5), showgrid=False),
            yaxis=dict(title="Flood Rate (%)", showgrid=True,
                       gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="DM Sans"), margin=dict(t=5,b=5,l=5,r=5),
        )
        st.plotly_chart(fig_sea, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
        st.markdown("</div>", unsafe_allow_html=True)

    # Annual trend
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
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=7, color=BLUE, line=dict(color="white",width=1.5)),
        fill="tozeroy", fillcolor="rgba(29,78,216,0.07)",
        hovertemplate="Year: %{x}<br>Flood rate: %{y:.1f}%<extra></extra>",
    ))
    fig_yr.update_layout(
        height=240,
        xaxis=dict(title="Year", showgrid=True, gridcolor="#F2F3F4",
                   dtick=1, tickangle=-45, tickfont=dict(size=9.5)),
        yaxis=dict(title="Flood Rate (%)", showgrid=True,
                   gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(family="DM Sans"), margin=dict(t=5,b=5,l=5,r=5),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
    )
    st.plotly_chart(fig_yr, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
    st.markdown("</div>", unsafe_allow_html=True)

    # Multi-county comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Multi-County Comparison</div>', unsafe_allow_html=True)
    defaults = [sel_county]
    for c in ["Malakal","Bor South","Rubkona","Juba","Renk"]:
        if c != sel_county and c in counties: defaults.append(c)
        if len(defaults) == 5: break
    compare = st.multiselect("Select counties:", counties, default=defaults, key="cmp")
    if compare:
        rows = []
        for c in compare:
            cd2 = monthly_df[monthly_df["county"]==c].groupby("year")["flood"].mean()*100
            for yr, r in cd2.items():
                rows.append({"County":c,"Year":yr,"Flood Rate (%)":round(r,2)})
        cmp_df = pd.DataFrame(rows)
        fig_cmp = px.line(cmp_df, x="Year", y="Flood Rate (%)", color="County",
                          markers=True,
                          color_discrete_sequence=px.colors.qualitative.Safe)
        fig_cmp.update_layout(
            height=300, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="DM Sans"), margin=dict(t=5,b=5,l=5,r=5),
            xaxis=dict(showgrid=True, gridcolor="#F2F3F4", dtick=2, tickfont=dict(size=9.5)),
            yaxis=dict(showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
    st.markdown("</div>", unsafe_allow_html=True)

    # System KPIs
    st.markdown("#### System-Wide Statistics")
    sk1,sk2,sk3,sk4,sk5 = st.columns(5, gap="small")
    for col, val, lbl, kc in [
        (sk1, f"{int(hist_df['flood_events'].sum()):,}",   "Total Flood Events",      RED),
        (sk2, f"{hist_df['flood_rate'].mean()*100:.1f}%",  "National Mean Rate",      AMBER),
        (sk3, str((hist_df["flood_rate"]>0.08).sum()),     "Critical Counties (>8%)", VIOLET),
        (sk4, str((hist_df["flood_rate"]<0.01).sum()),     "Low-Risk Counties (<1%)", EMERALD),
        (sk5, hist_df.loc[hist_df["flood_rate"].idxmax(),"county"],
                                                            "Highest-Risk County",    BLUE),
    ]:
        col.markdown(f"""
        <div class="kpi" style="--kpi-color:{kc}">
          <div class="kpi-val" style="font-size:1.4rem;color:{kc}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5  —  MODEL SCIENCE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Model Performance, Validation & Scientific Basis")

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
    <b>Logistic Regression was selected</b> because it achieved the highest F1 ({lr_f1:.4f}) and
    Precision ({lr_prec:.4f}) on the unseen 2024–2025 test set. Precision of {lr_prec:.0%} means
    {lr_prec:.0%} of flood alerts are genuine — critical for a humanitarian system where false alarms
    trigger costly evacuations and erode institutional trust. Random Forest achieved the highest
    AUC-ROC ({rf_auc:.4f}) but with precision of only {rf_prec:.4f}, meaning over half its alerts
    would be false positives.<br><br>
    Cross-validation (5-fold TimeSeriesSplit, 2011–2023) confirmed AUC = {cv_auc:.4f} ± {cv_std:.4f},
    showing stable generalisation over time. The dominant predictor (<b>{top_feat}</b>, coefficient
    {top_imp*100:.1f}%) reflects that flood conditions persist month-to-month. A leakage audit excluded
    <code>water_fraction</code> — it encodes the flood label definition itself. Climate-only variables
    still achieve AUC = {meta['ablation'][2]['Test AUC']:.3f}, confirming the model captures genuine
    climate-flood relationships.
    </div>
    """, unsafe_allow_html=True)

    # Metric guide
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">What Each Metric Means</div>', unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4, gap="small")
    for col, metric, defn, good in [
        (mc1, "AUC-ROC",
         "Ability to rank flood months above no-flood months. 1.0=perfect, 0.5=random. Best for imbalanced data.",
         "≥ 0.85 is strong"),
        (mc2, "F1 Score",
         "Harmonic mean of Precision & Recall. Balances false alarms vs missed floods.",
         "≥ 0.50 is solid for 1:20 imbalance"),
        (mc3, "Precision",
         "Of all flood predictions, what fraction were real floods? High = few false alarms.",
         "Higher = fewer wasted alerts"),
        (mc4, "Recall",
         "Of all real floods, what fraction did we catch? High = fewer missed events.",
         "Higher = fewer missed floods"),
    ]:
        col.markdown(f"""
        <div style="background:#F8F9FA;border-radius:8px;padding:0.9rem;
                    border-left:4px solid {BLUE}">
          <div style="font-weight:700;font-size:0.9rem;color:{TEXT_DARK}">{metric}</div>
          <div style="font-size:0.8rem;color:{TEXT_MUT};margin:6px 0">{defn}</div>
          <div style="font-size:0.75rem;color:{EMERALD};font-weight:600">{good}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # All model results
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Hold-Out Test Results — 2024–2025 (Never Seen During Training)</div>',
                unsafe_allow_html=True)
    tm = meta["test_metrics"]
    model_names = list(tm.keys())[:4]
    tc1, tc2, tc3, tc4 = st.columns(4, gap="small")
    for col, name in zip([tc1, tc2, tc3, tc4], model_names):
        mt = tm[name]
        is_best  = (name == meta["best_model_name"])
        auc_col  = EMERALD if is_best else BLUE
        card_cls = "model-card model-card-best" if is_best else "model-card"
        best_badge = (
            f'<div style="display:inline-block;background:{EMERALD};color:white;'
            f'font-size:0.65rem;font-weight:700;border-radius:20px;'
            f'padding:2px 8px;letter-spacing:.04em;margin-bottom:8px">✓ DEPLOYED</div>'
            if is_best else
            '<div style="height:22px;margin-bottom:8px"></div>'
        )
        col.markdown(f"""
        <div class="{card_cls}">
          {best_badge}
          <div style="font-size:0.78rem;font-weight:700;color:{TEXT_DARK};
                      margin-bottom:4px">{name}</div>
          <div class="model-auc" style="color:{auc_col}">{mt['auc_roc']:.4f}</div>
          <div style="font-size:0.68rem;color:{MUTED};margin-bottom:10px;
                      text-transform:uppercase;letter-spacing:.06em">AUC-ROC</div>
          <div style="font-size:0.79rem">
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

    # CV + Ablation
    mp1, mp2 = st.columns(2, gap="medium")

    with mp1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Cross-Validation AUC-ROC (5-Fold TimeSeriesSplit · 2011–2023)</div>',
                    unsafe_allow_html=True)
        cv_m   = meta["cv_metrics"]
        names  = list(cv_m.keys())
        means  = [cv_m[n]["auc_roc_mean"] for n in names]
        stds   = [cv_m[n]["auc_roc_std"]  for n in names]
        cvcolors = [EMERALD if n==meta["best_model_name"] else BLUE for n in names]
        fig_cv = go.Figure()
        for nm, mn, sd, cc in zip(names, means, stds, cvcolors):
            fig_cv.add_trace(go.Bar(
                name=nm, x=[nm], y=[mn],
                error_y=dict(type="data", array=[sd], visible=True, thickness=1.8, width=8),
                marker_color=cc, marker_line_width=0, width=0.5,
                text=[f"{mn:.4f}"], textposition="outside",
                textfont=dict(size=10, color=TEXT_DARK),
            ))
        fig_cv.add_hline(y=0.9, line_dash="dot", line_color="#AEB6BF", line_width=1,
                         annotation_text="0.90 reference",
                         annotation_font=dict(size=9, color=MUTED))
        fig_cv.update_layout(
            height=290, showlegend=False,
            yaxis=dict(range=[0,1.1], title="AUC-ROC",
                       showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
            xaxis=dict(tickfont=dict(size=9.5), showgrid=False),
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(family="DM Sans"), margin=dict(t=15,b=0,l=5,r=5),
        )
        st.plotly_chart(fig_cv, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
        st.markdown(f"""
        <div class="abox abox-info" style="font-size:0.8rem">
        The ±std bars show consistency across all 5 time windows.
        {meta['best_model_name']} achieved {cv_m[meta['best_model_name']]['auc_roc_mean']:.4f}
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
            abl_c = [SB, BLUE, "#AEB6BF"]
            fig_abl = go.Figure()
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
                           showgrid=True, gridcolor="#F2F3F4", tickfont=dict(size=9.5)),
                xaxis=dict(tickfont=dict(size=10), showgrid=False),
                paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                font=dict(family="DM Sans"), margin=dict(t=15,b=0,l=5,r=5),
                legend=dict(orientation="h", y=1.15, font=dict(size=9.5)),
            )
            st.plotly_chart(fig_abl, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
            delta = abl.iloc[0]["Test AUC"] - abl.iloc[2]["Test AUC"]
            st.markdown(f"""
            <div class="abox abox-info" style="font-size:0.8rem">
            <b>Key finding:</b> Removing <code>flood_prev_month</code> drops AUC
            {abl.iloc[0]['Test AUC']:.3f} → {abl.iloc[1]['Test AUC']:.3f}
            (−{abl.iloc[0]['Test AUC']-abl.iloc[1]['Test AUC']:.3f}).
            Pure climate variables alone achieve AUC {abl.iloc[2]['Test AUC']:.3f},
            confirming real climate-flood signals beyond simple persistence.
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # National flood heatmap
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">National Flood Calendar — Monthly × Annual Flood Rate (All 79 Counties)</div>',
                unsafe_allow_html=True)
    heat  = (monthly_df.groupby(["year","month"])["flood"].mean()*100).reset_index()
    pivot = heat.pivot(index="month", columns="year", values="flood")
    pivot.index = [MONTH_SHORT[i-1] for i in pivot.index]
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0,"#EAFAF1"],[0.25,"#F9E79F"],[0.6,"#E67E22"],[1.0,"#922B21"]],
        text=np.round(pivot.values,1),
        texttemplate="%{text}%",
        textfont=dict(size=8.5),
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Flood rate: %{z:.1f}%<extra></extra>",
        showscale=True,
        colorbar=dict(title=dict(text="Rate %",side="right"),
                      thickness=12, len=0.9, tickfont=dict(size=9)),
        xgap=2, ygap=2,
    ))
    fig_heat.update_layout(
        height=300, margin=dict(t=5,b=5,l=5,r=60),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9.5), title="Year", side="bottom"),
        yaxis=dict(title="Month", tickfont=dict(size=9.5)),
        paper_bgcolor=BG_CARD, font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config=dict(displayModeBar=False, scrollZoom=False))
    st.markdown("</div>", unsafe_allow_html=True)

    # Methodology
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Methodology &amp; Transparency</div>', unsafe_allow_html=True)
    meth = meta.get("methodology", {})
    excl = meta.get("excluded_features", [])
    mm1, mm2, mm3 = st.columns(3, gap="medium")
    with mm1:
        st.markdown(f"""
        <div class="abox abox-info">
        <b>Validation strategy</b><br>{meth.get('cv','')}<br><br>
        <b>Test set</b><br>{meth.get('test','')}
        </div>""", unsafe_allow_html=True)
    with mm2:
        st.markdown(f"""
        <div class="abox abox-info">
        <b>Class imbalance (1:20)</b><br>{meth.get('imbalance','')}<br><br>
        <b>Threshold selection</b><br>{meth.get('threshold','')}
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
  {meta['best_model_name']} · AUC-ROC {meta['test_metrics'][meta['best_model_name']]['auc_roc']} ·
  F1 {meta['test_metrics'][meta['best_model_name']]['f1']} ·
  Precision {meta['test_metrics'][meta['best_model_name']]['precision']} ·
  79 Counties · 2011–2025<br>
  Author: <b>Chol Monykuch</b> &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
