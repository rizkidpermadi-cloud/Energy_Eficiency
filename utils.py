"""
utils.py — Shared utilities for the Energy Efficiency Streamlit App.
Provides: path constants, model loading, data loading, shared CSS, and Plotly theme.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# ─── Path Resolution ──────────────────────────────────────────────────────────
# ROOT_DIR always points to the project root (where app.py lives),
# regardless of which page file imports this module.
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(ROOT_DIR, 'model')
DATA_PATH   = os.path.join(ROOT_DIR, 'energy_efficiency_data.csv')
KERAS_PATH  = os.path.join(MODEL_DIR, 'model_mp2.keras')
JOBLIB_PATH = os.path.join(MODEL_DIR, 'model_pipeline_mp2.joblib')

# ─── Model Metrics (from notebook evaluation) ─────────────────────────────────
MODEL_METRICS = {
    'Linear Regression (Ridge)': dict(
        r2=0.8821, mae=2.2748, rmse=3.2034,
        color='#2563EB', light='#DBEAFE', emoji='📈',
        desc='Model linier sederhana dengan regularisasi Ridge (α=1.0).'
    ),
    'Random Forest Regressor': dict(
        r2=0.9629, mae=1.1428, rmse=1.7957,
        color='#0D9488', light='#CCFBF1', emoji='🌲',
        desc='Ensemble 100 pohon keputusan, performa terbaik di antara ketiga model.'
    ),
    'Artificial Neural Network (ANN)': dict(
        r2=0.9357, mae=1.6189, rmse=2.3646,
        color='#7C3AED', light='#EDE9FE', emoji='🧠',
        desc='Jaringan saraf tiruan 3-layer dengan Dropout dan EarlyStopping.'
    ),
}

# ─── Feature Label Map ────────────────────────────────────────────────────────
LABEL_MAP = {
    'Relative_Compactness':      'Kepadatan Relatif',
    'Surface_Area':              'Luas Permukaan (m²)',
    'Wall_Area':                 'Luas Dinding (m²)',
    'Roof_Area':                 'Luas Atap (m²)',
    'Overall_Height':            'Tinggi Bangunan (m)',
    'Orientation':               'Orientasi',
    'Glazing_Area':              'Rasio Luas Kaca',
    'Glazing_Area_Distribution': 'Distribusi Kaca',
}

# ─── Plotly Light Theme ───────────────────────────────────────────────────────
CHART_COLORS = ['#2563EB', '#0D9488', '#7C3AED', '#D97706', '#DC2626', '#059669']

PLOTLY_LAYOUT = dict(
    template        = 'plotly_white',
    paper_bgcolor   = 'rgba(0,0,0,0)',
    plot_bgcolor    = 'rgba(248,250,252,0.6)',
    font            = dict(family='Inter, system-ui, sans-serif', color='#334155', size=13),
    margin          = dict(l=16, r=16, t=48, b=16),
    colorway        = CHART_COLORS,
    title_font      = dict(size=15, color='#0F172A', family='Inter, sans-serif'),
    legend          = dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#E2E8F0', borderwidth=1),
    xaxis           = dict(gridcolor='#F1F5F9', linecolor='#E2E8F0', tickfont=dict(color='#64748B')),
    yaxis           = dict(gridcolor='#F1F5F9', linecolor='#E2E8F0', tickfont=dict(color='#64748B')),
)

# ─── Shared CSS (light, clean, Inter) ─────────────────────────────────────────
LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* ── App background ── */
.stApp,
[data-testid="stAppViewContainer"] { background: #F8FAFC !important; }

/* ── Main content padding ── */
[data-testid="block-container"] { padding: 2rem 2.5rem 4rem !important; max-width: 1280px; margin: 0 auto; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }
section[data-testid="stSidebar"] * { color: #0F172A !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03) !important;
    transition: box-shadow 0.25s, transform 0.25s !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 24px rgba(37,99,235,0.12) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    color: #64748B !important; font-size: 11.5px !important;
    font-weight: 600 !important; letter-spacing: 0.4px !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #0F172A !important; font-size: 26px !important; font-weight: 800 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
}
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important; font-weight: 600 !important;
    color: #64748B !important; border-radius: 9px !important;
    padding: 8px 16px !important; border: none !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #2563EB !important; color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: #F1F5F9 !important; color: #1E40AF !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div > div {
    background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important; color: #0F172A !important;
    font-size: 14px !important; font-weight: 500 !important;
    transition: border-color 0.2s !important;
}
.stSelectbox > div > div > div:focus-within,
.stSelectbox > div > div > div:hover { border-color: #2563EB !important; }

/* ── Slider ── */
.stSlider p { color: #334155 !important; font-weight: 600 !important; font-size: 13.5px !important; }

/* ── Primary Button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1d4ed8 100%) !important;
    border: none !important; border-radius: 11px !important; color: #FFFFFF !important;
    font-weight: 700 !important; font-size: 15px !important; padding: 12px 24px !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
    transition: all 0.25s ease !important; letter-spacing: 0.1px !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,0.45) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important; color: #334155 !important;
    font-weight: 600 !important; font-size: 14px !important;
    transition: all 0.2s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #2563EB !important; color: #2563EB !important;
    background: #EFF6FF !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #E2E8F0 !important; }
.stDataFrame thead tr th { background: #F8FAFC !important; color: #64748B !important; font-size: 12px !important; font-weight: 600 !important; letter-spacing: 0.3px !important; }

/* ── Success / Info / Warning alerts ── */
[data-testid="stAlert"] { border-radius: 12px !important; font-weight: 500 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 10px; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ────── CUSTOM COMPONENT CLASSES ────── */

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 35%, #2563eb 70%, #0ea5e9 100%);
    border-radius: 20px; padding: 44px 48px; margin-bottom: 32px;
    position: relative; overflow: hidden;
    box-shadow: 0 12px 40px rgba(37,99,235,0.25);
}
.hero-banner::before {
    content: ""; position: absolute; top: -80px; right: -80px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: ""; position: absolute; bottom: -100px; left: -60px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(14,165,233,0.2), transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.25);
    color: rgba(255,255,255,0.9); border-radius: 999px; padding: 4px 14px;
    font-size: 11.5px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 14px;
}
.hero-title { font-size: 36px; font-weight: 900; color: #FFFFFF; line-height: 1.2; margin-bottom: 10px; letter-spacing: -0.5px; }
.hero-sub   { font-size: 15.5px; color: rgba(255,255,255,0.72); max-width: 580px; line-height: 1.65; }

/* White card */
.white-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.white-card:hover { box-shadow: 0 6px 24px rgba(37,99,235,0.1); border-color: #BFDBFE; }
.white-card h4 { color: #0F172A; font-size: 14.5px; font-weight: 700; margin-bottom: 8px; }
.white-card p  { color: #64748B; font-size: 13.5px; line-height: 1.65; margin: 0; }

/* Section heading */
.section-title { font-size: 20px; font-weight: 800; color: #0F172A; margin-bottom: 4px; }
.section-sub   { font-size: 14px; color: #64748B; margin-bottom: 22px; }

/* Horizontal rule */
.divider { height: 1px; background: #E2E8F0; margin: 28px 0; border: none; }

/* Stat chip */
.stat-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 999px;
    color: #1E40AF; padding: 4px 12px; font-size: 12px; font-weight: 600;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 1.5px solid #86EFAC; border-radius: 18px;
    padding: 30px 32px; margin-top: 24px;
    box-shadow: 0 4px 24px rgba(16,185,129,0.12);
}
.result-label { font-size: 11px; font-weight: 700; color: #16A34A; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.result-value { font-size: 52px; font-weight: 900; color: #15803D; letter-spacing: -2px; line-height: 1; }
.result-unit  { font-size: 20px; font-weight: 600; color: #16A34A; }
.result-desc  { font-size: 13.5px; color: #166534; margin-top: 14px; line-height: 1.65; padding-top: 14px; border-top: 1px solid #BBF7D0; }

/* Model selector button card */
.model-card {
    background: #FFFFFF; border: 2px solid #E2E8F0; border-radius: 14px;
    padding: 18px 20px; cursor: pointer; transition: all 0.2s ease;
    text-align: center;
}
.model-card:hover { border-color: #93C5FD; box-shadow: 0 4px 16px rgba(37,99,235,0.1); }
.model-card.selected { border-color: #2563EB; background: #EFF6FF; }
</style>
"""


def inject_css():
    """Inject shared light-theme CSS into the current Streamlit page."""
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


# ─── Asset & Data Loaders ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Memuat model AI…")
def load_assets():
    """Load Keras ANN + scikit-learn pipeline. Cached globally across pages."""
    import tensorflow as tf
    import joblib
    if not os.path.exists(KERAS_PATH) or not os.path.exists(JOBLIB_PATH):
        return None, None, None, None
    pipeline  = joblib.load(JOBLIB_PATH)
    scaler    = pipeline['scaler']
    lr_model  = pipeline['linear_regression']
    rf_model  = pipeline['random_forest']
    ann_model = tf.keras.models.load_model(KERAS_PATH)
    return ann_model, lr_model, rf_model, scaler


@st.cache_data(show_spinner="📂 Memuat dataset…")
def load_data() -> pd.DataFrame | None:
    """Load energy efficiency CSV. Returns None if not found."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None
