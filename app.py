"""
app.py — Home / Landing Page
Streamlit Multi-Page App: Energy Efficiency Analytics & Cooling Load Predictor
"""

import streamlit as st
import pandas as pd
import sys, os

# Ensure root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import inject_css, load_data, MODEL_METRICS, LABEL_MAP

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Efficiency App",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 20px;">
        <div style="font-size:42px; margin-bottom:6px;">⚡</div>
        <div style="font-size:17px; font-weight:800; color:#0F172A; letter-spacing:-0.3px;">Energy Efficiency</div>
        <div style="font-size:12px; color:#94A3B8; margin-top:3px;">Analytics & Predictor</div>
    </div>
    <div style="height:1px; background:#E2E8F0; margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:11px; font-weight:700; color:#94A3B8; letter-spacing:0.8px;
       text-transform:uppercase; margin-bottom:10px; padding:0 4px;">Navigasi</p>
    """, unsafe_allow_html=True)

    st.page_link("app.py",                            label="🏠 Beranda",                   help="Halaman utama")
    st.page_link("pages/1_Analisis_Visualisasi.py",   label="📊 Analisis & Visualisasi",    help="Eksplorasi dataset dan perbandingan model")
    st.page_link("pages/2_Prediksi_Cooling_Load.py",  label="🔮 Prediksi Cooling Load",     help="Kalkulator beban pendinginan interaktif")

    st.markdown("<div style='height:1px;background:#E2E8F0;margin:16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:11px; font-weight:700; color:#94A3B8; letter-spacing:0.8px;
       text-transform:uppercase; margin-bottom:10px; padding:0 4px;">Model Tersedia</p>
    """, unsafe_allow_html=True)

    for name, m in MODEL_METRICS.items():
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:8px 10px;
             background:{m['light']}; border-radius:10px; margin-bottom:6px; border:1px solid {m['color']}20;">
            <span style="font-size:16px;">{m['emoji']}</span>
            <div>
                <div style="font-size:12px; font-weight:600; color:#0F172A;">{name}</div>
                <div style="font-size:11px; color:#64748B;">R² {m['r2']*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="height:1px;background:#E2E8F0;margin:16px 0;"></div>
    <p style="font-size:11px; color:#CBD5E1; text-align:center; line-height:1.5;">
      Mini Project 2 · Rizki Dwi Permadi<br>Dataset: UCI Energy Efficiency
    </p>
    """, unsafe_allow_html=True)

# ─── Home Page Content ────────────────────────────────────────────────────────
df = load_data()

# Hero
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">⚡ Energy Efficiency Analytics</div>
    <div class="hero-title">Prediksi &amp; Analisis<br>Beban Pendinginan Bangunan</div>
    <div class="hero-sub">
        Aplikasi Machine Learning interaktif untuk menganalisis dataset efisiensi energi bangunan
        dan memprediksi <strong style="color:white;">Cooling Load</strong> menggunakan tiga model
        prediksi — Ridge Regression, Random Forest, dan Artificial Neural Network (ANN).
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Metrics ──────────────────────────────────────────────────────────────
if df is not None:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Sampel",        f"{df.shape[0]:,}", "bangunan")
    with c2: st.metric("Fitur Input",         f"{df.shape[1]-2}",  "atribut fisik")
    with c3: st.metric("Cooling Load Min",    f"{df['Cooling_Load'].min():.1f} kWh/m²")
    with c4: st.metric("Cooling Load Maks",   f"{df['Cooling_Load'].max():.1f} kWh/m²")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Feature navigation cards ──────────────────────────────────────────────────
st.markdown('<div class="section-title">🗺️ Navigasi Halaman</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Pilih halaman yang ingin Anda jelajahi.</div>', unsafe_allow_html=True)

nav_c1, nav_c2 = st.columns(2, gap="large")

with nav_c1:
    st.markdown("""
    <div class="white-card" style="border-left: 4px solid #2563EB; padding-left:20px;">
        <div style="font-size:32px; margin-bottom:10px;">📊</div>
        <h4 style="font-size:17px;">Analisis &amp; Visualisasi Data</h4>
        <p>Eksplorasi interaktif dataset — scatter plot, korelasi fitur, distribusi Cooling Load,
           heatmap korelasi, violin plot, dan perbandingan akurasi ketiga model dengan grafik batang.</p>
        <div style="margin-top:14px; display:flex; gap:6px; flex-wrap:wrap;">
            <span class="stat-chip">📈 Scatter Plot</span>
            <span class="stat-chip">🔥 Heatmap Korelasi</span>
            <span class="stat-chip">🎻 Violin Plot</span>
            <span class="stat-chip">📊 Perbandingan Model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Analisis_Visualisasi.py", label="Buka Halaman Analisis →", use_container_width=True)

with nav_c2:
    st.markdown("""
    <div class="white-card" style="border-left: 4px solid #0D9488; padding-left:20px;">
        <div style="font-size:32px; margin-bottom:10px;">🔮</div>
        <h4 style="font-size:17px;">Prediksi Cooling Load</h4>
        <p>Kalkulator prediksi interaktif — pilih model (ANN, Ridge, atau Random Forest),
           masukkan spesifikasi fisik bangunan, dan dapatkan estimasi Cooling Load
           beserta interpretasi efisiensi energi secara instan.</p>
        <div style="margin-top:14px; display:flex; gap:6px; flex-wrap:wrap;">
            <span class="stat-chip" style="background:#F0FDF4;border-color:#86EFAC;color:#15803D;">🌲 Random Forest</span>
            <span class="stat-chip" style="background:#EDE9FE;border-color:#C4B5FD;color:#6D28D9;">🧠 ANN</span>
            <span class="stat-chip">📈 Ridge</span>
            <span class="stat-chip" style="background:#F0FDF4;border-color:#86EFAC;color:#15803D;">🎯 Radar Chart</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Prediksi_Cooling_Load.py", label="Buka Halaman Prediksi →", use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Model comparison overview ─────────────────────────────────────────────────
st.markdown('<div class="section-title">🤖 Tiga Model Prediksi</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Perbandingan singkat kinerja model pada data uji (154 sampel).</div>', unsafe_allow_html=True)

m_cols = st.columns(3, gap="medium")
for i, (name, m) in enumerate(MODEL_METRICS.items()):
    with m_cols[i]:
        is_best = name == 'Random Forest Regressor'
        best_tag = '<span style="display:inline-block;background:#FEF3C7;border:1px solid #FCD34D;color:#B45309;border-radius:999px;padding:1px 8px;font-size:10px;font-weight:700;margin-left:6px;">🏆 TERBAIK</span>' if is_best else ''
        st.markdown(f"""
        <div class="white-card" style="text-align:center; border-top: 3px solid {m['color']};">
            <div style="font-size:28px;">{m['emoji']}</div>
            <div style="font-size:13px; font-weight:700; color:#0F172A; margin:8px 0 4px; line-height:1.3;">{name}{best_tag}</div>
            <div style="font-size:34px; font-weight:900; color:{m['color']}; letter-spacing:-1px;">{m['r2']*100:.1f}%</div>
            <div style="font-size:11px; color:#94A3B8; margin-top:2px;">R² Score</div>
            <div style="display:flex; justify-content:space-around; margin-top:14px; padding-top:12px; border-top:1px solid #F1F5F9;">
                <div><div style="font-size:10px;color:#94A3B8;font-weight:600;">MAE</div><div style="font-size:13px;color:#334155;font-weight:700;">{m['mae']}</div></div>
                <div><div style="font-size:10px;color:#94A3B8;font-weight:600;">RMSE</div><div style="font-size:13px;color:#334155;font-weight:700;">{m['rmse']}</div></div>
            </div>
            <div style="margin-top:12px; font-size:12px; color:#64748B; line-height:1.5; text-align:left;">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
if df is not None:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Preview Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">5 baris pertama dari dataset Energy Efficiency UCI (768 sampel bangunan).</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
