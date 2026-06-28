"""
pages/2_Prediksi_Cooling_Load.py
Halaman 2 — Kalkulator Prediksi Cooling Load
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import inject_css, load_assets, MODEL_METRICS, PLOTLY_LAYOUT

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Cooling Load | Energy Efficiency",
    page_icon="🔮", layout="wide", initial_sidebar_state="expanded",
)
inject_css()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 20px;">
        <div style="font-size:42px; margin-bottom:6px;">⚡</div>
        <div style="font-size:17px; font-weight:800; color:#0F172A;">Energy Efficiency</div>
        <div style="font-size:12px; color:#94A3B8; margin-top:3px;">Analytics & Predictor</div>
    </div>
    <div style="height:1px; background:#E2E8F0; margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)
    st.page_link("app.py",                           label="🏠 Beranda")
    st.page_link("pages/1_Analisis_Visualisasi.py",  label="📊 Analisis & Visualisasi")
    st.page_link("pages/2_Prediksi_Cooling_Load.py", label="🔮 Prediksi Cooling Load")
    st.markdown("<div style='height:1px;background:#E2E8F0;margin:16px 0;'></div>", unsafe_allow_html=True)

    # Model info panel
    st.markdown("""
    <p style="font-size:11px; font-weight:700; color:#94A3B8; letter-spacing:0.8px;
       text-transform:uppercase; margin-bottom:10px; padding:0 4px;">Info Model</p>
    """, unsafe_allow_html=True)
    for name, m in MODEL_METRICS.items():
        st.markdown(f"""
        <div style="background:{m['light']}; border:1px solid {m['color']}30; border-radius:10px;
             padding:10px 12px; margin-bottom:7px;">
            <div style="font-size:12px; font-weight:700; color:{m['color']};">{m['emoji']} {name}</div>
            <div style="font-size:11px; color:#64748B; margin-top:3px;">R² {m['r2']*100:.1f}% &nbsp;|&nbsp; MAE {m['mae']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="height:1px;background:#E2E8F0;margin:16px 0;"></div>
    <p style="font-size:11px; color:#CBD5E1; text-align:center; line-height:1.5;">
      Mini Project 2 · Rizki Dwi Permadi
    </p>""", unsafe_allow_html=True)

# ─── Load Assets ──────────────────────────────────────────────────────────────
ann_model, lr_model, rf_model, scaler = load_assets()

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner" style="background: linear-gradient(135deg, #064E3B 0%, #065F46 30%, #0D9488 70%, #14B8A6 100%);">
    <div class="hero-eyebrow">🔮 KALKULATOR PREDIKSI</div>
    <div class="hero-title">Prediksi Beban Pendinginan<br>(Cooling Load)</div>
    <div class="hero-sub">Masukkan spesifikasi fisik bangunan dan pilih model prediksi untuk memperoleh
    estimasi <strong style="color:white;">Cooling Load</strong> secara instan, lengkap dengan interpretasi
    efisiensi energi dan profil radar bangunan.</div>
</div>
""", unsafe_allow_html=True)

# ─── Status check ─────────────────────────────────────────────────────────────
if ann_model is None:
    st.error("⚠️ **Berkas model tidak ditemukan.**")
    st.info("💡 Pastikan `model/model_mp2.keras` dan `model/model_pipeline_mp2.joblib` ada. Jalankan seluruh sel di `notebooks/MP_02_(Rizki_Dwi_Permadi)_Fixed.ipynb` untuk membuatnya.")
    st.stop()

# Status badges
st.markdown("""
<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:28px;">
    <span style="display:inline-flex;align-items:center;gap:5px;background:#F0FDF4;border:1px solid #86EFAC;
          color:#15803D;border-radius:999px;padding:5px 12px;font-size:12px;font-weight:600;">
        ✅ ANN (Keras) — Loaded
    </span>
    <span style="display:inline-flex;align-items:center;gap:5px;background:#EFF6FF;border:1px solid #93C5FD;
          color:#1E40AF;border-radius:999px;padding:5px 12px;font-size:12px;font-weight:600;">
        ✅ Ridge Regression — Loaded
    </span>
    <span style="display:inline-flex;align-items:center;gap:5px;background:#F0FDF4;border:1px solid #6EE7B7;
          color:#065F46;border-radius:999px;padding:5px 12px;font-size:12px;font-weight:600;">
        ✅ Random Forest — Loaded
    </span>
</div>
""", unsafe_allow_html=True)

# ─── ① Model Selection ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">① Pilih Model Prediksi</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Klik salah satu model untuk memilihnya sebagai model prediksi aktif.</div>', unsafe_allow_html=True)

model_names = list(MODEL_METRICS.keys())

# Initialize session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Random Forest Regressor"

btn_cols = st.columns(3, gap="medium")
for i, (name, m) in enumerate(MODEL_METRICS.items()):
    with btn_cols[i]:
        is_active = st.session_state.selected_model == name
        is_best   = name == "Random Forest Regressor"
        bg     = m['light'] if is_active else "#FFFFFF"
        border = f"2px solid {m['color']}" if is_active else "2px solid #E2E8F0"
        check  = "✅ " if is_active else ""
        best_badge = "<br><span style='font-size:10px;color:#B45309;'>🏆 Akurasi Tertinggi</span>" if is_best else ""
        st.markdown(f"""
        <div style="background:{bg}; border:{border}; border-radius:14px; padding:18px 16px;
             text-align:center; transition:all 0.2s; margin-bottom:4px; min-height:110px;">
            <div style="font-size:24px;">{m['emoji']}</div>
            <div style="font-size:12.5px; font-weight:700; color:#0F172A; margin-top:6px; line-height:1.3;">
                {check}{name}{best_badge}
            </div>
            <div style="font-size:13px; font-weight:800; color:{m['color']}; margin-top:6px;">
                R² {m['r2']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Pilih {name}", key=f"sel_{i}", use_container_width=True):
            st.session_state.selected_model = name
            st.rerun()

active_model = st.session_state.selected_model
active_m     = MODEL_METRICS[active_model]

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─── ② Building Input Form ───────────────────────────────────────────────────
st.markdown('<div class="section-title">② Masukkan Spesifikasi Fisik Bangunan</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Atur nilai setiap atribut fisik bangunan menggunakan slider dan dropdown di bawah ini.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px;
         padding:16px 20px; margin-bottom:16px;">
        <div style="font-size:13px; font-weight:700; color:#1E40AF; margin-bottom:4px;">🏗️ Dimensi & Bentuk Bangunan</div>
    </div>
    """, unsafe_allow_html=True)
    relative_compactness = st.slider(
        "Kepadatan Relatif (Relative Compactness)", 0.62, 0.98, 0.76, 0.01,
        help="Rasio volume bangunan terhadap luas permukaan. Nilai tinggi (mendekati 1) = bentuk lebih kompak/kubik, biasanya lebih efisien.")
    surface_area = st.slider(
        "Luas Permukaan Total (m²)", 514.5, 808.5, 680.0, 0.5,
        help="Total luas permukaan luar bangunan termasuk atap dan semua dinding.")
    wall_area = st.slider(
        "Luas Dinding (m²)", 245.0, 416.5, 318.5, 0.5,
        help="Luas total semua dinding eksterior bangunan.")
    roof_area = st.slider(
        "Luas Atap (m²)", 110.25, 220.5, 176.0, 0.25,
        help="Luas atap bangunan dalam meter persegi.")

with col_right:
    st.markdown("""
    <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px;
         padding:16px 20px; margin-bottom:16px;">
        <div style="font-size:13px; font-weight:700; color:#166534; margin-bottom:4px;">🪟 Tinggi, Orientasi & Kaca</div>
    </div>
    """, unsafe_allow_html=True)
    overall_height = st.selectbox(
        "Tinggi Bangunan (m)",
        [3.5, 7.0], index=1,
        format_func=lambda x: f"{x} m  ({'1 lantai – rendah' if x == 3.5 else '2 lantai – tinggi'})",
        help="Tinggi total bangunan. 3.5 m = bangunan 1 lantai; 7.0 m = bangunan 2 lantai.")
    orientation = st.selectbox(
        "Orientasi Bangunan",
        [2, 3, 4, 5],
        format_func=lambda x: {2:"2 – Utara 🧭", 3:"3 – Timur 🧭", 4:"4 – Selatan 🧭", 5:"5 – Barat 🧭"}[x],
        help="Arah hadap utama bangunan terhadap mata angin.")
    glazing_area = st.slider(
        "Rasio Luas Kaca (Glazing Area)", 0.0, 0.40, 0.25, 0.05,
        help="Proporsi area kaca jendela dibandingkan total luas dinding luar (0 = tidak ada kaca, 0.40 = 40% kaca).")
    glazing_distribution = st.slider(
        "Distribusi Kaca (Glazing Distribution)", 0, 5, 3, 1,
        help="Pola persebaran kaca jendela di sekeliling bangunan: 0 = tidak ada, 1–5 = distribusi tertentu.")

# ─── ③ Predict ────────────────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">③ Hasil Prediksi</div>', unsafe_allow_html=True)

# Show active model info strip
st.markdown(f"""
<div style="background:{active_m['light']}; border:1.5px solid {active_m['color']}40;
     border-radius:12px; padding:12px 18px; margin-bottom:20px; display:flex; align-items:center; gap:12px;">
    <span style="font-size:22px;">{active_m['emoji']}</span>
    <div>
        <div style="font-size:12px; font-weight:600; color:{active_m['color']}; letter-spacing:0.5px;">MODEL AKTIF</div>
        <div style="font-size:14.5px; font-weight:700; color:#0F172A;">{active_model}</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <div style="font-size:11px; color:#94A3B8;">Akurasi</div>
        <div style="font-size:20px; font-weight:900; color:{active_m['color']};">{active_m['r2']*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("⚡  Hitung Prediksi Cooling Load", type="primary", use_container_width=True):
    # Build feature array
    features = np.array([[
        relative_compactness, surface_area, wall_area, roof_area,
        overall_height, orientation, glazing_area, glazing_distribution
    ]])
    features_scaled = scaler.transform(features)

    # Predict
    if active_model == "Artificial Neural Network (ANN)":
        prediction = float(ann_model.predict(features_scaled, verbose=0)[0][0])
    elif active_model == "Linear Regression (Ridge)":
        prediction = float(lr_model.predict(features_scaled)[0])
    else:
        prediction = float(rf_model.predict(features_scaled)[0])

    # Efficiency interpretation
    if prediction < 15:
        eff_emoji, eff_label, eff_desc, eff_bg, eff_border = (
            "🟢", "Sangat Efisien",
            "Konsumsi energi AC sangat rendah. Desain bangunan sudah sangat optimal untuk iklim tropis!",
            "#F0FDF4", "#86EFAC"
        )
    elif prediction < 25:
        eff_emoji, eff_label, eff_desc, eff_bg, eff_border = (
            "🟡", "Cukup Efisien",
            "Beban pendinginan sedang. Ada ruang perbaikan — coba kurangi rasio kaca atau optimalkan orientasi bangunan.",
            "#FFFBEB", "#FCD34D"
        )
    else:
        eff_emoji, eff_label, eff_desc, eff_bg, eff_border = (
            "🔴", "Kurang Efisien",
            "Beban pendinginan tinggi. Pertimbangkan memperbesarkepadatan relatif, mengurangi kaca, atau mengganti orientasi bangunan.",
            "#FFF1F2", "#FCA5A5"
        )

    # Result card
    st.markdown(f"""
    <div style="background:{eff_bg}; border:1.5px solid {eff_border}; border-radius:18px;
         padding:30px 32px; margin-top:4px; box-shadow: 0 4px 24px rgba(0,0,0,0.07);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:20px;">
            <div>
                <div style="font-size:11px; font-weight:700; color:#64748B; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">
                    Estimasi Cooling Load — {active_model}
                </div>
                <div style="font-size:58px; font-weight:900; color:#0F172A; letter-spacing:-2px; line-height:1;">
                    {prediction:.2f}
                    <span style="font-size:22px; font-weight:600; color:#475569;">kWh/m²</span>
                </div>
                <div style="margin-top:12px; display:inline-flex; align-items:center; gap:6px; background:white;
                     border-radius:999px; padding:6px 16px; border:1px solid {eff_border}; font-size:13px; font-weight:700; color:#374151;">
                    {eff_emoji} {eff_label}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#94A3B8; margin-bottom:4px;">Akurasi Model</div>
                <div style="font-size:30px; font-weight:900; color:{active_m['color']};">{active_m['r2']*100:.1f}%</div>
                <div style="font-size:11px; color:#94A3B8; margin-top:2px;">R² Score</div>
                <div style="font-size:12px; color:#94A3B8; margin-top:6px;">MAE {active_m['mae']} &nbsp;|&nbsp; RMSE {active_m['rmse']}</div>
            </div>
        </div>
        <div style="margin-top:18px; padding-top:16px; border-top:1px solid {eff_border};
             font-size:13.5px; color:#334155; line-height:1.65;">
            <strong>💡 Interpretasi:</strong> {eff_desc}
            Bangunan dengan spesifikasi ini membutuhkan energi pendinginan sebesar
            <strong>{prediction:.2f} kWh/m²</strong>.
            Semakin kecil nilai ini, semakin hemat penggunaan AC bangunan.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # Radar chart
    col_radar, col_table = st.columns([3, 2], gap="large")

    with col_radar:
        cats = ["Kepadatan\nRelatif", "Luas\nPermukaan", "Luas\nDinding",
                "Luas\nAtap",        "Tinggi",           "Orientasi",
                "Rasio\nKaca",       "Distribusi\nKaca"]
        raw  = [relative_compactness, surface_area, wall_area, roof_area,
                overall_height, orientation, glazing_area, glazing_distribution]
        mins = [0.62, 514.5, 245.0, 110.25, 3.5, 2, 0.0, 0]
        maxs = [0.98, 808.5, 416.5, 220.5,  7.0, 5, 0.4, 5]
        norm = [(v - lo) / (hi - lo) if hi != lo else 0 for v, lo, hi in zip(raw, mins, maxs)]
        norm_c = norm + [norm[0]]
        cats_c = cats + [cats[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=norm_c, theta=cats_c,
            fill='toself',
            fillcolor=f"{active_m['color']}20",
            line=dict(color=active_m['color'], width=2.5),
            marker=dict(size=7, color=active_m['color'], line=dict(color='white', width=1.5)),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(248,250,252,0.9)',
                radialaxis=dict(visible=True, range=[0, 1.1], showticklabels=False, gridcolor='#E2E8F0', linecolor='#E2E8F0'),
                angularaxis=dict(gridcolor='#E2E8F0', tickfont=dict(size=12, color='#334155', family='Inter')),
            ),
            showlegend=False,
            title="Profil Input Bangunan (Dinormalisasi 0–1)",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_table:
        st.markdown("**📋 Ringkasan Input yang Digunakan**")
        labels = ["Kepadatan Relatif", "Luas Permukaan (m²)", "Luas Dinding (m²)",
                  "Luas Atap (m²)", "Tinggi Bangunan (m)", "Orientasi",
                  "Rasio Luas Kaca", "Distribusi Kaca"]
        orient_map = {2:"Utara", 3:"Timur", 4:"Selatan", 5:"Barat"}
        values_disp = [
            f"{relative_compactness:.2f}",
            f"{surface_area:.1f}",
            f"{wall_area:.1f}",
            f"{roof_area:.2f}",
            f"{overall_height} m ({'1 lantai' if overall_height == 3.5 else '2 lantai'})",
            f"{orientation} – {orient_map[orientation]}",
            f"{glazing_area:.2f} ({glazing_area*100:.0f}%)",
            f"{glazing_distribution}",
        ]
        for label, val in zip(labels, values_disp):
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                 padding:9px 14px; background:#FFFFFF; border-radius:10px; margin-bottom:5px;
                 border:1px solid #E2E8F0; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <span style="color:#64748B; font-size:12.5px;">{label}</span>
                <span style="color:#0F172A; font-weight:700; font-size:13px;">{val}</span>
            </div>""", unsafe_allow_html=True)

        # Result summary box
        st.markdown(f"""
        <div style="background:{active_m['light']}; border:1.5px solid {active_m['color']}40;
             border-radius:12px; padding:16px 18px; margin-top:14px; text-align:center;">
            <div style="font-size:11px; font-weight:700; color:{active_m['color']}; letter-spacing:0.5px; margin-bottom:6px;">
                {active_m['emoji']} {active_model}
            </div>
            <div style="font-size:36px; font-weight:900; color:{active_m['color']}; letter-spacing:-1px;">
                {prediction:.2f}
            </div>
            <div style="font-size:13px; color:#64748B; margin-top:2px;">kWh/m²</div>
            <div style="margin-top:8px; display:inline-block; background:white; border-radius:999px;
                 padding:3px 12px; font-size:12px; font-weight:700; border:1px solid {eff_border}; color:#374151;">
                {eff_emoji} {eff_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
