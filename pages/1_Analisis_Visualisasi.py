"""
pages/1_Analisis_Visualisasi.py
Halaman 1 — Analisis & Visualisasi Data Efisiensi Energi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import inject_css, load_data, MODEL_METRICS, LABEL_MAP, PLOTLY_LAYOUT, CHART_COLORS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis & Visualisasi | Energy Efficiency",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded",
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
    st.markdown("""
    <p style="font-size:11px; color:#CBD5E1; text-align:center; line-height:1.5;">
      Mini Project 2 · Rizki Dwi Permadi
    </p>""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_data()

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">📊 ANALISIS DATA EKSPLORASI</div>
    <div class="hero-title">Analisis &amp; Visualisasi<br>Data Efisiensi Energi Bangunan</div>
    <div class="hero-sub">Jelajahi bagaimana atribut fisik bangunan mempengaruhi Cooling Load
    secara interaktif — mulai dari korelasi fitur hingga perbandingan performa ketiga model prediksi.</div>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.error("⚠️ Berkas `energy_efficiency_data.csv` tidak ditemukan di root direktori.")
    st.stop()

# ─── KPI ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Sampel",     f"{df.shape[0]:,}", "bangunan")
with k2: st.metric("Jumlah Fitur",     f"{df.shape[1]-2}",  "atribut input")
with k3: st.metric("Cooling Load Rata-rata", f"{df['Cooling_Load'].mean():.2f}", "kWh/m²")
with k4: st.metric("Cooling Load Std", f"{df['Cooling_Load'].std():.2f}", "kWh/m²")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_dataset, tab_explore, tab_dist, tab_model = st.tabs([
    "📋  Dataset",
    "🔍  Eksplorasi Fitur",
    "📉  Distribusi Target",
    "🎯  Perbandingan Model",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.markdown('<div class="section-title">Ringkasan Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Dataset <em>Energy Efficiency</em> dari UCI Machine Learning Repository — 768 sampel bangunan dengan 8 atribut fisik.</div>', unsafe_allow_html=True)

    col_tbl, col_desc = st.columns([3, 2], gap="large")

    with col_tbl:
        st.markdown("**Sampel Data (5 Baris Pertama)**")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown("**Statistika Deskriptif**")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

    with col_desc:
        st.markdown("**Deskripsi Kolom**")
        col_info = {
            "Relative_Compactness":      ("Kepadatan Relatif",        "Rasio volume terhadap luas permukaan. Makin tinggi = makin kompak."),
            "Surface_Area":              ("Luas Permukaan (m²)",       "Total luas permukaan luar bangunan."),
            "Wall_Area":                 ("Luas Dinding (m²)",         "Luas dinding eksterior bangunan."),
            "Roof_Area":                 ("Luas Atap (m²)",            "Luas atap bangunan."),
            "Overall_Height":            ("Tinggi Bangunan (m)",       "3.5 m = 1 lantai | 7.0 m = 2 lantai."),
            "Orientation":               ("Orientasi",                 "Arah hadap bangunan: 2 Utara, 3 Timur, 4 Selatan, 5 Barat."),
            "Glazing_Area":              ("Rasio Luas Kaca",           "Proporsi kaca jendela terhadap total dinding (0–0.4)."),
            "Glazing_Area_Distribution": ("Distribusi Kaca",           "Pola sebaran kaca di sekeliling bangunan (0–5)."),
            "Heating_Load":              ("Heating Load ⭐",           "Beban pemanasan (kWh/m²)."),
            "Cooling_Load":              ("Cooling Load 🎯",           "Target prediksi — beban pendinginan (kWh/m²)."),
        }
        for col, (label, desc) in col_info.items():
            is_target = col == "Cooling_Load"
            bg = "#EFF6FF" if is_target else "#FAFAFA"
            border = "#2563EB" if is_target else "#E2E8F0"
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:10px; padding:10px 14px; margin-bottom:6px;">
                <div style="font-size:11.5px; font-weight:700; color:#1E40AF; font-family:monospace;">{col}</div>
                <div style="font-size:12.5px; font-weight:600; color:#0F172A; margin-top:2px;">{label}</div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Eksplorasi Fitur
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown('<div class="section-title">Eksplorasi Pengaruh Fitur terhadap Cooling Load</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pilih karakteristik fisik bangunan untuk melihat hubungannya dengan Beban Pendinginan secara interaktif.</div>', unsafe_allow_html=True)

    features_list = [c for c in df.columns if c not in ['Heating_Load', 'Cooling_Load']]

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        sel = st.selectbox(
            "🔍 Pilih Fitur Bangunan:",
            features_list,
            format_func=lambda x: LABEL_MAP.get(x, x)
        )

    col_plot, col_info = st.columns([3, 1], gap="large")

    with col_plot:
        fig = px.scatter(
            df, x=sel, y="Cooling_Load",
            color="Overall_Height",
            color_continuous_scale=[[0, '#BFDBFE'], [0.5, '#3B82F6'], [1, '#1E3A8A']],
            labels={sel: LABEL_MAP.get(sel, sel), "Cooling_Load": "Cooling Load (kWh/m²)", "Overall_Height": "Tinggi (m)"},
            title=f"{LABEL_MAP.get(sel, sel)} vs Cooling Load",
            **PLOTLY_LAYOUT
        )
        fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color='white')))
        fig.update_layout(coloraxis_colorbar=dict(title="Tinggi (m)", thickness=14, len=0.7))
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        corr_val = df[sel].corr(df['Cooling_Load'])
        direction = "positif 📈" if corr_val > 0 else "negatif 📉"
        strength  = "Sangat kuat" if abs(corr_val) > 0.7 else ("Sedang" if abs(corr_val) > 0.4 else "Lemah")
        bar_color = "#16A34A" if corr_val > 0 else "#DC2626"
        bar_pct   = abs(corr_val) * 100
        st.markdown(f"""
        <div class="white-card" style="margin-top:48px;">
            <h4>💡 Insight Statistik</h4>
            <div style="font-size:12px; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Fitur</div>
            <div style="font-size:13px; font-weight:700; color:#1E40AF; margin-bottom:14px;">{LABEL_MAP.get(sel, sel)}</div>
            <div style="font-size:12px; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Korelasi Pearson</div>
            <div style="font-size:32px; font-weight:900; color:{bar_color}; letter-spacing:-1px;">{corr_val:.3f}</div>
            <div style="background:#F1F5F9; border-radius:999px; height:6px; margin:8px 0;">
                <div style="width:{bar_pct:.0f}%; background:{bar_color}; height:6px; border-radius:999px; transition:width 0.4s;"></div>
            </div>
            <div style="font-size:13px; color:#334155; margin-bottom:10px;"><strong>{strength}</strong> &amp; {direction}</div>
            <div style="font-size:12px; color:#64748B; line-height:1.6; padding-top:10px; border-top:1px solid #F1F5F9;">
                Warna titik menunjukkan tinggi bangunan. Bangunan lebih tinggi (biru tua) umumnya memiliki Cooling Load lebih tinggi.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Correlation bar chart
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Korelasi Semua Fitur dengan Cooling Load</div>', unsafe_allow_html=True)

    feat_cols = [c for c in df.columns if c not in ['Heating_Load', 'Cooling_Load']]
    corrs     = df[feat_cols].corrwith(df['Cooling_Load']).sort_values()
    colors_bar = ['#DC2626' if v < 0 else '#2563EB' for v in corrs.values]

    fig_bar = go.Figure(go.Bar(
        x=[LABEL_MAP.get(c, c) for c in corrs.index],
        y=corrs.values,
        marker_color=colors_bar,
        text=[f"{v:.3f}" for v in corrs.values],
        textposition='outside',
    ))
    fig_bar.add_hline(y=0, line_color='#94A3B8', line_width=1)
    fig_bar.update_layout(
        title="Koefisien Korelasi Pearson terhadap Cooling Load",
        xaxis_title="Fitur", yaxis_title="Korelasi (r)",
        yaxis=dict(range=[-1.1, 1.1], gridcolor='#F1F5F9', zerolinecolor='#E2E8F0'),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matriks Korelasi Antar Fitur</div>', unsafe_allow_html=True)

    corr_df  = df.drop(columns=['Heating_Load']).corr()
    tick_labels = [LABEL_MAP.get(c, c) for c in corr_df.columns]
    fig_hm = go.Figure(go.Heatmap(
        z             = corr_df.values,
        x             = tick_labels,
        y             = tick_labels,
        colorscale    = 'RdBu',
        zmid=0, zmin=-1, zmax=1,
        colorbar      = dict(thickness=14, title="r", len=0.85),
        text          = np.round(corr_df.values, 2),
        texttemplate  = "%{text}",
        hovertemplate = "%{x}<br>%{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(
        title="Matriks Korelasi (seluruh fitur + Cooling Load)",
        xaxis_tickangle=-30, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Distribusi Target
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.markdown('<div class="section-title">Distribusi Beban Pendinginan (Cooling Load)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Memahami sebaran nilai target prediksi pada 768 sampel bangunan dalam dataset.</div>', unsafe_allow_html=True)

    col_hist, col_stats = st.columns([3, 1], gap="large")

    with col_hist:
        fig_hist = px.histogram(
            df, x="Cooling_Load", nbins=35, marginal="box",
            color_discrete_sequence=['#3B82F6'],
            labels={"Cooling_Load": "Cooling Load (kWh/m²)"},
            title="Distribusi Cooling Load — 768 Sampel Bangunan",
            **PLOTLY_LAYOUT
        )
        fig_hist.update_traces(marker_line_color='white', marker_line_width=0.5, opacity=0.85)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stats:
        st.markdown("**Statistik Ringkasan**")
        stats = df['Cooling_Load'].describe()
        for label, key, icon in [
            ("Minimum",  "min",   "⬇️"),
            ("Q1 (25%)", "25%",   "📊"),
            ("Median",   "50%",   "📍"),
            ("Q3 (75%)", "75%",   "📊"),
            ("Maksimum", "max",   "⬆️"),
            ("Rata-rata","mean",  "⚖️"),
            ("Std Dev",  "std",   "〰️"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                 padding:9px 14px; background:#FFFFFF; border-radius:10px; margin-bottom:6px;
                 border:1px solid #E2E8F0; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <span style="color:#64748B; font-size:12.5px;">{icon} {label}</span>
                <span style="color:#1E40AF; font-weight:700; font-size:14px;">{stats[key]:.2f}</span>
            </div>""", unsafe_allow_html=True)

    # Violin per height
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribusi per Tinggi Bangunan</div>', unsafe_allow_html=True)
    fig_vio = px.violin(
        df, x="Overall_Height", y="Cooling_Load",
        color="Overall_Height", box=True, points="outliers",
        color_discrete_sequence=['#3B82F6', '#F59E0B'],
        labels={"Overall_Height": "Tinggi Bangunan (m)", "Cooling_Load": "Cooling Load (kWh/m²)"},
        title="Violin Plot — Cooling Load berdasarkan Tinggi Bangunan",
        **PLOTLY_LAYOUT
    )
    fig_vio.update_layout(showlegend=False)
    st.plotly_chart(fig_vio, use_container_width=True)

    # Box plot per orientation
    fig_box = px.box(
        df, x="Orientation", y="Cooling_Load", color="Orientation",
        color_discrete_sequence=CHART_COLORS,
        labels={"Orientation": "Orientasi Bangunan", "Cooling_Load": "Cooling Load (kWh/m²)"},
        title="Box Plot — Cooling Load berdasarkan Orientasi Bangunan",
        **PLOTLY_LAYOUT
    )
    orient_labels = {2: "2 – Utara", 3: "3 – Timur", 4: "4 – Selatan", 5: "5 – Barat"}
    fig_box.for_each_trace(lambda t: t.update(name=orient_labels.get(int(t.name), t.name)))
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Perbandingan Model
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-title">Perbandingan Kinerja Tiga Model Prediksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Hasil evaluasi pada data uji (154 sampel) berdasarkan eksperimen di notebook MP_02_Fixed.</div>', unsafe_allow_html=True)

    # Cards
    mc1, mc2, mc3 = st.columns(3, gap="medium")
    for col_w, (name, m) in zip([mc1, mc2, mc3], MODEL_METRICS.items()):
        is_best = name == 'Random Forest Regressor'
        best_tag = '<div style="display:inline-block;background:#FEF3C7;border:1px solid #FCD34D;color:#B45309;border-radius:999px;padding:2px 10px;font-size:10.5px;font-weight:700;margin-bottom:8px;">🏆 TERBAIK</div>' if is_best else ''
        with col_w:
            st.markdown(f"""
            <div class="white-card" style="text-align:center; border-top: 3px solid {m['color']}; padding: 24px 20px;">
                {best_tag}
                <div style="font-size:26px;">{m['emoji']}</div>
                <div style="font-size:13px; font-weight:700; color:#0F172A; margin:10px 0 4px; line-height:1.3;">{name}</div>
                <div style="font-size:40px; font-weight:900; color:{m['color']}; letter-spacing:-1.5px;">{m['r2']*100:.1f}%</div>
                <div style="font-size:11px; color:#94A3B8; margin-bottom:14px;">R² Score</div>
                <div style="display:flex; justify-content:space-around; padding:12px 0; border-top:1px solid #F1F5F9; border-bottom:1px solid #F1F5F9;">
                    <div><div style="font-size:10px;color:#94A3B8;font-weight:600;">MAE</div><div style="font-size:15px;font-weight:800;color:#334155;">{m['mae']}</div></div>
                    <div><div style="font-size:10px;color:#94A3B8;font-weight:600;">RMSE</div><div style="font-size:15px;font-weight:800;color:#334155;">{m['rmse']}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2, gap="medium")

    names  = list(MODEL_METRICS.keys())
    colors = [m['color'] for m in MODEL_METRICS.values()]

    with ch1:
        r2s = [m['r2'] for m in MODEL_METRICS.values()]
        fig_r2 = go.Figure(go.Bar(
            x=[n.replace(' (', '<br>(') for n in names],
            y=r2s,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v*100:.2f}%" for v in r2s],
            textposition='outside',
        ))
        fig_r2.update_layout(
            title="R² Score — Akurasi (Semakin Tinggi Semakin Baik)",
            yaxis=dict(range=[0.8, 1.02], tickformat='.0%', gridcolor='#F1F5F9'),
            showlegend=False, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with ch2:
        maes  = [m['mae']  for m in MODEL_METRICS.values()]
        rmses = [m['rmse'] for m in MODEL_METRICS.values()]
        short_names = ['Ridge', 'Random\nForest', 'ANN']
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(name='MAE',  x=short_names, y=maes,  marker_color='#3B82F6', marker_line_width=0, text=[f"{v:.3f}" for v in maes],  textposition='outside'))
        fig_err.add_trace(go.Bar(name='RMSE', x=short_names, y=rmses, marker_color='#F97316', marker_line_width=0, text=[f"{v:.3f}" for v in rmses], textposition='outside'))
        fig_err.update_layout(
            title="MAE & RMSE — Error (Semakin Rendah Semakin Baik)",
            barmode='group', yaxis=dict(gridcolor='#F1F5F9'),
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#E2E8F0', borderwidth=1),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # Radar comparison
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Radar Chart Perbandingan Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Perbandingan visual ketiga model berdasarkan metrik yang dinormalisasi (semakin luar semakin baik).</div>', unsafe_allow_html=True)

    # Normalize: R2 higher is better; MAE, RMSE lower is better → invert
    r2_vals   = [m['r2']  for m in MODEL_METRICS.values()]
    mae_vals  = [m['mae'] for m in MODEL_METRICS.values()]
    rmse_vals = [m['rmse']for m in MODEL_METRICS.values()]
    mae_norm  = [1 - (v - min(mae_vals))  / (max(mae_vals)  - min(mae_vals))  for v in mae_vals]
    rmse_norm = [1 - (v - min(rmse_vals)) / (max(rmse_vals) - min(rmse_vals)) for v in rmse_vals]
    r2_norm   = [(v - min(r2_vals)) / (max(r2_vals) - min(r2_vals)) for v in r2_vals]

    categories = ['R² Score', 'Low MAE', 'Low RMSE', 'R² Score']
    fig_radar = go.Figure()
    for i, (name, m) in enumerate(MODEL_METRICS.items()):
        vals = [r2_norm[i], mae_norm[i], rmse_norm[i], r2_norm[i]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories,
            fill='toself', name=name,
            line=dict(color=m['color'], width=2),
            fillcolor=m['color'] + '22',
            marker=dict(size=6, color=m['color']),
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(248,250,252,0.8)',
            radialaxis=dict(visible=True, range=[0, 1.1], showticklabels=False, gridcolor='#E2E8F0'),
            angularaxis=dict(gridcolor='#E2E8F0', tickfont=dict(size=13, color='#334155')),
        ),
        showlegend=True,
        title="Radar Perbandingan Model (dinormalisasi, lebih luar = lebih baik)",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_radar, use_container_width=True)
