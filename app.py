"""
CL Circular — Dashboard de Siniestralidad en Transporte de Mercancías
Fuente de datos: CNSF / AMIS — Bases de Mercancías 2015–2024

Instrucciones para ejecutar:
    pip install streamlit pandas openpyxl plotly scikit-learn pillow
    streamlit run app.py

Coloca este archivo junto a los xlsx de datos y el logo:
    Logo-Cl-Circular-1.png
    2015_Mercancias_Bases.xlsx  ... 2024_Mercancias_Bases__3_.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import os
import base64

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CL Circular · Dashboard Estratégico",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  PALETA DE COLORES BRAND CL CIRCULAR
# ─────────────────────────────────────────────
CL = {
    "navy":      "#0D2B4E",
    "blue":      "#1A5FA8",
    "teal":      "#2BA8C5",
    "green_dark":"#2E7D32",
    "green":     "#5BAD44",
    "green_lt":  "#8DC63F",
    "bg":        "#0A1628",
    "surface":   "#0F2035",
    "surface2":  "#152840",
    "border":    "#1E3A5F",
    "text":      "#E8F1FB",
    "text_dim":  "#7EA8C9",
    "red":       "#E53935",
    "orange":    "#F57C00",
    "gold":      "#F9A825",
}

COLOR_SEQ = [CL["teal"], CL["green"], CL["blue"], CL["green_lt"],
             CL["green_dark"], CL["navy"], CL["orange"], CL["gold"]]

# ─────────────────────────────────────────────
#  CSS GLOBAL
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    .stApp {{
        background-color: {CL['bg']};
        font-family: 'DM Sans', sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CL['surface']} 0%, {CL['navy']} 100%);
        border-right: 1px solid {CL['border']};
    }}
    section[data-testid="stSidebar"] * {{ color: {CL['text']} !important; }}
    h1, h2, h3 {{ font-family: 'Syne', sans-serif !important; color: {CL['text']} !important; }}
    p, li, span, div {{ color: {CL['text']}; }}
    [data-testid="stMetric"] {{
        background: {CL['surface']};
        border: 1px solid {CL['border']};
        border-radius: 12px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {CL['text_dim']} !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    [data-testid="stMetricValue"] {{
        color: {CL['text']} !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {CL['surface']};
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {CL['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: {CL['text_dim']} !important;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        padding: 8px 20px;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: {CL['blue']} !important;
        color: {CL['text']} !important;
    }}
    .stSelectbox > div > div {{
        background: {CL['surface2']};
        border: 1px solid {CL['border']};
        border-radius: 8px;
        color: {CL['text']};
    }}
    hr {{ border-color: {CL['border']}; margin: 0.5rem 0; }}
    .cl-card {{
        background: {CL['surface']};
        border: 1px solid {CL['border']};
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }}
    .cl-card-accent {{
        background: linear-gradient(135deg, {CL['navy']} 0%, {CL['surface2']} 100%);
        border: 1px solid {CL['teal']}44;
        border-left: 3px solid {CL['teal']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }}
    .cl-insight {{
        background: linear-gradient(135deg, {CL['navy']} 0%, #0a1e35 100%);
        border: 1px solid {CL['green']}55;
        border-left: 4px solid {CL['green']};
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
    }}
    .cl-insight-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {CL['green_lt']};
        margin-bottom: 4px;
    }}
    .cl-insight-text {{ font-size: 0.92rem; color: {CL['text']}; line-height: 1.5; }}
    .cl-section-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: {CL['text']};
        margin-bottom: 4px;
    }}
    .cl-section-sub {{ font-size: 0.88rem; color: {CL['text_dim']}; margin-bottom: 18px; }}
    .cl-tag {{
        display: inline-block;
        background: {CL['teal']}22;
        border: 1px solid {CL['teal']}55;
        color: {CL['teal']};
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }}
    .cl-badge-red {{
        background: {CL['red']}22;
        border: 1px solid {CL['red']}55;
        color: #ff6b6b;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .cl-badge-green {{
        background: {CL['green']}22;
        border: 1px solid {CL['green']}55;
        color: {CL['green_lt']};
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .cl-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 20px 0;
        border-bottom: 1px solid {CL['border']};
        margin-bottom: 28px;
    }}
    .cl-topbar-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        color: {CL['text_dim']};
        letter-spacing: 0.05em;
    }}
    .cl-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .cl-table th {{
        background: {CL['surface2']};
        color: {CL['text_dim']};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.72rem;
        padding: 10px 14px;
        border-bottom: 1px solid {CL['border']};
        text-align: left;
    }}
    .cl-table td {{
        padding: 9px 14px;
        border-bottom: 1px solid {CL['border']}66;
        color: {CL['text']};
    }}
    .cl-table tr:hover td {{ background: {CL['surface2']}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#FFFFFF"),
    margin=dict(l=10, r=10, t=40, b=10),
)
LEGEND_BASE = dict(
    bgcolor="rgba(0,0,0,0)",
    bordercolor=CL["border"],
    borderwidth=1,
    font=dict(size=11, color="#FFFFFF"),
)

def axis_style(title=""):
    return dict(
        title=title,
        gridcolor=CL["border"],
        linecolor=CL["border"],
        tickcolor="#FFFFFF",
        tickfont=dict(size=10, color="#FFFFFF"),
        titlefont=dict(size=11, color="#FFFFFF"),
        zeroline=False,
    )

def fmt_millon(v):
    return f"${v/1000:.1f}B" if v >= 1000 else f"${v:.0f}M"

def safe_pct(numerator, denominator):
    return (numerator / denominator * 100) if denominator else 0.0

def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(255,255,255,{alpha})"
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────
#  CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando bases de datos CNSF/AMIS…")
def load_data():
    col_names = [
        'MONEDA','FORMA_VENTA','TIPO_PRONOSTICO','TIPO_SEGURO','ENTIDAD',
        'TIPO_MERCANCIA','MEDIO_TRANSPORTE','COBERTURA','LUGAR',
        'CAUSA_SINIESTRO','NUM_SINIESTROS','MONTO_SINIESTRO','GASTOS_AJUSTE',
        'SALVAMENTOS','MONTO_PAGADO','MONTO_DEDUCIBLE','MONTO_COASEGURO',
        'MONTO_REASEGURO','MONTO_RECUPERADO'
    ]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_map = {}
    for y in range(2015, 2025):
        for fname in os.listdir(base_dir):
            if str(y) in fname and fname.endswith('.xlsx') and 'Mercancias' in fname:
                file_map[y] = os.path.join(base_dir, fname)
                break

    dfs = []
    for year, path in sorted(file_map.items()):
        try:
            df = pd.read_excel(path, sheet_name='Siniestros', header=1, skiprows=[0])
            df.columns = col_names[:len(df.columns)]
            df['AÑO'] = year
            if year == 2024:
                df = df.drop_duplicates()
            dfs.append(df)
        except Exception as e:
            st.warning(f"No se pudo cargar {year}: {e}")

    if not dfs:
        st.error("No se encontraron archivos de datos.")
        st.stop()

    all_df = pd.concat(dfs, ignore_index=True)
    for col in ['NUM_SINIESTROS','MONTO_SINIESTRO','GASTOS_AJUSTE','MONTO_PAGADO']:
        all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

    entity_map = {
        'Distrito Federal': 'Ciudad de México','DISTRITO FEDERAL': 'Ciudad de México',
        'Mexico': 'Estado de México','MEXICO': 'Estado de México',
        'JALISCO': 'Jalisco','NUEVO LEON': 'Nuevo León','Nuevo Leon': 'Nuevo León',
        'PUEBLA': 'Puebla','GUANAJUATO': 'Guanajuato','AGUASCALIENTES': 'Aguascalientes',
        'CHIHUAHUA': 'Chihuahua','COAHUILA': 'Coahuila','BAJA CALIFORNIA': 'Baja California',
        'SONORA': 'Sonora','TAMAULIPAS': 'Tamaulipas','VERACRUZ': 'Veracruz',
        'QUERETARO': 'Querétaro','Querétaro': 'Querétaro',
        'SAN LUIS POTOSI': 'San Luis Potosí','San Luis Potosí': 'San Luis Potosí',
        'MICHOACAN': 'Michoacán',
    }
    all_df['ENTIDAD_NORM'] = all_df['ENTIDAD'].map(
        lambda x: entity_map.get(str(x).strip(), str(x).strip())
    )

    def categorize(c):
        c = str(c).lower()
        if 'robo' in c and ('violencia' in c or 'asalto' in c):
            return 'Robo con violencia'
        elif 'robo' in c:
            return 'Robo sin violencia'
        elif any(k in c for k in ['colisión','volcadura','descarrilamiento','colision']):
            return 'Colisión / Volcadura'
        elif any(k in c for k in ['rotura','rajadura','abolladura']):
            return 'Daño físico'
        elif 'mojadura' in c or 'humedad' in c:
            return 'Mojadura'
        elif 'refrigeración' in c or 'temperatura' in c:
            return 'Falla refrigeración'
        elif 'maniobra' in c or 'carga' in c:
            return 'Maniobras'
        else:
            return 'Otras causas'

    all_df['CATEGORIA'] = all_df['CAUSA_SINIESTRO'].apply(categorize)
    terr = all_df[all_df['MEDIO_TRANSPORTE'].str.contains('Terrestre', na=False, case=False)].copy()
    return all_df, terr


SEG_KEYWORDS = {
    "🍺  Cerveza": [
        'Cervezas y/o refrescos y/o vinos y/o licores y/o jugos en latas',
        'Cerveza embotellada',
    ],
    "🌾  Harinas": [
        'Harina en sacos y/o a granel',
        'Harina de pescado',
    ],
    "🔩  Autopartes": [
        'Refacciones automotrices ',
        'Refacciones automotrices',
        'Automóviles y camiones a bordo de vehículos',
        'Maquinaria nueva y refacciones',
        'Llantas y artefactos de hule',
        'Lubricantes automotrices',
        'Máquinas de coser y/o de tejer, partes y refacciones',
    ],
}
SEG_COLORS = {
    "🍺  Cerveza":    CL["orange"],
    "🌾  Harinas":    CL["green"],
    "🔩  Autopartes": CL["teal"],
}


# ─────────────────────────────────────────────
#  MODELO ML — FUNCIONES
# ─────────────────────────────────────────────
def holt_winters_doble(y: np.ndarray, alpha: float, beta: float, h: int = 1):
    """Suavizamiento exponencial doble (Holt-Winters sin estacionalidad).

    Ecuaciones:
        L_t = α·y_t + (1−α)·(L_{t-1} + T_{t-1})
        T_t = β·(L_t − L_{t-1}) + (1−β)·T_{t-1}
        ŷ_{t+h} = L_t + h·T_t

    Args:
        y: serie temporal (array 1D)
        alpha: suavizamiento del nivel
        beta:  suavizamiento de la tendencia
        h:     horizonte de pronóstico
    """
    n = len(y)
    L, T, fitted = np.zeros(n), np.zeros(n), np.zeros(n)
    L[0] = y[0]
    T[0] = (y[1] - y[0]) if n > 1 else 0.0
    fitted[0] = L[0]
    for t in range(1, n):
        L[t] = alpha * y[t] + (1 - alpha) * (L[t-1] + T[t-1])
        T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
        fitted[t] = L[t] + T[t]
    forecast = np.array([L[-1] + (i + 1) * T[-1] for i in range(h)])
    return fitted, forecast, L[-1], T[-1]


def optimizar_holt(y: np.ndarray) -> dict:
    """Búsqueda en grilla para α y β óptimos minimizando RMSE en muestra."""
    best = {'alpha': 0.5, 'beta': 0.3, 'rmse': np.inf}
    for a in np.arange(0.05, 1.0, 0.05):
        for b in np.arange(0.05, 1.0, 0.05):
            fitted, _, _, _ = holt_winters_doble(y, a, b)
            rmse = np.sqrt(np.mean((y - fitted) ** 2))
            if rmse < best['rmse']:
                best = {'alpha': float(a), 'beta': float(b), 'rmse': rmse}
    return best


@st.cache_data(show_spinner="Entrenando modelo predictivo…")
def entrenar_modelo(monto_m_tuple: tuple) -> dict:
    """Entrena ensamble Holt-Winters + Ridge Polinómica y proyecta 2025-2027.

    Cachea resultados para evitar re-entrenamiento en cada interacción.

    Retorna:
        dict con proyecciones, bandas de confianza y métricas de validación.
    """
    y_all  = np.array(monto_m_tuple)
    t_all  = np.arange(len(y_all)).reshape(-1, 1)

    TRAIN_END  = 7   # 2015–2021
    PROJ_STEPS = 3
    PROJ_YEARS = [2025, 2026, 2027]

    y_train = y_all[:TRAIN_END]
    y_test  = y_all[TRAIN_END:]
    t_train = t_all[:TRAIN_END]
    t_test  = t_all[TRAIN_END:]
    t_proj  = np.arange(len(y_all), len(y_all) + PROJ_STEPS).reshape(-1, 1)

    # ── Holt-Winters (serie completa) ──
    params = optimizar_holt(y_all)
    _, hw_proj, _, _ = holt_winters_doble(y_all, params['alpha'], params['beta'], h=PROJ_STEPS)

    # ── Ridge Polinómica grado 2 (serie completa) ──
    ridge_pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=2, include_bias=True)),
        ('ridge', Ridge(alpha=1.0))
    ])
    ridge_pipe.fit(t_all, y_all)
    ridge_proj = ridge_pipe.predict(t_proj)

    # ── Proyección ensamble ──
    proj_base = (hw_proj + ridge_proj) / 2

    # ── Walk-forward validation (2019–2023, índices 4-8) ──
    wf_errors = []
    for pred_idx in range(4, 9):
        y_tr = y_all[:pred_idx]
        t_tr = t_all[:pred_idx]
        t_pr = t_all[pred_idx:pred_idx + 1]
        p    = optimizar_holt(y_tr)
        _, hw_p, _, _ = holt_winters_doble(y_tr, p['alpha'], p['beta'], h=1)
        pipe_wf = Pipeline([
            ('poly',  PolynomialFeatures(degree=2, include_bias=True)),
            ('ridge', Ridge(alpha=1.0))
        ])
        pipe_wf.fit(t_tr, y_tr)
        ensemble_pred = (hw_p[0] + pipe_wf.predict(t_pr)[0]) / 2
        wf_errors.append((ensemble_pred - y_all[pred_idx]) / y_all[pred_idx])

    mape_wf = float(np.mean(np.abs(wf_errors)) * 100)

    # ── Bandas de confianza bootstrap (n=5000) ──
    np.random.seed(42)
    boot = np.zeros((5000, PROJ_STEPS))
    for i in range(5000):
        samp = np.random.choice(wf_errors, size=PROJ_STEPS, replace=True)
        for h in range(PROJ_STEPS):
            boot[i, h] = proj_base[h] * (1 + samp[h] * (1 + 0.1 * h))
    ci_lower = np.percentile(boot, 10, axis=0)
    ci_upper = np.percentile(boot, 90, axis=0)

    # ── Validación en test 2022–2024 ──
    _, hw_test, _, _ = holt_winters_doble(y_train, params['alpha'], params['beta'], h=len(y_test))
    ridge_test = ridge_pipe.predict(t_test)
    ensemble_test = (hw_test + ridge_test) / 2
    mae_test  = float(mean_absolute_error(y_test, ensemble_test))
    mape_test = float(np.mean(np.abs((y_test - ensemble_test) / y_test)) * 100)
    r2_test   = float(r2_score(y_test, ensemble_test))

    return {
        'years':         list(range(2015, 2025)),
        'y_all':         y_all.tolist(),
        'proj_years':    PROJ_YEARS,
        'proj_base':     proj_base.tolist(),
        'ci_lower':      ci_lower.tolist(),
        'ci_upper':      ci_upper.tolist(),
        'hw_proj':       hw_proj.tolist(),
        'ridge_proj':    ridge_proj.tolist(),
        'mape_wf':       mape_wf,
        'mae_test':      mae_test,
        'mape_test':     mape_test,
        'r2_test':       r2_test,
        'alpha_hw':      float(params['alpha']),
        'beta_hw':       float(params['beta']),
        'test_years':    [2022, 2023, 2024],
        'y_test':        y_test.tolist(),
        'ensemble_test': ensemble_test.tolist(),
        'wf_errors':     wf_errors,
    }


def calcular_escenario(proj_base: list, reduccion: float, penetracion: list):
    """Aplica reducción de siniestralidad con sensor sobre proyección base.

    Args:
        proj_base:   Monto proyectado sin sensor por año [$M].
        reduccion:   Reducción anual del monto (0.05 = 5%).
        penetracion: Fracción del mercado con sensor [0-1] por año.

    Returns:
        (proj_con_sensor, ahorro) — ambos en $M por año.
    """
    pb = np.array(proj_base)
    pn = np.array(penetracion)
    proj_con = pb * (1 - reduccion * pn)
    return proj_con.tolist(), (pb - proj_con).tolist()


# ─────────────────────────────────────────────
#  LOGO
# ─────────────────────────────────────────────
def get_logo_b64():
    for p in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo-Cl-Circular-1.png"),
        "Logo-Cl-Circular-1.png",
    ]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None


# ─────────────────────────────────────────────
#  CARGA EFECTIVA
# ─────────────────────────────────────────────
all_data, terr = load_data()
logo_b64 = get_logo_b64()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    if logo_b64:
        st.markdown(
            f'<div style="text-align:center;padding:20px 0 8px 0;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="width:160px;filter:brightness(1.1);">'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<p style="text-align:center;font-size:0.7rem;color:{CL["text_dim"]};'
        f'letter-spacing:0.12em;text-transform:uppercase;padding-bottom:16px;">Dashboard Estratégico</p>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<hr style="border-color:{CL["border"]};margin:0 0 20px 0;">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.7rem;color:{CL["text_dim"]};text-transform:uppercase;'
        f'letter-spacing:0.1em;font-weight:600;margin-bottom:8px;">Navegación</p>',
        unsafe_allow_html=True,
    )
    seccion = st.radio(
        label="",
        options=[
            "01 · Panorama Nacional",
            "02 · Mapa de Riesgo",
            "03 · Análisis de Causas",
            "04 · Industrias Objetivo",
            "05 · Caso del Sensor",
            "06 · Proyección ML",
        ],
        label_visibility="collapsed",
    )
    st.markdown(f'<hr style="border-color:{CL["border"]};margin:20px 0;">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.7rem;color:{CL["text_dim"]};text-transform:uppercase;'
        f'letter-spacing:0.1em;font-weight:600;margin-bottom:8px;">Filtros Globales</p>',
        unsafe_allow_html=True,
    )
    year_range = st.slider("Periodo de análisis", min_value=2015, max_value=2024, value=(2015, 2024))
    st.markdown(f'<hr style="border-color:{CL["border"]};margin:16px 0;">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.68rem;color:{CL["text_dim"]};line-height:1.6;">'
        f'Fuente: CNSF / AMIS<br>Sistema Estadístico de Transporte<br>de Mercancías 2015–2024<br><br>'
        f'<span style="color:{CL["border"]};">Uso confidencial · CL Circular</span></p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  FILTRO POR AÑO
# ─────────────────────────────────────────────
df = terr[(terr['AÑO'] >= year_range[0]) & (terr['AÑO'] <= year_range[1])].copy()


# ═══════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — PANORAMA NACIONAL
# ═══════════════════════════════════════════════════════════════════════
if seccion == "01 · Panorama Nacional":

    st.markdown(f"""
    <div class="cl-topbar">
      <div>
        <div class="cl-section-title">Panorama Nacional</div>
        <div class="cl-section-sub">Siniestralidad en transporte terrestre de mercancías · México {year_range[0]}–{year_range[1]}</div>
      </div>
      <div><span class="cl-tag">CNSF / AMIS</span><span class="cl-tag">Terrestre</span></div>
    </div>
    """, unsafe_allow_html=True)

    total_monto = df['MONTO_SINIESTRO'].sum()
    total_sin   = df['NUM_SINIESTROS'].sum()
    total_gasto = df['GASTOS_AJUSTE'].sum()
    robo_monto  = df[df['CAUSA_SINIESTRO'].str.contains(
        'robo total con violencia|robo parcial con violencia|robo total sin violencia|robo parcial sin violencia|extravío',
        case=False, na=False, regex=True)]['MONTO_SINIESTRO'].sum()
    pct_robo    = safe_pct(robo_monto, total_monto)
    costo_prom  = total_monto / total_sin if total_sin > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Monto Total Siniestrado", f"${total_monto/1e9:.1f}B")
    with c2: st.metric("Siniestros Registrados", f"{total_sin/1e3:.0f}K")
    with c3: st.metric("Gastos de Ajuste / Peritaje", f"${total_gasto/1e6:.0f}M")
    with c4: st.metric("% Causado por Robo", f"{pct_robo:.1f}%")
    with c5: st.metric("Costo Promedio por Siniestro", f"${costo_prom/1000:.0f}K")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        trend = df.groupby('AÑO').agg(
            monto=('MONTO_SINIESTRO','sum'),
            siniestros=('NUM_SINIESTROS','sum'),
            gastos=('GASTOS_AJUSTE','sum'),
        ).reset_index()
        trend['monto_m']  = trend['monto'] / 1e6
        trend['gastos_m'] = trend['gastos'] / 1e6

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        highlight_years = [2017, 2018, 2019]
        bar_colors = [CL["red"] if y in highlight_years else CL["teal"] for y in trend['AÑO']]

        fig.add_trace(go.Bar(
            x=trend['AÑO'], y=trend['monto_m'],
            name="Monto siniestrado ($M)",
            marker_color=bar_colors, marker_opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Monto: $%{y:.0f}M<extra></extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=trend['AÑO'], y=trend['gastos_m'],
            name="Gastos de ajuste ($M)",
            line=dict(color=CL["green_lt"], width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=6, color=CL["green_lt"]),
            hovertemplate="<b>%{x}</b><br>Gastos ajuste: $%{y:.0f}M<extra></extra>",
        ), secondary_y=True)
        fig.add_annotation(
            text="Mayor crecimiento: 2017–2019",
            x=2018, y=trend['monto_m'].max() + 100,
            showarrow=True, arrowhead=2, ax=0, ay=-40,
            font=dict(size=10, color=CL["text"]),
            bgcolor="rgba(0,0,0,0.7)", bordercolor=CL["border"], borderwidth=1, borderpad=4,
        )
        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Evolución del Monto Siniestrado y Costo de Peritaje", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(**axis_style("Año"), dtick=1),
            yaxis=dict(**axis_style("Monto siniestrado ($M)"), tickprefix="$"),
            yaxis2=dict(**axis_style("Gastos ajuste ($M)"), tickprefix="$", overlaying="y", side="right", showgrid=False),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cat_totals = df.groupby('CATEGORIA')['MONTO_SINIESTRO'].sum().reset_index()
        cat_totals = cat_totals.sort_values('MONTO_SINIESTRO', ascending=False)
        cat_totals['monto_m'] = cat_totals['MONTO_SINIESTRO'] / 1e6
        cat_colors = {
            'Robo con violencia': CL["red"], 'Robo sin violencia': CL["orange"],
            'Colisión / Volcadura': CL["teal"], 'Daño físico': CL["blue"],
            'Maniobras': CL["green"], 'Mojadura': CL["green_dark"],
            'Falla refrigeración': CL["gold"], 'Otras causas': CL["text_dim"],
        }
        fig2 = go.Figure(go.Pie(
            labels=cat_totals['CATEGORIA'], values=cat_totals['monto_m'],
            hole=0.58,
            marker=dict(colors=[cat_colors.get(c, CL["text_dim"]) for c in cat_totals['CATEGORIA']],
                        line=dict(color=CL["bg"], width=2)),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>$%{value:.0f}M<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Causas por Monto", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            height=340, showlegend=True,
            legend=dict(font=dict(size=9), orientation="v", x=1.0),
            annotations=[dict(
                text=f"<b>{pct_robo:.0f}%</b><br><span style='font-size:9px'>ROBO</span>",
                x=0.5, y=0.5, font=dict(size=13, color=CL["text"], family="Syne"), showarrow=False,
            )],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div class="cl-insight">
      <div class="cl-insight-title">📈 Insight clave</div>
      <div class="cl-insight-text">Entre 2017 y 2019 se observó el mayor crecimiento en siniestros, lo que evidencia un periodo crítico de incremento en riesgo y costos para aseguradoras.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cl-insight">
      <div class="cl-insight-title">📌 Insight clave</div>
      <div class="cl-insight-text">El robo con violencia acumula el mayor monto siniestrado en todos los años del periodo. La tendencia se dispara en 2017–2019, se contrae en 2020 (pandemia) y repunta con fuerza desde 2021. Esto indica un problema estructural, no cíclico.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    causa_yr = df.groupby(['AÑO','CATEGORIA'])['MONTO_SINIESTRO'].sum().reset_index()
    causa_yr['monto_m'] = causa_yr['MONTO_SINIESTRO'] / 1e6
    top_cats   = ['Robo con violencia', 'Colisión / Volcadura', 'Robo sin violencia', 'Daño físico', 'Maniobras']
    cat_colors2 = {
        'Robo con violencia': CL["red"], 'Robo sin violencia': CL["orange"],
        'Colisión / Volcadura': CL["teal"], 'Daño físico': CL["blue"], 'Maniobras': CL["green"],
    }
    causa_yr_f = causa_yr[causa_yr['CATEGORIA'].isin(top_cats)]
    fig3 = go.Figure()
    for cat in reversed(top_cats):
        sub = causa_yr_f[causa_yr_f['CATEGORIA'] == cat]
        fig3.add_trace(go.Scatter(
            x=sub['AÑO'], y=sub['monto_m'], name=cat,
            mode='lines+markers',
            line=dict(color=cat_colors2[cat], width=2.5), marker=dict(size=6),
            fill='tonexty' if cat == top_cats[0] else None,
            fillcolor=hex_to_rgba(cat_colors2[cat], 0.12),
            hovertemplate=f"<b>{cat}</b><br>Año: %{{x}}<br>$%{{y:.0f}}M<extra></extra>",
        ))
    fig3.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Monto Siniestrado por Causa — Evolución Anual", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
        xaxis=dict(**axis_style("Año"), dtick=1),
        yaxis=dict(**axis_style("Monto ($M)"), tickprefix="$"),
        height=320, hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📊 Ver tabla completa de datos por año"):
        tbl = df.groupby('AÑO').agg(
            Siniestros=('NUM_SINIESTROS','sum'),
            Monto_M=('MONTO_SINIESTRO', lambda x: round(x.sum()/1e6,1)),
            Gastos_Ajuste_M=('GASTOS_AJUSTE', lambda x: round(x.sum()/1e6,1)),
        ).reset_index()
        tbl['Ratio_Gastos_%'] = (tbl['Gastos_Ajuste_M'] / tbl['Monto_M'] * 100).round(2)
        tbl.columns = ['Año','Siniestros','Monto ($M)','Gastos Ajuste ($M)','% Gastos/Monto']
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 — MAPA DE RIESGO
# ═══════════════════════════════════════════════════════════════════════
elif seccion == "02 · Mapa de Riesgo":

    st.markdown(f"""
    <div class="cl-topbar">
      <div>
        <div class="cl-section-title">Mapa de Riesgo Geográfico</div>
        <div class="cl-section-sub">Concentración de siniestralidad por entidad federativa · {year_range[0]}–{year_range[1]}</div>
      </div>
      <div><span class="cl-tag">Por Estado</span><span class="cl-tag">Terrestre</span></div>
    </div>
    """, unsafe_allow_html=True)

    seg_filter_geo = st.selectbox("Ver por segmento:", ["Todo el mercado"] + list(SEG_KEYWORDS.keys()), key="seg_geo")
    df_geo = df.copy() if seg_filter_geo == "Todo el mercado" else df[df['TIPO_MERCANCIA'].isin(SEG_KEYWORDS[seg_filter_geo])].copy()

    geo = df_geo.groupby('ENTIDAD_NORM').agg(
        monto=('MONTO_SINIESTRO','sum'), count=('NUM_SINIESTROS','sum'), gastos=('GASTOS_AJUSTE','sum'),
    ).reset_index()
    geo = geo[~geo['ENTIDAD_NORM'].str.contains('EXTRA|No disp|nan', na=False, case=False)]
    geo['monto_m']  = (geo['monto'] / 1e6).round(1)
    geo['gastos_m'] = (geo['gastos'] / 1e6).round(1)
    geo = geo.sort_values('monto_m', ascending=False).reset_index(drop=True)
    geo['rank'] = geo.index + 1
    geo['pct']  = (geo['monto'] / geo['monto'].sum() * 100).round(1)

    col_m, col_t = st.columns([2, 1])

    with col_m:
        top15 = geo.head(15).sort_values('monto_m')
        accent_color = SEG_COLORS.get(seg_filter_geo, CL["teal"])
        bar_colors = [
            CL["red"] if i >= len(top15) - 3 else
            CL["orange"] if i >= len(top15) - 6 else accent_color
            for i in range(len(top15))
        ]
        fig_geo = go.Figure(go.Bar(
            x=top15['monto_m'], y=top15['ENTIDAD_NORM'], orientation='h',
            marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0)', width=0)),
            text=top15['monto_m'].apply(lambda v: f"${v:.0f}M"), textposition='outside',
            textfont=dict(size=10, color=CL["text"]),
            hovertemplate="<b>%{y}</b><br>Monto: $%{x:.0f}M<extra></extra>",
        ))
        fig_geo.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Top 15 Entidades · Monto Siniestrado", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(**axis_style("Monto ($M)"), tickprefix="$"),
            yaxis=dict(tickfont=dict(size=10, color=CL["text"]), showgrid=False),
            height=460,
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with col_t:
        top3_pct = geo.head(3)['pct'].sum()
        top5_pct = geo.head(5)['pct'].sum()
        st.markdown(f"""
        <div class="cl-card">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">Concentración geográfica</p>
          <p style="font-family:'Syne';font-size:2.2rem;font-weight:800;color:{CL['red']};margin:0;">{top3_pct:.0f}%</p>
          <p style="font-size:0.83rem;color:{CL['text_dim']};margin-bottom:16px;">del monto está en los 3 estados con mayor riesgo</p>
          <p style="font-family:'Syne';font-size:1.8rem;font-weight:800;color:{CL['orange']};margin:0;">{top5_pct:.0f}%</p>
          <p style="font-size:0.83rem;color:{CL['text_dim']};">en los 5 estados principales</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        rows_html = ""
        medals = ["🥇","🥈","🥉","4°","5°","6°","7°","8°"]
        for _, row in geo.head(8).iterrows():
            medal = medals[int(row['rank'])-1]
            rows_html += f"""<tr>
              <td style="color:{CL['text_dim']};font-size:0.8rem;">{medal}</td>
              <td><b>{row['ENTIDAD_NORM']}</b></td>
              <td style="text-align:right;color:{CL['teal']};">${row['monto_m']:.0f}M</td>
              <td style="text-align:right;color:{CL['text_dim']};">{row['pct']:.1f}%</td>
            </tr>"""
        st.markdown(f"""
        <div class="cl-card" style="padding:16px;">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Ranking por monto</p>
          <table class="cl-table"><thead><tr><th>#</th><th>Estado</th><th style="text-align:right;">Monto</th><th style="text-align:right;">% Total</th></tr></thead>
          <tbody>{rows_html}</tbody></table>
        </div>
        """, unsafe_allow_html=True)

    top3_states = geo.head(3)['ENTIDAD_NORM'].tolist()
    st.markdown(f"""
    <div class="cl-insight">
      <div class="cl-insight-title">📌 Implicación estratégica</div>
      <div class="cl-insight-text">{top3_states[0]} + {top3_states[1]} + {top3_states[2]} concentran el {geo.head(3)['pct'].sum():.0f}% del monto siniestrado.
      Un piloto de CL Circular en estos tres estados cubre la mayor parte del riesgo nacional con penetración mínima.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_sc = go.Figure()
    size_vals = (geo['gastos_m'] / geo['gastos_m'].max() * 40 + 6).tolist()
    fig_sc.add_trace(go.Scatter(
        x=geo['count'], y=geo['monto_m'], mode='markers',
        marker=dict(
            size=size_vals,
            color=geo['monto_m'],
            colorscale=[[0, CL["blue"]], [0.5, CL["teal"]], [1, CL["red"]]],
            showscale=True,
            colorbar=dict(title="Monto ($M)", tickfont=dict(color="#FFFFFF"), titlefont=dict(color="#FFFFFF"), x=1.02),
            line=dict(color=CL["bg"], width=1),
        ),
        text=geo['ENTIDAD_NORM'], customdata=geo['gastos_m'],
        hovertemplate="<b>%{text}</b><br>Siniestros: %{x:.0f}<br>Monto: $%{y:.0f}M<br>Gastos ajuste: $%{customdata:.0f}M<extra></extra>",
    ))
    fig_sc.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Frecuencia vs. Monto por Estado (tamaño = gastos de ajuste)", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
        xaxis=dict(**axis_style("Número de siniestros")),
        yaxis=dict(**axis_style("Monto siniestrado ($M)"), tickprefix="$"),
        height=380,
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — ANÁLISIS DE CAUSAS
# ═══════════════════════════════════════════════════════════════════════
elif seccion == "03 · Análisis de Causas":

    st.markdown(f"""
    <div class="cl-topbar">
      <div>
        <div class="cl-section-title">Análisis de Causas</div>
        <div class="cl-section-sub">¿Por qué se pierde la mercancía? · Transporte terrestre {year_range[0]}–{year_range[1]}</div>
      </div>
      <div><span class="cl-tag">Causas</span><span class="cl-tag">Tendencia</span></div>
    </div>
    """, unsafe_allow_html=True)

    def custom_categorize(row):
        causa = str(row['CAUSA_SINIESTRO'])
        if pd.Series([causa]).str.contains('robo.*con violencia', case=False, na=False, regex=True).iloc[0]:
            return 'Robo con violencia'
        elif pd.Series([causa]).str.contains('robo total con violencia|robo parcial con violencia|robo total sin violencia|robo parcial sin violencia|extravío', case=False, na=False, regex=True).iloc[0]:
            return 'Robo sin violencia'
        elif any(k in causa.lower() for k in ['colisión','volcadura','descarrilamiento','colision','choque','atropellamiento']):
            return 'Colisión / Volcadura'
        elif any(k in causa.lower() for k in ['rotura','rajadura','abolladura','daño','avería','falla mecánica']):
            return 'Daño físico'
        elif any(k in causa.lower() for k in ['maniobra','carga','descarga','estacionamiento','movimiento']):
            return 'Maniobras'
        elif any(k in causa.lower() for k in ['mojadura','humedad','agua','lluvia','inundación']):
            return 'Mojadura'
        elif any(k in causa.lower() for k in ['refrigeración','temperatura','calor','frio','congelación']):
            return 'Falla refrigeración'
        else:
            return 'Otras causas'

    df_for_chart = df.copy()
    df_for_chart['CUSTOM_CATEGORIA'] = df_for_chart.apply(custom_categorize, axis=1)

    cat_totals = df_for_chart.groupby('CUSTOM_CATEGORIA').agg(
        monto=('MONTO_SINIESTRO','sum'), count=('NUM_SINIESTROS','sum'), gastos=('GASTOS_AJUSTE','sum'),
    ).reset_index()
    cat_totals['monto_m']  = cat_totals['monto'] / 1e6
    cat_totals['gastos_m'] = cat_totals['gastos'] / 1e6
    cat_totals['pct'] = (cat_totals['monto'] / cat_totals['monto'].sum() * 100).round(1)
    cat_totals = cat_totals.sort_values('monto_m', ascending=False)

    cat_color_map = {
        'Robo con violencia': CL["red"], 'Colisión / Volcadura': CL["teal"],
        'Otras causas': CL["text_dim"], 'Daño físico': CL["blue"],
        'Maniobras': CL["green"], 'Mojadura': CL["green_dark"],
        'Robo sin violencia': CL["orange"], 'Falla refrigeración': CL["gold"],
    }

    col1, col2 = st.columns([3, 2])

    with col1:
        causa_yr = df_for_chart.groupby(['AÑO','CUSTOM_CATEGORIA'])['MONTO_SINIESTRO'].sum().reset_index()
        causa_yr['monto_m'] = causa_yr['MONTO_SINIESTRO'] / 1e6
        fig_stack = go.Figure()
        for cat in reversed(cat_totals['CUSTOM_CATEGORIA'].tolist()):
            sub = causa_yr[causa_yr['CUSTOM_CATEGORIA'] == cat]
            fig_stack.add_trace(go.Bar(
                x=sub['AÑO'], y=sub['monto_m'], name=cat,
                marker_color=cat_color_map.get(cat, CL["text_dim"]),
                hovertemplate=f"<b>{cat}</b><br>Año: %{{x}}<br>$%{{y:.0f}}M<extra></extra>",
            ))
        max_y_2024 = causa_yr[causa_yr['AÑO'] == 2024]['monto_m'].sum()
        fig_stack.add_annotation(
            x=2024, y=max_y_2024 + 50, text="*Ajuste retroactivo",
            showarrow=True, arrowhead=2, ax=0, ay=-30,
            font=dict(size=9, color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0.7)", bordercolor=CL["border"], borderwidth=1, borderpad=4,
        )
        fig_stack.update_layout(
            **PLOTLY_BASE, barmode='stack',
            title=dict(text="Composición del Monto Siniestrado por Causa", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(**axis_style("Año"), dtick=1),
            yaxis=dict(**axis_style("Monto ($M)"), tickprefix="$"),
            height=360, legend=dict(font=dict(size=9), orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_stack, use_container_width=True)
        st.caption("* 2024 incluye acumulado hasta cierre de año con ajustes retroactivos — ver fuente CNSF/AMIS")

    with col2:
        fig_wf2 = go.Figure(go.Bar(
            x=cat_totals['monto_m'], y=cat_totals['CUSTOM_CATEGORIA'], orientation='h',
            marker_color=[cat_color_map.get(c, CL["text_dim"]) for c in cat_totals['CUSTOM_CATEGORIA']],
            text=cat_totals.apply(lambda r: f"${r['monto_m']:.0f}M  ({r['pct']:.1f}%)", axis=1),
            textposition='outside', textfont=dict(size=9, color=CL["text"]),
            hovertemplate="<b>%{y}</b><br>$%{x:.0f}M<extra></extra>",
        ))
        fig_wf2.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Monto por Causa (total)", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(**axis_style(""), tickprefix="$", showgrid=False),
            yaxis=dict(tickfont=dict(size=10), showgrid=False),
            height=360,
        )
        st.plotly_chart(fig_wf2, use_container_width=True)

    df_kpi = df[(df['AÑO'] >= year_range[0]) & (df['AÑO'] <= year_range[1])]
    total_monto_tab3 = df_kpi['MONTO_SINIESTRO'].sum()
    robo_viol = df_kpi[df_kpi['CAUSA_SINIESTRO'].str.contains('robo.*con violencia', case=False, na=False, regex=True)]['MONTO_SINIESTRO'].sum() / 1e6
    robo_total = df_kpi[df_kpi['CAUSA_SINIESTRO'].str.contains('robo total con violencia|robo parcial con violencia|robo total sin violencia|robo parcial sin violencia|extravío', case=False, na=False, regex=True)]['MONTO_SINIESTRO'].sum() / 1e6
    pct_robo_tab3 = safe_pct(robo_total * 1e6, total_monto_tab3)
    gasto_total = df_kpi['GASTOS_AJUSTE'].sum() / 1e6

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="cl-card-accent">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;">Robo con violencia</p>
          <p style="font-family:'Syne';font-size:1.8rem;font-weight:800;color:{CL['red']};margin:0;">${robo_viol:.0f}M</p>
          <p style="font-size:0.82rem;color:{CL['text_dim']};">causa #1 en monto — verificable con sensor</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="cl-card-accent">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;">Total robos (con + sin violencia)</p>
          <p style="font-family:'Syne';font-size:1.8rem;font-weight:800;color:{CL['orange']};margin:0;">${robo_total:.0f}M</p>
          <p style="font-size:0.82rem;color:{CL['text_dim']};">{pct_robo_tab3:.1f}% del monto total siniestrado</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="cl-card-accent">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;">Gastos de ajuste acumulados</p>
          <p style="font-family:'Syne';font-size:1.8rem;font-weight:800;color:{CL['gold']};margin:0;">${gasto_total:.0f}M</p>
          <p style="font-size:0.82rem;color:{CL['text_dim']};">costo de investigar siniestros sin evidencia</p></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cl-insight" style="margin-top:16px;">
      <div class="cl-insight-title">📌 El problema de la información asimétrica</div>
      <div class="cl-insight-text">Sin un registro objetivo del trayecto, la aseguradora depende de declaraciones unilaterales para resolver siniestros. Los ${gasto_total:.0f}M en gastos de ajuste son el costo directo de operar en ese vacío de información. El sensor de CL Circular convierte cada evento en un registro continuo, verificable y con timestamp.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ratio_yr = df.groupby('AÑO').agg(monto=('MONTO_SINIESTRO','sum'), gastos=('GASTOS_AJUSTE','sum')).reset_index()
    ratio_yr['ratio'] = ratio_yr.apply(lambda r: safe_pct(r['gastos'], r['monto']), axis=1).round(2)

    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=ratio_yr['AÑO'], y=ratio_yr['ratio'], mode='lines+markers',
        line=dict(color=CL["gold"], width=2.5), marker=dict(size=7, color=CL["gold"]),
        fill='tozeroy', fillcolor=hex_to_rgba(CL["gold"], 0.14),
        hovertemplate="<b>%{x}</b><br>Ratio: %{y:.2f}%<extra></extra>",
    ))
    fig_ratio.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Ratio Gastos de Ajuste / Monto Siniestrado — Tendencia del Costo Operativo Aseguradora", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
        xaxis=dict(**axis_style("Año"), dtick=1),
        yaxis=dict(**axis_style("% Gastos / Monto"), ticksuffix="%"),
        height=260,
    )
    st.plotly_chart(fig_ratio, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — INDUSTRIAS OBJETIVO
# ═══════════════════════════════════════════════════════════════════════
elif seccion == "04 · Industrias Objetivo":

    st.markdown(f"""
    <div class="cl-topbar">
      <div>
        <div class="cl-section-title">Industrias Objetivo</div>
        <div class="cl-section-sub">Perfil de siniestralidad por segmento · Cerveza · Harinas · Autopartes · {year_range[0]}–{year_range[1]}</div>
      </div>
      <div><span class="cl-tag">Segmentos</span><span class="cl-tag">ROI del Sensor</span></div>
    </div>
    """, unsafe_allow_html=True)

    seg_tabs = st.tabs(["⚖️ Comparativa", "🍺 Cerveza", "🌾 Harinas", "🔩 Autopartes"])

    with seg_tabs[0]:
        comp_data = []
        for seg_name, kws in SEG_KEYWORDS.items():
            sub = df[df['TIPO_MERCANCIA'].isin(kws)]
            robo = sub[sub['CATEGORIA'].str.contains('Robo', na=False)]
            monto_t = sub['MONTO_SINIESTRO'].sum()
            monto_r = robo['MONTO_SINIESTRO'].sum()
            gastos_t = sub['GASTOS_AJUSTE'].sum()
            comp_data.append({
                'Segmento': seg_name,
                'Monto_M': monto_t / 1e6, 'Robo_M': monto_r / 1e6,
                'Pct_Robo': monto_r / monto_t * 100 if monto_t > 0 else 0,
                'Gastos_M': gastos_t / 1e6,
                'Ratio_Gastos': gastos_t / monto_t * 100 if monto_t > 0 else 0,
            })
        comp_df = pd.DataFrame(comp_data)

        col1, col2 = st.columns(2)
        with col1:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Monto total",
                x=comp_df['Segmento'], y=comp_df['Monto_M'],
                marker_color=[SEG_COLORS[s] for s in comp_df['Segmento']],
                marker_opacity=0.45,
                hovertemplate="<b>%{x}</b><br>Monto total: $%{y:.0f}M<extra></extra>",
            ))
            fig_comp.add_trace(go.Bar(
                name="Monto por robo",
                x=comp_df['Segmento'], y=comp_df['Robo_M'],
                marker_color=[SEG_COLORS[s] for s in comp_df['Segmento']],
                marker_opacity=1.0,
                customdata=comp_df['Pct_Robo'],
                hovertemplate="<b>%{x}</b><br>Monto robo: $%{y:.0f}M (%{customdata:.1f}% del total)<extra></extra>",
            ))
            fig_comp.update_layout(
                **PLOTLY_BASE, barmode='group',
                title=dict(text="Monto Total vs. Monto por Robo", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
                xaxis=dict(tickfont=dict(size=10, color="#FFFFFF")),
                yaxis=dict(**axis_style("Monto ($M)"), tickprefix="$", type='log'),
                height=300,
                legend=dict(**LEGEND_BASE, orientation='h', y=-0.28),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown(f'<p style="font-size:0.8rem;color:{CL["text_dim"]};margin-top:-10px;">Escala logarítmica · barra translúcida = monto total · barra sólida = monto por robo</p>', unsafe_allow_html=True)

        with col2:
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Bar(
                x=comp_df['Segmento'], y=comp_df['Pct_Robo'],
                marker=dict(color=[SEG_COLORS[s] for s in comp_df['Segmento']], line=dict(color=CL["bg"], width=2)),
                text=comp_df['Pct_Robo'].apply(lambda v: f"{v:.1f}%"),
                textposition='outside', textfont=dict(size=13, color=CL["text"], family="Syne"),
                hovertemplate="<b>%{x}</b><br>% robo: %{y:.1f}%<extra></extra>",
            ))
            fig_pct.update_layout(
                **PLOTLY_BASE,
                title=dict(text="% del Monto Causado por Robo", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
                xaxis=dict(tickfont=dict(size=10)),
                yaxis=dict(**axis_style("% del monto"), ticksuffix="%", range=[0, 100]),
                height=300,
                shapes=[dict(type='line', x0=-0.5, x1=2.5, y0=50, y1=50,
                             line=dict(color=CL["text_dim"], dash="dot", width=1))],
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        rows = ""
        for _, r in comp_df.iterrows():
            label_robo = f'<span class="cl-badge-red">{r["Pct_Robo"]:.1f}%</span>' if r['Pct_Robo'] > 50 else f'{r["Pct_Robo"]:.1f}%'
            rows += f"""<tr>
              <td><b>{r['Segmento']}</b></td>
              <td style="text-align:right;color:{CL['teal']};">${r['Monto_M']:.0f}M</td>
              <td style="text-align:right;color:{CL['red']};">${r['Robo_M']:.0f}M</td>
              <td style="text-align:right;">{label_robo}</td>
              <td style="text-align:right;color:{CL['gold']};">${r['Gastos_M']:.1f}M</td>
              <td style="text-align:right;color:{CL['text_dim']};">{r['Ratio_Gastos']:.1f}%</td>
            </tr>"""
        st.markdown(f"""
        <div class="cl-card" style="margin-top:8px;">
          <table class="cl-table"><thead><tr>
            <th>Segmento</th><th style="text-align:right;">Monto Total</th><th style="text-align:right;">Monto Robo</th>
            <th style="text-align:right;">% Robo</th><th style="text-align:right;">Gastos Ajuste</th><th style="text-align:right;">% Gastos/Monto</th>
          </tr></thead><tbody>{rows}</tbody></table>
        </div>""", unsafe_allow_html=True)

    def render_segment_tab(seg_name, color):
        kws = SEG_KEYWORDS[seg_name]
        sub = df[df['TIPO_MERCANCIA'].isin(kws)].copy()
        if sub.empty:
            st.warning("Sin datos para este segmento en el periodo seleccionado.")
            return None
        monto_total  = sub['MONTO_SINIESTRO'].sum()
        monto_robo   = sub[sub['CATEGORIA'].str.contains('Robo', na=False)]['MONTO_SINIESTRO'].sum()
        pct_robo     = safe_pct(monto_robo, monto_total)
        gastos_ajuste = sub['GASTOS_AJUSTE'].sum()
        num_siniestros = int(sub['NUM_SINIESTROS'].sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Monto Total Siniestrado", f"${monto_total/1e6:.0f}M")
        with c2: st.metric("Monto por Robo", f"${monto_robo/1e6:.0f}M", f"{pct_robo:.1f}% del total")
        with c3: st.metric("Gastos de Ajuste", f"${gastos_ajuste/1e6:.1f}M", f"{safe_pct(gastos_ajuste, monto_total):.1f}% del monto")
        with c4: st.metric("Siniestros Registrados", f"{num_siniestros:,.0f}")

        col_l, col_r = st.columns(2)
        with col_l:
            yr_seg = sub.groupby('AÑO').agg(
                monto=('MONTO_SINIESTRO','sum'),
                robo=('MONTO_SINIESTRO', lambda x: x[sub.loc[x.index,'CATEGORIA'].str.contains('Robo',na=False)].sum()),
            ).reset_index()
            yr_seg['monto_m'] = yr_seg['monto'] / 1e6
            yr_seg['robo_m']  = yr_seg['robo'] / 1e6
            fig_yr = go.Figure()
            fig_yr.add_trace(go.Bar(x=yr_seg['AÑO'], y=yr_seg['monto_m'], name="Monto total", marker_color=color, marker_opacity=0.4))
            fig_yr.add_trace(go.Bar(x=yr_seg['AÑO'], y=yr_seg['robo_m'], name="Del cual: robo", marker_color=color, marker_opacity=1.0))
            fig_yr.update_layout(**PLOTLY_BASE, barmode='overlay',
                title=dict(text="Tendencia Anual — Total vs. Robo", font=dict(size=12, family="Syne", color="#FFFFFF"), x=0),
                xaxis=dict(**axis_style(), dtick=1), yaxis=dict(**axis_style("$M"), tickprefix="$"), height=300)
            st.plotly_chart(fig_yr, use_container_width=True)

        with col_r:
            cat_seg = sub.groupby('CATEGORIA')['MONTO_SINIESTRO'].sum().reset_index()
            cat_seg['monto_m'] = cat_seg['MONTO_SINIESTRO'] / 1e6
            cat_seg = cat_seg.sort_values('monto_m')
            cat_color_map2 = {
                'Robo con violencia': CL["red"], 'Robo sin violencia': CL["orange"],
                'Colisión / Volcadura': CL["teal"], 'Daño físico': CL["blue"],
                'Maniobras': CL["green"], 'Mojadura': CL["green_dark"],
                'Falla refrigeración': CL["gold"], 'Otras causas': CL["text_dim"],
            }
            fig_cat = go.Figure(go.Bar(
                x=cat_seg['monto_m'], y=cat_seg['CATEGORIA'], orientation='h',
                marker_color=[cat_color_map2.get(c, color) for c in cat_seg['CATEGORIA']],
                text=cat_seg['monto_m'].apply(lambda v: f"${v:.0f}M"), textposition='outside',
                textfont=dict(size=9, color=CL["text"]),
            ))
            fig_cat.update_layout(**PLOTLY_BASE,
                title=dict(text="Desglose por Causa", font=dict(size=12, family="Syne", color="#FFFFFF"), x=0),
                xaxis=dict(**axis_style(), tickprefix="$", showgrid=False),
                yaxis=dict(tickfont=dict(size=9), showgrid=False), height=300)
            st.plotly_chart(fig_cat, use_container_width=True)

        geo_seg = sub.groupby('ENTIDAD_NORM').agg(monto=('MONTO_SINIESTRO','sum')).reset_index()
        geo_seg = geo_seg[~geo_seg['ENTIDAD_NORM'].str.contains('EXTRA|No disp|nan', na=False, case=False)]
        geo_seg['monto_m'] = geo_seg['monto'] / 1e6
        geo_seg = geo_seg.sort_values('monto_m', ascending=False).head(6)
        fig_geo2 = go.Figure(go.Bar(
            x=geo_seg['ENTIDAD_NORM'], y=geo_seg['monto_m'], marker_color=color,
            marker_opacity=[1.0, 0.85, 0.70, 0.60, 0.50, 0.40][:len(geo_seg)],
            text=geo_seg['monto_m'].apply(lambda v: f"${v:.0f}M"), textposition='outside',
            textfont=dict(size=10, color=CL["text"]),
        ))
        fig_geo2.update_layout(**PLOTLY_BASE,
            title=dict(text="Top 6 Estados con Mayor Siniestralidad", font=dict(size=12, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(tickfont=dict(size=10)), yaxis=dict(**axis_style("$M"), tickprefix="$"), height=260)
        st.plotly_chart(fig_geo2, use_container_width=True)

        return {'monto_total': monto_total, 'monto_robo': monto_robo,
                'pct_robo': pct_robo, 'gastos_ajuste': gastos_ajuste, 'num_siniestros': num_siniestros}

    with seg_tabs[1]:
        d = render_segment_tab("🍺  Cerveza", CL["orange"])
        if d:
            st.markdown(f"""<div class="cl-insight"><div class="cl-insight-title">📌 Pitch para aseguradora — Cerveza</div>
              <div class="cl-insight-text">El robo con violencia domina ({d['pct_robo']:.1f}% del monto). El sensor documenta apertura de puertas, desvío de ruta y timestamps en tiempo real — convierte un evento disputado en evidencia técnica irrefutable.</div></div>""", unsafe_allow_html=True)
    with seg_tabs[2]:
        d = render_segment_tab("🌾  Harinas", CL["green"])
        if d:
            st.markdown(f"""<div class="cl-insight"><div class="cl-insight-title">📌 Pitch para aseguradora — Harinas</div>
              <div class="cl-insight-text">El {d['pct_robo']:.1f}% del monto es robo. Adicionalmente, la contaminación por contacto genera pérdidas verificables con monitoreo de condiciones de carga.</div></div>""", unsafe_allow_html=True)
    with seg_tabs[3]:
        d = render_segment_tab("🔩  Autopartes", CL["teal"])
        if d:
            st.markdown(f"""<div class="cl-insight"><div class="cl-insight-title">📌 Pitch para aseguradora — Autopartes</div>
              <div class="cl-insight-text">El segmento más grande (${d['monto_total']/1e6:.0f}M). El pitch aquí es doble: el robo (${d['monto_robo']/1e6:.0f}M) se resuelve con GPS; el daño físico y maniobras se resuelven con registro de impactos y vibración.</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECCIÓN 5 — CASO DEL SENSOR
# ═══════════════════════════════════════════════════════════════════════
elif seccion == "05 · Caso del Sensor":

    st.markdown(f"""
    <div class="cl-topbar">
      <div>
        <div class="cl-section-title">El Caso del Sensor CL Circular</div>
        <div class="cl-section-sub">¿Por qué la aseguradora debería ofrecer el chip a sus clientes? — Modelo de ROI interactivo</div>
      </div>
      <div><span class="cl-tag">Business Case</span><span class="cl-tag">ROI</span></div>
    </div>
    """, unsafe_allow_html=True)

    rows_comp = [
        ("Evidencia del siniestro", "Declaraciones y peritaje post-facto (días o semanas)", "Logs continuos: GPS, apertura puertas, temperatura, impactos — disponibles en minutos", CL["red"], CL["green"]),
        ("Costo de ajuste / peritaje", "Tendencia creciente: alcanzó 6.5% del monto en 2023", "Resolución más rápida → reducción estimada 30–50%", CL["orange"], CL["green"]),
        ("Disputas", "Sin datos objetivos → aseguradora absorbe incertidumbre", "Evento verificable → se resuelve con datos, no con negociación", CL["red"], CL["green"]),
        ("Cadena de frío / condiciones", "Alerta post-daño (el producto ya se perdió)", "Alerta en tiempo real → acción correctiva antes del siniestro", CL["orange"], CL["teal"]),
        ("Score de riesgo", "Estático — basado en historial", "Dinámico — por comportamiento real → pricing más preciso", CL["text_dim"], CL["teal"]),
    ]
    table_rows = ""
    for dim, sin, con, c1, c2 in rows_comp:
        table_rows += f"""<tr>
          <td style="font-weight:600;color:{CL['text_dim']};font-size:0.82rem;">{dim}</td>
          <td style="color:#ff7070;">{sin}</td>
          <td style="color:{CL['green_lt']};">{con}</td>
        </tr>"""
    st.markdown(f"""
    <div class="cl-card">
      <table class="cl-table"><thead><tr>
        <th>Dimensión</th>
        <th style="color:{CL['red']};">❌  Sin Sensor (hoy)</th>
        <th style="color:{CL['green_lt']};">✅  Con Sensor CL Circular</th>
      </tr></thead><tbody>{table_rows}</tbody></table>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="cl-section-title" style="font-size:1.1rem;margin-bottom:4px;">Calculadora de ROI — Modelo Interactivo</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cl-section-sub">Ajusta los parámetros para estimar el impacto económico del sensor en una cartera de pólizas</div>', unsafe_allow_html=True)

    col_s, col_r = st.columns([1, 2])

    with col_s:
        seg_calc         = st.selectbox("Segmento objetivo", list(SEG_KEYWORDS.keys()), key="seg_calc")
        n_polizas        = st.slider("Número de pólizas con sensor", 100, 10000, 1000, step=100)
        prima_prom       = st.slider("Prima promedio por póliza ($MXN)", 50_000, 2_000_000, 300_000, step=50_000, format="$%d")
        loss_ratio_actual = st.slider("Loss ratio actual (%)", 40, 90, 68)
        reduccion_lr     = st.slider("Reducción estimada de loss ratio con sensor (pp)", 1, 20, 5)
        reduccion_gastos = st.slider("Reducción en gastos de ajuste (%)", 10, 60, 30)
        costo_sensor     = st.slider("Costo anual del sensor por unidad ($MXN)\n*(~$25 USD/viaje · 52 viajes/año · $17.50 MXN/USD)*", 1_000, 55_000, 22_750, step=250)

    with col_r:
        prima_total       = n_polizas * prima_prom
        siniestros_sin    = prima_total * (loss_ratio_actual / 100)
        siniestros_con    = prima_total * ((loss_ratio_actual - reduccion_lr) / 100)
        ahorro_siniestros = siniestros_sin - siniestros_con

        kws_calc = SEG_KEYWORDS[seg_calc]
        sub_calc = df[df['TIPO_MERCANCIA'].isin(kws_calc)]
        ratio_gastos_seg = (sub_calc['GASTOS_AJUSTE'].sum() / sub_calc['MONTO_SINIESTRO'].sum()) if sub_calc['MONTO_SINIESTRO'].sum() > 0 else 0.05
        gastos_sin     = siniestros_sin * ratio_gastos_seg
        gastos_con     = gastos_sin * (1 - reduccion_gastos / 100)
        ahorro_gastos  = gastos_sin - gastos_con

        costo_total_sensores = n_polizas * costo_sensor
        otros_beneficios     = n_polizas * 50_000
        ahorro_total         = ahorro_siniestros + ahorro_gastos + otros_beneficios
        roi_neto             = ahorro_total - costo_total_sensores
        roi_pct              = (roi_neto / costo_total_sensores * 100) if costo_total_sensores > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Prima total cartera", f"${prima_total/1e6:.0f}M", help=f"{n_polizas:,} pólizas × ${prima_prom/1e3:.0f}K")
        with k2: st.metric("Ahorro en siniestros", f"${ahorro_siniestros/1e6:.1f}M", f"-{reduccion_lr}pp loss ratio")
        with k3: st.metric("Ahorro en peritajes", f"${ahorro_gastos/1e6:.1f}M", f"-{reduccion_gastos}% gastos ajuste")
        with k4: st.metric("ROI neto del sensor", f"${roi_neto/1e6:.1f}M", f"{roi_pct:.0f}% sobre costo sensores")

        st.markdown("<br>", unsafe_allow_html=True)

        _wfall_vals = [-costo_total_sensores/1e6, ahorro_siniestros/1e6, ahorro_gastos/1e6, otros_beneficios/1e6, roi_neto/1e6]
        fig_wfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Costo sensores", "Ahorro siniestros", "Ahorro peritajes", "Otros beneficios", "ROI Neto"],
            y=_wfall_vals,
            connector=dict(line=dict(color=CL["border"], width=1)),
            decreasing=dict(marker=dict(color=CL["red"])),
            increasing=dict(marker=dict(color=CL["green"])),
            totals=dict(marker=dict(color=CL["teal"])),
            text=[f"${v:+.1f}M" for v in _wfall_vals],
            textposition="outside", textfont=dict(color=CL["text"], size=11),
            hovertext=[f"${v:+.1f}M" for v in _wfall_vals],
            hovertemplate="<b>%{x}</b><br>%{hovertext}<extra></extra>",
        ))
        fig_wfall.update_layout(
            **PLOTLY_BASE,
            title=dict(text=f"Modelo de Ahorro — {n_polizas:,} pólizas · {seg_calc}", font=dict(size=12, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(tickfont=dict(size=10)), yaxis=dict(**axis_style("$M"), tickprefix="$"), height=320,
        )
        st.plotly_chart(fig_wfall, use_container_width=True)
        st.markdown(
            f'<p style="font-size:0.72rem;color:{CL["text_dim"]};font-style:italic;">'
            f'* Otros beneficios fijo: ${otros_beneficios/1e6:.1f}M ({n_polizas:,} pólizas × $50K) — retención, reducción de fraude y pricing dinámico. '
            f'Ratio gastos de ajuste del segmento: {ratio_gastos_seg*100:.1f}% (dato real CNSF/AMIS).</p>',
            unsafe_allow_html=True,
        )

        with st.expander("🔍 Ver supuestos del modelo — Datos reales de CNSF/AMIS"):
            base_data = []
            for seg_name, kws in SEG_KEYWORDS.items():
                sub2 = df[df['TIPO_MERCANCIA'].isin(kws)]
                m = sub2['MONTO_SINIESTRO'].sum()
                g = sub2['GASTOS_AJUSTE'].sum()
                r = sub2[sub2['CATEGORIA'].str.contains('Robo',na=False)]['MONTO_SINIESTRO'].sum()
                base_data.append({
                    'Segmento': seg_name,
                    'Monto siniestrado ($M)': round(m/1e6, 1),
                    '% Robo': round(r/m*100, 1) if m > 0 else 0,
                    'Gastos ajuste ($M)': round(g/1e6, 1),
                    'Ratio gastos/monto': f"{g/m*100:.1f}%" if m > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(base_data), use_container_width=True, hide_index=True)
            st.caption("Fuente: CNSF/AMIS · Transporte terrestre · Periodo filtrado por slider")

    st.markdown(f"""
    <div class="cl-insight" style="margin-top:8px;">
      <div class="cl-insight-title">📌 Conclusión estratégica</div>
      <div class="cl-insight-text">El sensor no compite con el seguro: lo hace más rentable y defendible. La aseguradora gana en tres frentes simultáneos: (1) menor loss ratio; (2) menor costo operativo por resolución más rápida; (3) mejor pricing por score de riesgo dinámico.</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SECCIÓN 6 — PROYECCIÓN ML
# ═══════════════════════════════════════════════════════════════════════
elif seccion == "06 · Proyección ML":

    st.markdown(f"""
    <div class="cl-topbar">
      <div>
        <div class="cl-section-title">Proyección de Siniestralidad 2025–2027</div>
        <div class="cl-section-sub">
          Modelo ensamble: Holt-Winters + Ridge Polinómica · Simulación de escenarios con sensor
        </div>
      </div>
      <div>
        <span class="cl-tag">ML</span>
        <span class="cl-tag">Proyección</span>
        <span class="cl-tag">Contrafactual</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Entrenar modelo (usa toda la serie 2015-2024, ignorar filtro de año) ──
    anual_full = terr.groupby('AÑO').agg(monto=('MONTO_SINIESTRO', 'sum')).reset_index()
    anual_full['monto_M'] = anual_full['monto'] / 1e6
    monto_tuple = tuple(anual_full['monto_M'].tolist())
    modelo = entrenar_modelo(monto_tuple)

    proj_base  = modelo['proj_base']
    proj_years = modelo['proj_years']

    # ── KPIs del modelo ──
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("MAPE Walk-Forward", f"{modelo['mape_wf']:.1f}%",
                  help="Error porcentual absoluto medio · validación 2019–2023")
    with col_b:
        st.metric("R² en Test 2022–2024", f"{modelo['r2_test']:.3f}",
                  help="Coeficiente de determinación en periodo de prueba")
    with col_c:
        st.metric("Proyección 2025", f"${proj_base[0]:,.0f}M",
                  help="Monto siniestrado proyectado sin sensor")
    with col_d:
        st.metric("α Holt-Winters", f"{modelo['alpha_hw']:.2f}",
                  help=f"Parámetro de nivel. β={modelo['beta_hw']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Controles de escenario ──
    st.markdown(
        f'<div class="cl-section-title" style="font-size:1.0rem;margin-bottom:6px;">'
        f'Parámetros del Escenario con Sensor</div>', unsafe_allow_html=True
    )
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        reduccion_pct = st.slider("Reducción de siniestralidad con sensor (%)", 1, 20, 5,
                                  help="Reducción anual del monto proyectado. Literatura UBI reporta 3–15pp.")
        pen_2025 = st.slider("Penetración del mercado 2025 (%)", 5, 60, 30)
    with col_c2:
        pen_2026 = st.slider("Penetración del mercado 2026 (%)", 10, 80, 50)
        pen_2027 = st.slider("Penetración del mercado 2027 (%)", 20, 100, 70)

    reduccion   = reduccion_pct / 100
    penetracion = [pen_2025/100, pen_2026/100, pen_2027/100]

    # Escenarios fijos
    ESCENARIOS_FIJOS = {
        'Conservador (3%)':  (0.03, [0.20, 0.35, 0.50]),
        'Base (5%)':         (0.05, [0.30, 0.50, 0.70]),
        'Optimista (10%)':   (0.10, [0.40, 0.60, 0.80]),
    }
    COLORS_ESC = {
        'Conservador (3%)':  CL['gold'],
        'Base (5%)':         CL['green'],
        'Optimista (10%)':   CL['teal'],
    }

    # Escenario personalizado
    proj_custom, ahorro_custom = calcular_escenario(proj_base, reduccion, penetracion)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══ GRÁFICA 1: Histórico + Proyección ═══════════════════════════
    col_g1, col_g2 = st.columns([3, 2])

    with col_g1:
        st.markdown(
            f'<div class="cl-section-title" style="font-size:1.0rem;margin-bottom:6px;">'
            f'Proyección 2025–2027 — Histórico vs. Escenarios</div>', unsafe_allow_html=True
        )

        fig_proj = go.Figure()

        # Banda de confianza 80%
        fig_proj.add_trace(go.Scatter(
            x=proj_years + proj_years[::-1],
            y=modelo['ci_upper'] + modelo['ci_lower'][::-1],
            fill='toself',
            fillcolor=hex_to_rgba(CL['text_dim'], 0.12),
            line=dict(color='rgba(0,0,0,0)'),
            name='IC 80% sin sensor', showlegend=True, hoverinfo='skip',
        ))

        # Histórico
        fig_proj.add_trace(go.Scatter(
            x=modelo['years'], y=modelo['y_all'],
            mode='lines+markers', name='Histórico (real)',
            line=dict(color=CL['teal'], width=2.5), marker=dict(size=7, color=CL['teal']),
            hovertemplate='<b>%{x}</b> (real)<br>$%{y:.0f}M<extra></extra>',
        ))

        # Proyección base
        last_x = modelo['years'][-1]
        last_y = modelo['y_all'][-1]
        fig_proj.add_trace(go.Scatter(
            x=[last_x] + proj_years, y=[last_y] + proj_base,
            mode='lines+markers', name='Sin sensor (base)',
            line=dict(color=CL['text_dim'], width=2, dash='dot'),
            marker=dict(size=7, color=CL['text_dim'], symbol='diamond'),
            hovertemplate='<b>%{x}</b> (proyección)<br>$%{y:.0f}M<extra></extra>',
        ))

        # Escenarios fijos
        for nombre, (red, pen) in ESCENARIOS_FIJOS.items():
            proj_e, _ = calcular_escenario(proj_base, red, pen)
            fig_proj.add_trace(go.Scatter(
                x=[last_x] + proj_years, y=[last_y] + proj_e,
                mode='lines+markers', name=f'Con sensor — {nombre}',
                line=dict(color=COLORS_ESC[nombre], width=2),
                marker=dict(size=6, color=COLORS_ESC[nombre]),
                hovertemplate=f'<b>%{{x}}</b> — {nombre}<br>$%{{y:.0f}}M<extra></extra>',
            ))

        # Tu escenario
        fig_proj.add_trace(go.Scatter(
            x=[last_x] + proj_years, y=[last_y] + proj_custom,
            mode='lines+markers', name=f'Tu escenario ({reduccion_pct}%)',
            line=dict(color=CL['orange'], width=3, dash='dashdot'),
            marker=dict(size=8, color=CL['orange'], symbol='star'),
            hovertemplate=f'Tu escenario ({reduccion_pct}%)<br><b>%{{x}}</b><br>$%{{y:.0f}}M<extra></extra>',
        ))

        fig_proj.add_vline(
            x=2024.5, line=dict(color=CL['border'], width=1.5, dash='dash'),
            annotation_text='→ proyección',
            annotation_font=dict(size=9, color="#FFFFFF"),
        )
        fig_proj.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Monto Siniestrado — Histórico y Proyección con/sin Sensor", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(**axis_style("Año"), dtick=1),
            yaxis=dict(**axis_style("Monto ($M)"), tickprefix="$"),
            height=420, hovermode='x unified',
            legend=dict(**LEGEND_BASE, orientation='v', x=1.02, y=1),
        )
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_g2:
        # Tabla de ahorro
        st.markdown(
            f'<div class="cl-section-title" style="font-size:1.0rem;margin-bottom:6px;">'
            f'Ahorro Proyectado Acumulado</div>', unsafe_allow_html=True
        )
        rows_html = ""
        for nombre, (red, pen) in ESCENARIOS_FIJOS.items():
            _, ahorro_e = calcular_escenario(proj_base, red, pen)
            total_ahorro = sum(ahorro_e)
            pct_ahorro   = total_ahorro / sum(proj_base) * 100
            color = COLORS_ESC[nombre]
            rows_html += f"""<tr>
              <td><b style="color:{color};">{nombre}</b></td>
              <td style="text-align:right;color:{CL['green_lt']};">${total_ahorro:,.0f}M</td>
              <td style="text-align:right;color:{CL['text_dim']};">{pct_ahorro:.1f}%</td>
            </tr>"""

        total_custom = sum(ahorro_custom)
        pct_custom   = total_custom / sum(proj_base) * 100
        rows_html += f"""<tr style="background:{CL['surface2']};">
          <td><b style="color:{CL['orange']};">Tu escenario ({reduccion_pct}%)</b></td>
          <td style="text-align:right;color:{CL['orange']};">${total_custom:,.0f}M</td>
          <td style="text-align:right;color:{CL['orange']};">{pct_custom:.1f}%</td>
        </tr>"""

        st.markdown(f"""
        <div class="cl-card">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Ahorro 2025–2027 (acumulado)</p>
          <table class="cl-table"><thead><tr>
            <th>Escenario</th><th style="text-align:right;">Ahorro total</th><th style="text-align:right;">% del base</th>
          </tr></thead><tbody>{rows_html}</tbody></table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.8rem;color:{CL["text_dim"]};text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:8px;">Tu escenario — detalle anual</div>',
            unsafe_allow_html=True
        )
        for i, yr in enumerate(proj_years):
            st.markdown(f"""
            <div class="cl-card-accent" style="padding:10px 14px;margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.85rem;color:{CL['text_dim']};">{yr}</span>
                <span style="font-family:'Syne';font-size:1.1rem;font-weight:700;color:{CL['orange']};">${ahorro_custom[i]:,.0f}M</span>
              </div>
              <div style="font-size:0.75rem;color:{CL['text_dim']};">
                Base: ${proj_base[i]:,.0f}M → Con sensor: ${proj_custom[i]:,.0f}M (penetración: {penetracion[i]*100:.0f}%)
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ═══ GRÁFICA 2: Validación del modelo ═══════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="cl-section-title" style="font-size:1.0rem;margin-bottom:6px;">'
        f'Validación del Modelo — Real vs. Ajustado</div>', unsafe_allow_html=True
    )

    col_v1, col_v2 = st.columns([2, 1])

    with col_v1:
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=modelo['years'], y=modelo['y_all'],
            mode='lines+markers', name='Real (CNSF/AMIS)',
            line=dict(color=CL['teal'], width=2.5), marker=dict(size=8, color=CL['teal']),
        ))
        fig_val.add_trace(go.Scatter(
            x=modelo['test_years'], y=modelo['ensemble_test'],
            mode='lines+markers', name='Modelo (test 2022-2024)',
            line=dict(color=CL['orange'], width=2, dash='dot'),
            marker=dict(size=8, color=CL['orange'], symbol='square'),
            hovertemplate='<b>%{x}</b> (pred)<br>$%{y:.0f}M<extra></extra>',
        ))
        fig_val.add_vrect(
            x0=2021.5, x1=2024.5,
            fillcolor=hex_to_rgba(CL['orange'], 0.06),
            line=dict(color=CL['orange'], width=1, dash='dot'),
            annotation_text='Periodo de prueba',
            annotation_position='top left',
            annotation_font=dict(size=9, color="#FFFFFF"),
        )
        fig_val.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Real vs. Predicción del Modelo (2022–2024)", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
            xaxis=dict(**axis_style("Año"), dtick=1),
            yaxis=dict(**axis_style("Monto ($M)"), tickprefix="$"),
            height=300, hovermode='x unified',
        )
        st.plotly_chart(fig_val, use_container_width=True)

    with col_v2:
        st.markdown(f"""
        <div class="cl-card">
          <p style="font-size:0.7rem;color:{CL['text_dim']};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;">Métricas de Validación</p>

          <p style="font-size:0.72rem;color:{CL['text_dim']};margin-bottom:2px;">MAPE — Walk-forward (2019–2023)</p>
          <p style="font-family:'Syne';font-size:1.6rem;font-weight:700;color:{CL['gold']};margin:0 0 12px 0;">{modelo['mape_wf']:.1f}%</p>

          <p style="font-size:0.72rem;color:{CL['text_dim']};margin-bottom:2px;">MAE — Test 2022–2024</p>
          <p style="font-family:'Syne';font-size:1.6rem;font-weight:700;color:{CL['teal']};margin:0 0 12px 0;">${modelo['mae_test']:,.0f}M</p>

          <p style="font-size:0.72rem;color:{CL['text_dim']};margin-bottom:2px;">R² — Test 2022–2024</p>
          <p style="font-family:'Syne';font-size:1.6rem;font-weight:700;color:{CL['green']};margin:0 0 12px 0;">{modelo['r2_test']:.3f}</p>

          <p style="font-size:0.72rem;color:{CL['text_dim']};margin-bottom:2px;">Parámetros HW óptimos</p>
          <p style="font-size:0.85rem;color:{CL['text']};margin:0;">
            α (nivel) = {modelo['alpha_hw']:.2f}<br>
            β (tendencia) = {modelo['beta_hw']:.2f}
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ═══ GRÁFICA 3: Barras de ahorro ═══════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="cl-section-title" style="font-size:1.0rem;margin-bottom:6px;">'
        f'Comparativa de Ahorro por Escenario y Año</div>', unsafe_allow_html=True
    )

    fig_bar = go.Figure()
    for nombre, (red, pen) in ESCENARIOS_FIJOS.items():
        _, ahorro_e = calcular_escenario(proj_base, red, pen)
        fig_bar.add_trace(go.Bar(
            name=nombre, x=proj_years, y=ahorro_e,
            marker_color=COLORS_ESC[nombre],
            hovertemplate=f'<b>{nombre}</b><br>%{{x}}<br>${{y:.0f}}M<extra></extra>',
        ))
    fig_bar.add_trace(go.Bar(
        name=f'Tu escenario ({reduccion_pct}%)', x=proj_years, y=ahorro_custom,
        marker_color=CL['orange'], marker_pattern_shape='/',
        hovertemplate=f'Tu escenario<br>%{{x}}<br>${{y:.0f}}M<extra></extra>',
    ))
    fig_bar.update_layout(
        **PLOTLY_BASE, barmode='group',
        title=dict(text="Ahorro Proyectado por Escenario — Con Sensor vs. Sin Sensor ($M MXN)", font=dict(size=13, family="Syne", color="#FFFFFF"), x=0),
        xaxis=dict(**axis_style("Año"), dtick=1),
        yaxis=dict(**axis_style("Ahorro ($M)"), tickprefix="$"),
        height=320, legend=dict(**LEGEND_BASE, orientation='h', y=-0.2),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ═══ METODOLOGÍA ═══════════════════════════════════════════════
    with st.expander("📐 Metodología del modelo — Detalle técnico"):
        st.markdown(f"""
        **Modelo de Proyección: Ensamble Holt-Winters Doble + Ridge Polinómica**

        ---

        **Holt-Winters Doble (sin estacionalidad)**

        Captura la tendencia local mediante dos ecuaciones de suavizamiento exponencial:
        - Nivel: $L_t = \\alpha \\cdot y_t + (1-\\alpha)(L_{{t-1}} + T_{{t-1}})$
        - Tendencia: $T_t = \\beta(L_t - L_{{t-1}}) + (1-\\beta)T_{{t-1}}$
        - Pronóstico: $\\hat{{y}}_{{t+h}} = L_t + h \\cdot T_t$

        Parámetros óptimos: α={modelo['alpha_hw']:.2f} y β={modelo['beta_hw']:.2f} (búsqueda en grilla sobre [0.05, 0.95]).

        ---

        **Regresión Ridge Polinómica (grado 2, λ=1.0)**

        Modela la trayectoria cuadrática: $\\hat{{y}}_t = \\beta_0 + \\beta_1 t + \\beta_2 t^2$

        La regularización Ridge (L2) previene sobreajuste con solo 10 observaciones:
        $\\min ||y - X\\beta||^2 + \\lambda||\\beta||^2$

        ---

        **Ensamble y Bandas de Confianza**

        La proyección final es el promedio de ambos modelos. Las bandas de confianza al 80% se calculan
        por bootstrap (n=5,000 muestras) sobre los errores del walk-forward validation (2019–2023).

        ---

        **Simulación de Escenarios con Sensor (contrafactual)**

        $\\hat{{y}}_{{sensor,t}} = \\hat{{y}}_{{base,t}} \\times (1 - r \\cdot p_t)$

        Donde $r$ es la reducción anual y $p_t$ la penetración del mercado en el año $t$.

        **Limitación:** Las reducciones son supuestos de simulación, no evidencia empírica propia.
        Se requiere un piloto de 12–18 meses para validar.

        ---
        *Fuente: CNSF/AMIS — Bases de Mercancías 2015–2024 · Filtro: transporte terrestre*
        """)

    # ═══ INSIGHT FINAL ═════════════════════════════════════════════
    _, ah_con = calcular_escenario(proj_base, 0.03, [0.20, 0.35, 0.50])
    _, ah_opt = calcular_escenario(proj_base, 0.10, [0.40, 0.60, 0.80])

    st.markdown(f"""
    <div class="cl-insight" style="margin-top:8px;">
      <div class="cl-insight-title">📌 Interpretación para el socio formador</div>
      <div class="cl-insight-text">
        El modelo proyecta un monto siniestrado de <b>${proj_base[0]:,.0f}M – ${proj_base[2]:,.0f}M</b> para 2025–2027
        en ausencia de intervención (CAGR histórico: 6.9% anual). Con el sensor y penetración gradual del mercado,
        el ahorro acumulado proyectado oscila entre
        <b>${sum(ah_con):,.0f}M</b> (escenario conservador, 3%) y
        <b>${sum(ah_opt):,.0f}M</b> (escenario optimista, 10%).
        Estos ahorros son el argumento económico central para el convenio con aseguradoras:
        el sensor no compite con la póliza — la hace más rentable.
      </div>
    </div>
    """, unsafe_allow_html=True)
