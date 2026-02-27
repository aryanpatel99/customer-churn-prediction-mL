"""
app.py  –  Premium Executive Churn Intelligence Dashboard
Connects to the trained ML pipeline and presents predictions
in a luxury dark-theme layout with glassmorphism, sparklines, and animations.
"""

import sys, os, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -- path setup --
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.model_pipeline import predict_churn

# ============================================
#  Page config
# ============================================
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="\U0001F4C9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
#  Premium CSS  – dark theme + system fonts
# ============================================
DARK_CSS = """
<style>
:root {
  --bg-base:     #0B0F19;
  --bg-surface:  #111827;
  --bg-card:     #1A1F2E;
  --bg-card-alt: #1E2538;
  --border:      rgba(255,255,255,0.06);
  --border-hover:rgba(255,255,255,0.12);
  --glow:        rgba(224,86,102,0.08);
  --text-1:      #F1F5F9;
  --text-2:      #94A3B8;
  --text-3:      #64748B;
  --coral:       #E05666;
  --coral-soft:  #F2707F;
  --green:       #34D399;
  --green-dim:   #059669;
  --amber:       #F59E0B;
  --blue:        #60A5FA;
  --purple:      #A78BFA;
  --rose:        #FB7185;
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
  color: var(--text-1) !important;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: linear-gradient(180deg, #0B0F19 0%, #0F1420 40%, #111827 100%) !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D1120 0%, #111827 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
}

.block-container {
  padding: 1.4rem 2.2rem 1.2rem 2.2rem !important;
  max-width: 1440px !important;
}

div[data-testid="stMetric"] { display: none !important; }

.stDownloadButton > button {
  background: var(--bg-card) !important;
  color: var(--coral-soft) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
  transition: all 0.25s ease;
}
.stDownloadButton > button:hover {
  border-color: var(--coral) !important;
  box-shadow: 0 0 20px rgba(224,86,102,0.15) !important;
  transform: translateY(-1px);
}

[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
}

.section-divider {
  border: none; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06) 20%, rgba(255,255,255,0.06) 80%, transparent);
  margin: 0.8rem 0 1.2rem 0;
}

@keyframes fadeUp {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn {
  from { opacity:0; }
  to   { opacity:1; }
}

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 24px 18px 24px;
  min-height: 140px;
  display: flex; flex-direction: column; justify-content: space-between;
  animation: fadeUp 0.5s ease-out both;
  transition: all 0.3s ease;
  position: relative; overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-clr, #E05666), transparent);
  opacity: 0; transition: opacity 0.3s ease;
}
.kpi-card:hover {
  border-color: var(--border-hover);
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(0,0,0,0.3), 0 0 40px var(--glow);
}
.kpi-card:hover::before { opacity: 1; }
.kpi-card .kpi-icon {
  position: absolute; top: 18px; right: 20px;
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.05rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
}
.kpi-card .kpi-label {
  font-size: 0.76rem; font-weight: 500;
  color: var(--text-2); text-transform: uppercase;
  letter-spacing: 0.8px; margin-bottom: 8px;
}
.kpi-card .kpi-value {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  font-size: 2rem; font-weight: 700; line-height: 1.1; margin-bottom: 2px;
}
.kpi-card .kpi-sub {
  font-size: 0.72rem; color: var(--text-3); margin-top: 6px;
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 6px;
  background: rgba(100,116,139,0.12);
}
.kpi-card .sparkline-area { margin-top: 10px; height: 32px; overflow: hidden; }

/* -- Native st.container(border=True) card styling -- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 20px 18px 10px 18px !important;
  overflow: hidden;
  animation: fadeIn 0.6s ease-out both;
  transition: border-color 0.3s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--border-hover) !important;
}
.chart-title {
  font-size: 0.92rem; font-weight: 600; color: var(--text-1);
  margin-bottom: 4px; padding-left: 2px;
}
.chart-subtitle {
  font-size: 0.72rem; color: var(--text-3);
  margin-bottom: 6px; padding-left: 2px;
}

.section-header {
  font-size: 1.0rem; font-weight: 600; color: var(--text-1);
  margin: 0.5rem 0 0.6rem 0;
  display: flex; align-items: center; gap: 10px;
  letter-spacing: 0.2px;
}
.section-header .dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 8px currentColor;
}

.rank-row {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.rank-row:last-child { border-bottom: none; }
.rank-num { color: var(--text-3); font-size: 0.78rem; font-weight: 500; min-width: 16px; text-align: right; }
.rank-label { color: var(--text-2); font-size: 0.82rem; font-weight: 500; min-width: 140px; }
.rank-value { color: var(--text-1); font-size: 0.82rem; font-weight: 600; font-family: -apple-system, system-ui, sans-serif; min-width: 65px; text-align: right; }
.rank-bar-bg { flex: 1; height: 8px; background: rgba(255,255,255,0.04); border-radius: 4px; overflow: hidden; }
.rank-bar { height: 100%; border-radius: 4px; transition: width 0.8s ease-out; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ============================================
#  Helper - run predictions & enrich
# ============================================
@st.cache_data(show_spinner="Running churn model ...")
def run_predictions(df_raw: pd.DataFrame) -> pd.DataFrame:
    result_df, _ = predict_churn(df_raw)
    result_df["Probability_Num"] = (
        result_df["Probability"].str.replace("%", "", regex=False).astype(float)
    )
    result_df["Risk Bucket"] = pd.cut(
        result_df["Probability_Num"], bins=[0, 30, 60, 100],
        labels=["Low", "Medium", "High"], include_lowest=True,
    )
    result_df["Tenure Bucket"] = pd.cut(
        result_df["Tenure in Months"], bins=[-1, 6, 24, 200],
        labels=["New (0-6)", "Growing (6-24)", "Loyal (24+)"],
    )
    result_df["Referral Bucket"] = pd.cut(
        result_df["Number of Referrals"], bins=[-1, 0, 3, 100],
        labels=["0", "1-3", "4+"],
    )
    return result_df


# ============================================
#  Synthetic data generator
# ============================================
def generate_random_customers(n: int) -> pd.DataFrame:
    rng = np.random.default_rng()
    return pd.DataFrame({
        "Customer ID": [f"SYN-{i:05d}" for i in range(1, n + 1)],
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 80, n),
        "Married": rng.choice(["Yes", "No"], n),
        "Number of Dependents": rng.integers(0, 5, n),
        "Number of Referrals": rng.integers(0, 10, n),
        "Tenure in Months": rng.integers(1, 72, n),
        "Offer": rng.choice(["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"], n),
        "Contract": rng.choice(["Month-to-Month", "One Year", "Two Year"], n),
        "Paperless Billing": rng.choice(["Yes", "No"], n),
        "Payment Method": rng.choice(["Credit Card", "Bank Withdrawal", "Mailed Check"], n),
        "Phone Service": rng.choice(["Yes", "No"], n),
        "Multiple Lines": rng.choice(["Yes", "No"], n),
        "Internet Service": rng.choice(["Yes", "No"], n),
        "Internet Type": rng.choice(["Cable", "DSL", "Fiber Optic"], n),
        "Online Security": rng.choice(["Yes", "No"], n),
        "Online Backup": rng.choice(["Yes", "No"], n),
        "Device Protection Plan": rng.choice(["Yes", "No"], n),
        "Premium Tech Support": rng.choice(["Yes", "No"], n),
        "Streaming TV": rng.choice(["Yes", "No"], n),
        "Streaming Movies": rng.choice(["Yes", "No"], n),
        "Streaming Music": rng.choice(["Yes", "No"], n),
        "Unlimited Data": rng.choice(["Yes", "No"], n),
        "Avg Monthly Long Distance Charges": np.round(rng.uniform(0, 50, n), 2),
        "Avg Monthly GB Download": np.round(rng.uniform(0, 500, n), 1),
        "Monthly Charge": np.round(rng.uniform(20, 120, n), 2),
        "Total Charges": np.round(rng.uniform(100, 8000, n), 2),
        "Total Refunds": np.round(rng.uniform(0, 50, n), 2),
        "Total Extra Data Charges": np.round(rng.uniform(0, 100, n), 2),
        "Total Long Distance Charges": np.round(rng.uniform(0, 200, n), 2),
        "Total Revenue": np.round(rng.uniform(100, 10000, n), 2),
    })


# ============================================
#  Plotly dark chart defaults (premium palette)
# ============================================
CARD_BG    = "#1A1F2E"
GRID_CLR   = "rgba(255,255,255,0.05)"
TEXT_CLR   = "#94A3B8"
TITLE_CLR  = "#F1F5F9"

LAYOUT_DARK = dict(
    plot_bgcolor=CARD_BG,
    paper_bgcolor=CARD_BG,
    margin=dict(l=55, r=25, t=50, b=55),
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, system-ui, sans-serif", size=12, color=TEXT_CLR),
    hoverlabel=dict(bgcolor="#1E2538", font_size=12, font_color=TITLE_CLR,
                    bordercolor="rgba(255,255,255,0.1)"),
    xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR,
               tickfont=dict(color=TEXT_CLR, size=11)),
    yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, zerolinecolor=GRID_CLR,
               tickfont=dict(color=TEXT_CLR, size=11)),
    bargap=0.30,
    showlegend=False,
)

ACCENT_RED    = "#E05666"
ACCENT_GREEN  = "#059669"
ACCENT_BLUE   = "#60A5FA"
ACCENT_ORANGE = "#F59E0B"
ACCENT_PURPLE = "#A78BFA"
ACCENT_CORAL  = "#F2707F"
ACCENT_ROSE   = "#FB7185"
GREEN_BRIGHT  = "#34D399"


# ============================================
#  Sparkline SVG builder
# ============================================
def sparkline_svg(data, color, w=200, h=32):
    if not data or len(data) < 2:
        return ""
    mn, mx = min(data), max(data)
    rng = mx - mn if mx != mn else 1
    pts = []
    for i, v in enumerate(data):
        x = i / (len(data) - 1) * w
        y = h - ((v - mn) / rng) * (h - 4) - 2
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    fill_pts = f"0,{h} " + poly + f" {w},{h}"
    return (
        f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<polygon points="{fill_pts}" fill="{color}" opacity="0.12"/>'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ============================================
#  KPI card HTML builder (with sparkline)
# ============================================
def kpi_card(label, value, color, icon="", sub="", spark_data=None, delay=0):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    spark_html = ""
    if spark_data:
        spark_html = f'<div class="sparkline-area">{sparkline_svg(spark_data, color)}</div>'
    return textwrap.dedent(f"""
    <div class="kpi-card" style="--accent-clr:{color};animation-delay:{delay}s">
        <div class="kpi-icon">{icon}</div>
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            {sub_html}
        </div>
        {spark_html}
    </div>
    """)


def chart_title_html(title, subtitle=""):
    sub = f'<div class="chart-subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="chart-title">{title}</div>{sub}'


def section_header(title, color):
    return f'<div class="section-header"><span class="dot" style="background:{color};color:{color}"></span>{title}</div>'


def fmt_money(v):
    if v >= 1_000_000: return f"${v/1_000_000:.1f}M"
    if v >= 1_000: return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def ranked_list_html(items, bar_color, max_val=None):
    """Render horizontal bar ranked list from [(label, value), ...] pairs."""
    if max_val is None:
        max_val = max(v for _, v in items) if items else 1
    rows = []
    for i, (label, val) in enumerate(items, 1):
        pct = min(val / max_val * 100, 100) if max_val else 0
        rows.append(
            f'<div class="rank-row">'
            f'<span class="rank-num">{i}</span>'
            f'<span class="rank-label">{label}</span>'
            f'<span class="rank-value">{val:.1f}%</span>'
            f'<div class="rank-bar-bg"><div class="rank-bar" '
            f'style="width:{pct:.1f}%;background:linear-gradient(90deg,{bar_color},{bar_color}dd)"></div></div>'
            f'</div>'
        )
    return "\n".join(rows)


# ============================================
#  Sidebar - Data Source
# ============================================
st.sidebar.markdown("#### Data Source")
data_mode = st.sidebar.radio(
    "Choose input",
    ["Default Dataset", "Upload CSV", "Generate Synthetic"],
    index=0, label_visibility="collapsed",
)

df_input = None

if data_mode == "Default Dataset":
    default_path = os.path.join(PROJECT_ROOT, "data", "raw", "telecom_customer_churn.csv")
    if os.path.exists(default_path):
        df_input = pd.read_csv(default_path)
    else:
        st.sidebar.error("Default dataset not found.")

elif data_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)

else:
    n_syn = st.sidebar.number_input("Customers to generate", 10, 10000, 200, step=50)
    if st.sidebar.button("Generate", use_container_width=True):
        df_input = generate_random_customers(int(n_syn))
        st.session_state["syn_data"] = df_input
    elif "syn_data" in st.session_state:
        df_input = st.session_state["syn_data"]

if df_input is None:
    st.markdown(
        '<div style="text-align:center;padding:120px 0;">'
        '<h2 style="color:#94A3B8;font-weight:500;font-family:-apple-system,system-ui,sans-serif">'
        'Customer Churn Intelligence</h2>'
        '<p style="color:#64748B;font-size:0.9rem">Select a data source in the sidebar to begin.</p>'
        '</div>', unsafe_allow_html=True)
    st.stop()

try:
    df = run_predictions(df_input)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()


# ============================================
#  Sidebar - Filters
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("#### Filters")

filter_contract = st.sidebar.multiselect(
    "Contract",
    options=sorted(df["Contract"].dropna().unique()),
    default=sorted(df["Contract"].dropna().unique()),
)
filter_tenure = st.sidebar.multiselect(
    "Tenure Bucket",
    options=["New (0-6)", "Growing (6-24)", "Loyal (24+)"],
    default=["New (0-6)", "Growing (6-24)", "Loyal (24+)"],
)
filter_referral = st.sidebar.multiselect(
    "Referral Bucket",
    options=["0", "1-3", "4+"],
    default=["0", "1-3", "4+"],
)
filter_risk = st.sidebar.multiselect(
    "Risk Bucket",
    options=["Low", "Medium", "High"],
    default=["Low", "Medium", "High"],
)
charge_min = float(df["Monthly Charge"].min())
charge_max = float(df["Monthly Charge"].max())
filter_charge = st.sidebar.slider(
    "Monthly Charge Range",
    charge_min, charge_max,
    (charge_min, charge_max), step=1.0,
)

mask = (
    df["Contract"].isin(filter_contract)
    & df["Tenure Bucket"].isin(filter_tenure)
    & df["Referral Bucket"].isin(filter_referral)
    & df["Risk Bucket"].isin(filter_risk)
    & df["Monthly Charge"].between(filter_charge[0], filter_charge[1])
)
dff = df[mask].copy()


# ============================================
#  HEADER
# ============================================
header_left, header_right = st.columns([3.5, 1.5])
with header_left:
    st.markdown(
        '<p style="color:#64748B;font-size:0.78rem;margin:0 0 2px 0;letter-spacing:1px;'
        'text-transform:uppercase;font-weight:500">Churn Analytics</p>'
        '<h1 style="margin:0;font-size:1.75rem;font-weight:700;color:#F1F5F9;'
        'font-family:-apple-system,system-ui,sans-serif;letter-spacing:-0.3px">'
        'Executive Summary</h1>'
        '<p style="color:#64748B;font-size:0.82rem;margin-top:4px">'
        'Predicted churn analysis &bull; Model threshold 30%</p>',
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f'<div style="text-align:right;padding-top:22px">'
        f'<span style="color:#94A3B8;font-size:0.78rem">'
        f'Showing&nbsp;&nbsp;<strong style="color:#F1F5F9">{len(dff):,}</strong>'
        f'&nbsp;&nbsp;of&nbsp;&nbsp;<strong style="color:#F1F5F9">{len(df):,}</strong>'
        f'&nbsp;&nbsp;customers</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ============================================
#  ROW 1 - KPI Cards (with sparklines)
# ============================================
total = len(dff)
churned_df = dff[dff["Churned"] == "Yes"]
n_churn = len(churned_df)
churn_rate = (n_churn / total * 100) if total else 0
rev_at_risk = churned_df["Monthly Charge"].sum()
avg_tenure_churn = churned_df["Tenure in Months"].mean() if n_churn else 0

# Build sparkline data
prob_hist, _ = np.histogram(dff["Probability_Num"].dropna(), bins=12)
spark_prob = prob_hist.tolist()
ten_hist, _ = np.histogram(churned_df["Tenure in Months"].dropna(), bins=10)
spark_ten = ten_hist.tolist()
rev_hist, _ = np.histogram(churned_df["Monthly Charge"].dropna(), bins=10)
spark_rev = rev_hist.tolist()
contract_sp = [
    len(dff[(dff["Contract"] == c) & (dff["Churned"] == "Yes")])
    for c in ["Month-to-Month", "One Year", "Two Year"]
]
spark_contract = (contract_sp * 4)[:12]

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
with k1:
    st.markdown(kpi_card("Total Customers", f"{total:,}", ACCENT_BLUE, "\U0001F465",
                         f"{total:,} in filter", spark_prob, 0.0), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Predicted Churners", f"{n_churn:,}", ACCENT_ROSE, "\u26A0\uFE0F",
                         f"{churn_rate:.1f}% of total", spark_contract, 0.08), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Churn Rate", f"{churn_rate:.1f}%", ACCENT_ORANGE, "\U0001F4CA",
                         "threshold 30%", spark_prob[::-1], 0.12), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Revenue at Risk", fmt_money(rev_at_risk), ACCENT_CORAL, "\U0001F4B0",
                         "monthly recurring", spark_rev, 0.18), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_card("Avg Tenure (Churners)", f"{avg_tenure_churn:.1f} mo", ACCENT_PURPLE, "\u23F1\uFE0F",
                         "months before churn", spark_ten, 0.24), unsafe_allow_html=True)


# ============================================
#  ROW 2 - Risk Overview
# ============================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(section_header("Risk Overview", ACCENT_RED), unsafe_allow_html=True)
r2a, r2b = st.columns(2, gap="medium")

with r2a:
    with st.container(border=True):
        st.markdown(chart_title_html("Churn Distribution", "Retained vs Churned customers"), unsafe_allow_html=True)
        churn_counts = dff["Churned"].value_counts().reindex(["No", "Yes"]).reset_index()
        churn_counts.columns = ["Churned", "Count"]
        churn_counts["Label"] = churn_counts["Churned"].map({"No": "Retained", "Yes": "Churned"})
        fig1 = go.Figure(go.Bar(
            x=churn_counts["Label"], y=churn_counts["Count"],
            marker=dict(color=[ACCENT_GREEN, ACCENT_RED], line=dict(width=0), cornerradius=6),
            text=[f"{v:,}" for v in churn_counts["Count"]], textposition="outside",
            textfont=dict(color=TITLE_CLR, size=14, family="-apple-system, system-ui, sans-serif"),
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        ))
        fig1.update_layout(**LAYOUT_DARK, height=370)
        max_y1 = churn_counts["Count"].max()
        fig1.update_yaxes(title_text="Customers", title_font=dict(color=TEXT_CLR, size=11),
                          range=[0, max_y1 * 1.18])
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with r2b:
    with st.container(border=True):
        st.markdown(chart_title_html("Probability Distribution", "Churn probability spread"), unsafe_allow_html=True)
        fig2 = go.Figure(go.Histogram(
            x=dff["Probability_Num"], nbinsx=25,
            marker=dict(color=ACCENT_BLUE, line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                        cornerradius=3),
            hovertemplate="Prob: %{x:.0f}%%<br>Count: %{y}<extra></extra>",
        ))
        fig2.update_layout(**LAYOUT_DARK, height=370)
        fig2.update_xaxes(title_text="Churn Probability (%)", title_font=dict(color=TEXT_CLR, size=11))
        fig2.update_yaxes(title_text="Customers", title_font=dict(color=TEXT_CLR, size=11))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ============================================
#  ROW 3 - Primary Drivers
# ============================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(section_header("Primary Churn Drivers", ACCENT_CORAL), unsafe_allow_html=True)
r3a, r3b = st.columns(2, gap="medium")

with r3a:
    with st.container(border=True):
        st.markdown(chart_title_html("Churn Rate by Contract", "Month-to-Month = highest risk"), unsafe_allow_html=True)
        contract_order = ["Month-to-Month", "One Year", "Two Year"]
        cr_contract = (
            dff.groupby("Contract")["Churned"]
            .apply(lambda s: round((s == "Yes").mean() * 100, 1))
            .reindex(contract_order).reset_index(name="Churn Rate %")
        )
        bar_colors = [ACCENT_RED if v > 30 else ACCENT_ORANGE if v > 15 else ACCENT_GREEN
                      for v in cr_contract["Churn Rate %"]]
        fig3 = go.Figure(go.Bar(
            x=cr_contract["Contract"], y=cr_contract["Churn Rate %"],
            marker=dict(color=bar_colors, cornerradius=6),
            text=[f"{v:.1f}%" for v in cr_contract["Churn Rate %"]],
            textposition="outside",
            textfont=dict(color=TITLE_CLR, size=13, family="-apple-system, system-ui, sans-serif"),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig3.update_layout(**LAYOUT_DARK, height=370)
        max_cr = max(cr_contract["Churn Rate %"]) if max(cr_contract["Churn Rate %"]) > 0 else 100
        fig3.update_yaxes(title_text="Churn Rate %", title_font=dict(color=TEXT_CLR, size=11),
                          range=[0, max_cr * 1.3])
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with r3b:
    with st.container(border=True):
        st.markdown(chart_title_html("Churn Rate by Tenure", "New customers churn significantly more"), unsafe_allow_html=True)
        tenure_order = ["New (0-6)", "Growing (6-24)", "Loyal (24+)"]
        cr_tenure = (
            dff.groupby("Tenure Bucket", observed=False)["Churned"]
            .apply(lambda s: round((s == "Yes").mean() * 100, 1))
            .reindex(tenure_order).reset_index(name="Churn Rate %")
        )
        fig4 = go.Figure(go.Bar(
            x=cr_tenure["Tenure Bucket"], y=cr_tenure["Churn Rate %"],
            marker=dict(color=[ACCENT_RED, ACCENT_ORANGE, ACCENT_GREEN], cornerradius=6),
            text=[f"{v:.1f}%" for v in cr_tenure["Churn Rate %"]],
            textposition="outside",
            textfont=dict(color=TITLE_CLR, size=13, family="-apple-system, system-ui, sans-serif"),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig4.update_layout(**LAYOUT_DARK, height=370)
        max_ten = max(cr_tenure["Churn Rate %"]) if max(cr_tenure["Churn Rate %"]) > 0 else 100
        fig4.update_yaxes(title_text="Churn Rate %", title_font=dict(color=TEXT_CLR, size=11),
                          range=[0, max_ten * 1.3])
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})


# ============================================
#  ROW 4 - Engagement & Revenue
# ============================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(section_header("Engagement & Revenue", ACCENT_ORANGE), unsafe_allow_html=True)
r4a, r4b = st.columns(2, gap="medium")

with r4a:
    with st.container(border=True):
        st.markdown(chart_title_html("Churn Rate by Referrals", "More referrals = lower churn"), unsafe_allow_html=True)
        ref_order = ["0", "1-3", "4+"]
        cr_ref = (
            dff.groupby("Referral Bucket", observed=False)["Churned"]
            .apply(lambda s: round((s == "Yes").mean() * 100, 1))
            .reindex(ref_order)
        )
        ref_items = [(k, v) for k, v in cr_ref.items()]
        st.markdown(ranked_list_html(ref_items, ACCENT_RED), unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        # Payment Method ranked list
        st.markdown(
            '<div style="margin-top:8px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.04)">'
            '<div style="font-size:0.82rem;font-weight:600;color:#94A3B8;margin-bottom:8px">'
            'Churn Rate by Payment Method</div>', unsafe_allow_html=True)
        cr_pay = (
            dff.groupby("Payment Method")["Churned"]
            .apply(lambda s: round((s == "Yes").mean() * 100, 1))
            .sort_values(ascending=False)
        )
        pay_items = [(k, v) for k, v in cr_pay.items()]
        st.markdown(ranked_list_html(pay_items, ACCENT_ORANGE), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with r4b:
    with st.container(border=True):
        st.markdown(chart_title_html("Monthly Charge: Churned vs Retained", "Pricing pressure on churning customers"), unsafe_allow_html=True)
        retained = dff[dff["Churned"] == "No"]["Monthly Charge"]
        churned  = dff[dff["Churned"] == "Yes"]["Monthly Charge"]
        fig6 = go.Figure()
        fig6.add_trace(go.Box(
            y=retained, name="Retained",
            marker_color=GREEN_BRIGHT, boxmean=True,
            line=dict(width=1.5, color=GREEN_BRIGHT),
            fillcolor="rgba(52,211,153,0.15)",
            hovertemplate="Retained<br>$%{y:.0f}<extra></extra>",
        ))
        fig6.add_trace(go.Box(
            y=churned, name="Churned",
            marker_color=ACCENT_RED, boxmean=True,
            line=dict(width=1.5, color=ACCENT_RED),
            fillcolor="rgba(224,86,102,0.15)",
            hovertemplate="Churned<br>$%{y:.0f}<extra></extra>",
        ))
        fig6.update_layout(**LAYOUT_DARK, height=400)
        fig6.update_yaxes(title_text="Monthly Charge ($)", title_font=dict(color=TEXT_CLR, size=11))
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})


# ============================================
#  ROW 5 - High Risk Customers
# ============================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(section_header("High Risk Customers", ACCENT_PURPLE), unsafe_allow_html=True)

action_cols = [
    "Customer ID", "Probability", "Risk Bucket",
    "Tenure in Months", "Number of Referrals",
    "Monthly Charge", "Contract",
]
high_risk = (
    dff[dff["Risk Bucket"].isin(["High", "Medium"])]
    .sort_values("Probability_Num", ascending=False)[action_cols]
    .reset_index(drop=True)
)

with st.container(border=True):
    if high_risk.empty:
        st.markdown(
            '<p style="color:#64748B;text-align:center;padding:30px 0">'
            'No medium/high risk customers in current filter selection.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(
            high_risk,
            use_container_width=True,
            height=380,
            column_config={
                "Customer ID": st.column_config.TextColumn("Customer ID", width="medium"),
                "Probability": st.column_config.TextColumn("Churn Prob.", width="small"),
                "Risk Bucket": st.column_config.TextColumn("Risk", width="small"),
                "Tenure in Months": st.column_config.NumberColumn("Tenure (mo)", width="small"),
                "Number of Referrals": st.column_config.NumberColumn("Referrals", width="small"),
                "Monthly Charge": st.column_config.NumberColumn("Monthly ($)", format="$%.2f", width="small"),
                "Contract": st.column_config.TextColumn("Contract", width="medium"),
            },
        )


# ============================================
#  Download buttons
# ============================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

dl_cols = [
    "Customer ID", "Gender", "Age", "Married",
    "Number of Dependents", "Number of Referrals",
    "Tenure in Months", "Offer", "Phone Service",
    "Internet Service", "Internet Type", "Contract",
    "Paperless Billing", "Payment Method",
    "Monthly Charge", "Total Charges", "Total Revenue",
    "Probability", "Churned",
]
dl_cols_present = [c for c in dl_cols if c in dff.columns]
csv_full = dff[dl_cols_present].to_csv(index=False).encode("utf-8")

d1, d2, _ = st.columns([1.6, 1.6, 4.8], gap="medium")
with d1:
    st.download_button(
        "\u2B07  Download Full Results",
        data=csv_full,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    if not high_risk.empty:
        csv_hr = high_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            "\u2B07  Download High Risk Only",
            data=csv_hr,
            file_name="high_risk_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )
