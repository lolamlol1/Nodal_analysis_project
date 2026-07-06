"""
app.py  —  Nodal Analysis Web Application
==========================================
PIPESIM-grade Nodal Analysis for production wells.

Run locally:
    streamlit run app.py

Features:
  • 5 IPR models: Linear PI, Vogel, Fetkovitch, Standing, Jones
  • 2 VLP correlations: Beggs & Brill, Hagedorn & Brown
  • PVT: Standing, Beggs-Robinson, Hall-Yarborough, Lee-Gonzalez-Eakin
  • Operating point solver (IPR ∩ VLP intersection)
  • Pressure loss breakdown: gravity, friction, acceleration
  • Packer & completion string design
  • Multi-tubing size VLP comparison
  • Sensitivity analysis (WHP, Water Cut, GOR)
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import io
from datetime import datetime

# ── Local package ──────────────────────────────────────────────────────────── #
from nodal.fluid_properties import FluidProperties
from nodal.ipr import IPRCalculator
from nodal.vlp import VLPCalculator, STANDARD_TUBING_SIZES
from nodal.completion_design import CompletionDesigner
from nodal.nodal_solver import NodalAnalysis

# ══════════════════════════════════════════════════════════════════════════════ #
#  PAGE CONFIG & THEME                                                          #
# ══════════════════════════════════════════════════════════════════════════════ #
st.set_page_config(
    page_title="Nodal Analysis | Petroleum Engineering",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────── #
st.markdown("""
<style>
  /* Import Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 40%, #0a0e1a 100%);
    color: #e2e8f0;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
  }
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #63b3ed;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 1.2rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
  }

  /* Header banner */
  .header-banner {
    background: linear-gradient(90deg,
      rgba(14,165,233,0.15) 0%,
      rgba(56,189,248,0.08) 50%,
      rgba(14,165,233,0.15) 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center,
      rgba(56,189,248,0.05) 0%, transparent 70%);
    animation: shimmer 4s ease-in-out infinite;
  }
  @keyframes shimmer {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.05); }
  }
  .header-title {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradshift 3s linear infinite;
  }
  @keyframes gradshift {
    0% { background-position: 0% }
    100% { background-position: 200% }
  }
  .header-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.3rem;
    font-weight: 400;
  }

  /* KPI metric cards */
  .kpi-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
  .kpi-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg,
      rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.9) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
    backdrop-filter: blur(10px);
  }
  .kpi-card:hover {
    border-color: rgba(56,189,248,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(56,189,248,0.15);
  }
  .kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.1;
  }
  .kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
    font-weight: 500;
  }
  .kpi-unit {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 400;
  }

  /* Success/Warning/Error badges */
  .badge-valid {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    color: #4ade80;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-invalid {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #f87171;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-warning {
    display: inline-block;
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.4);
    color: #fbbf24;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.8);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,179,237,0.1);
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 1rem;
    transition: all 0.2s ease;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56,189,248,0.2), rgba(129,140,248,0.2)) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
  }

  /* Section headers */
  .section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
  }

  /* Number inputs */
  .stNumberInput > div > div > input {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,179,237,0.2);
    color: #e2e8f0;
    border-radius: 8px;
  }
  .stSelectbox > div > div {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 8px;
  }

  /* Run button */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    transition: all 0.25s ease;
    cursor: pointer;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(14,165,233,0.4);
  }

  /* Tables */
  .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
  }

  /* Info boxes */
  .info-box {
    background: linear-gradient(135deg,
      rgba(14,165,233,0.08) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
  }

  /* Divider */
  hr { border-color: rgba(99,179,237,0.1); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════ #
#  PLOTLY THEME                                                                 #
# ══════════════════════════════════════════════════════════════════════════════ #
PLOT_BG    = "#0a0e1a"
PLOT_PAPER = "#0d1528"
GRID_COLOR = "rgba(99,179,237,0.08)"
FONT_COLOR = "#94a3b8"

COLORS = {
    "ipr_primary":  "#38bdf8",   # sky blue
    "vogel":        "#38bdf8",
    "linear_pi":    "#818cf8",   # indigo
    "fetkovitch":   "#34d399",   # emerald
    "standing":     "#fb923c",   # orange
    "jones":        "#f472b6",   # pink
    "vlp_primary":  "#fbbf24",   # amber
    "vlp_hagedorn": "#a78bfa",   # violet
    "operating":    "#f87171",   # red
    "gravity":      "#38bdf8",
    "friction":     "#f97316",
    "accel":        "#a78bfa",
}

TUBING_COLORS = ["#38bdf8", "#34d399", "#fb923c", "#f472b6", "#818cf8"]


def make_base_layout(title: str, xaxis_title: str, yaxis_title: str,
                     height: int = 520, y_range: list = None) -> dict:
    yaxis_cfg = dict(
        title=dict(text=yaxis_title, font=dict(color="#94a3b8")),
        gridcolor=GRID_COLOR,
        zerolinecolor="rgba(99,179,237,0.15)",
        tickfont=dict(color=FONT_COLOR),
        showgrid=True,
    )
    if y_range is not None:
        yaxis_cfg["range"] = y_range
        yaxis_cfg["autorange"] = False
    return dict(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0", family="Inter"), x=0.03),
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=12),
        height=height,
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color="#94a3b8")),
            gridcolor=GRID_COLOR,
            zerolinecolor="rgba(99,179,237,0.15)",
            tickfont=dict(color=FONT_COLOR),
            showgrid=True,
        ),
        yaxis=yaxis_cfg,
        legend=dict(
            bgcolor="rgba(10,14,26,0.8)",
            bordercolor="rgba(99,179,237,0.2)",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )


# ══════════════════════════════════════════════════════════════════════════════ #
#  ANIMATED FIGURE BUILDERS                                                     #
# ══════════════════════════════════════════════════════════════════════════════ #
_ANIM_OPTS = dict(
    frame=dict(duration=18, redraw=True),
    transition=dict(duration=0),
    mode="immediate",
)


def render_animated(fig, height=580):
    """
    Renders a Plotly animated figure that auto-plays only when the user
    opens the Streamlit tab containing it (not simultaneously on page load).
    Uses window.frameElement.offsetWidth to detect tab visibility.
    """
    fig.update_layout(autosize=True)
    html_str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        div_id="anim-plot",
        config={"responsive": True, "displayModeBar": True,
                "modeBarButtonsToRemove": ["toImage"],
                "displaylogo": False},
    )
    autoplay_js = (
        "<style>"
        "body{margin:0;padding:0;background:#0d1528;overflow:hidden;}"
        "#anim-plot{width:100%!important;}"
        "</style>"
        "<script>"
        "var _played=false;"
        "function _triggerAnim(){"
        "  if(_played)return;"
        "  var gd=document.getElementById('anim-plot');"
        "  if(!gd||!gd._fullLayout){return setTimeout(_triggerAnim,80);}"
        "  _played=true;"
        "  Plotly.animate('anim-plot',null,{"
        "    frame:{duration:18,redraw:true},"
        "    transition:{duration:0},"
        "    mode:'immediate'"
        "  });"
        "}"
        "function _waitVisible(){"
        "  var fe=window.frameElement;"
        "  if(fe&&fe.offsetWidth===0){return setTimeout(_waitVisible,120);}"
        "  _triggerAnim();"
        "}"
        "_waitVisible();"
        "</script>"
    )
    html_str = html_str.replace("</body>", autoplay_js + "</body>")
    components.html(html_str, height=height + 55, scrolling=False)


def _anim_frames_two_curves(
    x1, y1, x2, y2,
    name1, color1, dash1,
    name2, color2, dash2,
    op_x=None, op_y=None,
    n_steps=40,
):
    """
    Build Plotly animation frames that draw curve-1 first, then curve-2,
    then pop the operating point.  Returns (initial_traces, frames).
    """
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)

    def _trace1(n):
        return go.Scatter(
            x=x1[:n], y=y1[:n],
            mode="lines",
            name=name1,
            line=dict(color=color1, width=3, dash=dash1),
            hovertemplate="%{x:.0f} STB/d | %{y:.0f} psia<extra>" + name1 + "</extra>",
        )

    def _trace2(n):
        return go.Scatter(
            x=x2[:n], y=y2[:n],
            mode="lines",
            name=name2,
            line=dict(color=color2, width=3, dash=dash2),
            hovertemplate="%{x:.0f} STB/d | %{y:.0f} psia<extra>" + name2 + "</extra>",
        )

    def _op_trace(visible):
        return go.Scatter(
            x=[op_x] if visible and op_x else [],
            y=[op_y] if visible and op_y else [],
            mode="markers",
            name="Operating Point",
            marker=dict(symbol="star", size=20, color=COLORS["operating"],
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>OPERATING POINT</b><br>"
                f"Rate: {op_x:,.0f} STB/d<br>"
                f"FBHP: {op_y:,.0f} psia<extra></extra>"
            ) if op_x else "",
        )

    # Index breakpoints for each step count
    def _idx(total_pts, step, total_steps):
        return max(1, round(total_pts * step / total_steps))

    frames = []
    # Phase 1 – draw curve 1 (IPR)
    for k in range(1, n_steps + 1):
        frames.append(go.Frame(
            data=[_trace1(_idx(len(x1), k, n_steps)), _trace2(0), _op_trace(False)],
            name=f"p1_{k}",
        ))
    # Phase 2 – draw curve 2 (VLP)
    for k in range(1, n_steps + 1):
        frames.append(go.Frame(
            data=[_trace1(len(x1)), _trace2(_idx(len(x2), k, n_steps)), _op_trace(False)],
            name=f"p2_{k}",
        ))
    # Phase 3 – reveal operating point (3 frames for a "pop")
    for _ in range(3):
        frames.append(go.Frame(
            data=[_trace1(len(x1)), _trace2(len(x2)), _op_trace(True)],
            name="op",
        ))

    init_traces = [_trace1(0), _trace2(0), _op_trace(False)]
    return init_traces, frames


def build_animated_nodal_fig(
    q_ipr, pwf_ipr, vlp_q, vlp_fbhp,
    ipr_name, vlp_name,
    op, Pr, whp, bubble_pt,
    p_y_range, p_axis_max,
    height=580,
):
    """Return a fully animated Nodal Analysis figure."""
    # Clip VLP
    vlp_q    = np.array(vlp_q)
    vlp_fbhp = np.array(vlp_fbhp)
    mask     = vlp_fbhp <= p_axis_max * 1.02
    vlp_q_c, vlp_fbhp_c = vlp_q[mask], vlp_fbhp[mask]

    init_traces, frames = _anim_frames_two_curves(
        x1=q_ipr,    y1=pwf_ipr,
        x2=vlp_q_c,  y2=vlp_fbhp_c,
        name1=f"IPR ({ipr_name})",   color1=COLORS["ipr_primary"],   dash1="solid",
        name2=f"VLP ({vlp_name})",   color2=COLORS["vlp_primary"],   dash2="dash",
        op_x=op["q"] if op["q"] > 0 else None,
        op_y=op["Pwf"] if op["q"] > 0 else None,
        n_steps=45,
    )

    fig = go.Figure(data=init_traces, frames=frames)

    # Drop-lines (static shapes)
    if op["q"] > 0 and op["Pwf"] > 0:
        for shp in [
            dict(type="line", x0=0, x1=op["q"],    y0=op["Pwf"], y1=op["Pwf"],
                 line=dict(color=COLORS["operating"], width=1, dash="dot")),
            dict(type="line", x0=op["q"], x1=op["q"], y0=0,        y1=op["Pwf"],
                 line=dict(color=COLORS["operating"], width=1, dash="dot")),
        ]:
            fig.add_shape(**shp)

    # Reference hlines
    if bubble_pt <= p_axis_max:
        fig.add_hline(y=bubble_pt, line_dash="dot",
                      line_color="rgba(251,191,36,0.5)", line_width=1.5,
                      annotation_text=f"Pb = {bubble_pt:.0f} psia",
                      annotation_font_color="#fbbf24", annotation_font_size=10)
    if Pr <= p_axis_max:
        fig.add_hline(y=Pr, line_dash="dot",
                      line_color="rgba(56,189,248,0.3)", line_width=1,
                      annotation_text=f"Pr = {Pr:,.0f} psia",
                      annotation_font_color="#38bdf8", annotation_font_size=10)
    fig.add_hline(y=whp, line_dash="dot",
                  line_color="rgba(148,163,184,0.3)", line_width=1,
                  annotation_text=f"WHP = {whp:.0f} psia",
                  annotation_font_color="#94a3b8", annotation_font_size=10)

    layout = make_base_layout(
        "Nodal Analysis — Inflow vs. Outflow Performance",
        "Flow Rate (STB/day)", "Flowing Bottom-Hole Pressure (psia)",
        height=height, y_range=p_y_range,
    )
    layout["updatemenus"] = []
    fig.update_layout(**layout)
    fig.update_layout(xaxis_range=[0, max(q_ipr) * 1.05])

    return fig


def build_animated_ipr_fig(
    ipr_curves, model_display_map, primary_model,
    op, bubble_pt, p_y_range, p_axis_max,
    height=520,
):
    """Animated IPR all-models comparison figure."""
    model_keys = list(ipr_curves.keys())
    n = len(model_keys)
    n_steps_each = max(20, 80 // n)

    # Build frames: each model draws in sequence
    total_frames = n * n_steps_each + 3  # +3 for op-point reveal
    frames = []

    def _clipped(key):
        q, p = ipr_curves[key]
        q, p = np.array(q), np.array(p)
        m = p <= p_axis_max * 1.02
        return q[m], p[m]

    def _trace(key, n_pts):
        q, p = _clipped(key)
        is_prim = (key == primary_model)
        label = model_display_map.get(key, key)
        return go.Scatter(
            x=q[:n_pts], y=p[:n_pts],
            name=label,
            mode="lines",
            line=dict(
                color=COLORS.get(key, "#94a3b8"),
                width=3 if is_prim else 1.5,
                dash="solid" if is_prim else "dash",
            ),
            hovertemplate="%{x:.0f} STB/d | %{y:.0f} psia<extra>" + label + "</extra>",
        )

    def _op_trace(vis):
        return go.Scatter(
            x=[op["q"]] if vis and op["q"] > 0 else [],
            y=[op["Pwf"]] if vis and op["q"] > 0 else [],
            name="Operating Point", mode="markers",
            marker=dict(symbol="star", size=18, color=COLORS["operating"],
                        line=dict(color="white", width=1.5)),
        )

    def _snap(drawn_up_to_model_idx, partial_pts):
        """Snapshot of all traces given how far we've animated."""
        data = []
        for mi, key in enumerate(model_keys):
            q, p = _clipped(key)
            if mi < drawn_up_to_model_idx:
                data.append(_trace(key, len(q)))
            elif mi == drawn_up_to_model_idx:
                data.append(_trace(key, partial_pts))
            else:
                data.append(_trace(key, 0))
        return data

    for mi, key in enumerate(model_keys):
        q, _ = _clipped(key)
        for k in range(1, n_steps_each + 1):
            pts = max(1, round(len(q) * k / n_steps_each))
            fd = _snap(mi, pts) + [_op_trace(False)]
            frames.append(go.Frame(data=fd, name=f"m{mi}_k{k}"))

    # reveal op point
    final_traces = _snap(n, 0) + [_op_trace(True)]
    for _ in range(3):
        frames.append(go.Frame(data=final_traces, name="op"))

    # initial (empty)
    init_data = [_trace(k, 0) for k in model_keys] + [_op_trace(False)]
    fig = go.Figure(data=init_data, frames=frames)

    if bubble_pt <= p_axis_max:
        fig.add_hline(y=bubble_pt, line_dash="dot",
                      line_color="rgba(251,191,36,0.5)", line_width=1.5,
                      annotation_text=f"Pb = {bubble_pt:.0f} psia",
                      annotation_font_color="#fbbf24", annotation_font_size=10)

    layout = make_base_layout(
        "Inflow Performance Relationship — All Models",
        "Flow Rate (STB/day)", "Flowing Bottom-Hole Pressure (psia)",
        height=height, y_range=p_y_range,
    )
    layout["updatemenus"] = []
    fig.update_layout(**layout)

    # Fix X-axis to full range from the start so animation is fully visible
    all_q_vals = [q for key in model_keys for q in _clipped(key)[0]]
    q_max = max(all_q_vals) if all_q_vals else 10000
    fig.update_layout(xaxis_range=[0, q_max * 1.08])

    return fig


def build_animated_vlp_fig(
    multi_vlp, ipr_q, ipr_pwf, ipr_name,
    primary_tubing_id, op,
    p_y_range, p_axis_max,
    height=480,
):
    """Animated multi-tubing VLP comparison figure."""
    tubing_keys = list(multi_vlp.keys())
    n = len(tubing_keys)
    n_steps_each = max(20, 80 // (n + 1))  # +1 for IPR

    def _clipped_vlp(key):
        q, f = multi_vlp[key]
        q, f = np.array(q), np.array(f)
        m = f <= p_axis_max * 1.02
        return q[m], f[m]

    def _clipped_ipr():
        q, p = np.array(ipr_q), np.array(ipr_pwf)
        m = p <= p_axis_max * 1.02
        return q[m], p[m]

    def _trace_vlp(key, n_pts, idx):
        q, f = _clipped_vlp(key)
        is_prim = abs(STANDARD_TUBING_SIZES.get(key, 0) - primary_tubing_id) < 0.01
        return go.Scatter(
            x=q[:n_pts], y=f[:n_pts],
            name=key, mode="lines",
            line=dict(color=TUBING_COLORS[idx % len(TUBING_COLORS)],
                      width=3 if is_prim else 1.5,
                      dash="solid" if is_prim else "dash"),
            hovertemplate="%{x:.0f} STB/d | %{y:.0f} psia<extra>" + key + "</extra>",
        )

    def _trace_ipr(n_pts):
        q, p = _clipped_ipr()
        return go.Scatter(
            x=q[:n_pts], y=p[:n_pts],
            name=f"IPR ({ipr_name})", mode="lines",
            line=dict(color=COLORS["ipr_primary"], width=2.5, dash="dot"),
        )

    def _op_trace(vis):
        return go.Scatter(
            x=[op["q"]] if vis and op["q"] > 0 else [],
            y=[op["Pwf"]] if vis and op["q"] > 0 else [],
            name="Operating Point", mode="markers",
            marker=dict(symbol="star", size=18, color=COLORS["operating"],
                        line=dict(color="white", width=1.5)),
        )

    # All curve keys in draw order: VLP tubing sizes then IPR
    all_keys = tubing_keys  # IPR drawn last
    total_phases = len(all_keys) + 1  # +1 for IPR

    def _snap(phase_idx, partial_pts):
        traces = []
        for ti, key in enumerate(tubing_keys):
            q, _ = _clipped_vlp(key)
            if ti < phase_idx:
                traces.append(_trace_vlp(key, len(q), ti))
            elif ti == phase_idx:
                traces.append(_trace_vlp(key, partial_pts, ti))
            else:
                traces.append(_trace_vlp(key, 0, ti))
        # IPR phase
        q_ipr_c, _ = _clipped_ipr()
        if phase_idx > len(tubing_keys):
            traces.append(_trace_ipr(len(q_ipr_c)))
        elif phase_idx == len(tubing_keys):
            traces.append(_trace_ipr(partial_pts))
        else:
            traces.append(_trace_ipr(0))
        return traces

    frames = []
    for pi in range(len(all_keys)):
        q, _ = _clipped_vlp(all_keys[pi])
        for k in range(1, n_steps_each + 1):
            pts = max(1, round(len(q) * k / n_steps_each))
            fd = _snap(pi, pts) + [_op_trace(False)]
            frames.append(go.Frame(data=fd, name=f"v{pi}_k{k}"))
    # IPR phase
    q_ipr_c, _ = _clipped_ipr()
    for k in range(1, n_steps_each + 1):
        pts = max(1, round(len(q_ipr_c) * k / n_steps_each))
        fd = _snap(len(all_keys), pts) + [_op_trace(False)]
        frames.append(go.Frame(data=fd, name=f"ipr_k{k}"))
    # Op point reveal
    final = _snap(total_phases, 0) + [_op_trace(True)]
    for _ in range(3):
        frames.append(go.Frame(data=final, name="op"))

    init_data = ([_trace_vlp(k, 0, i) for i, k in enumerate(tubing_keys)]
                 + [_trace_ipr(0), _op_trace(False)])
    fig = go.Figure(data=init_data, frames=frames)

    layout = make_base_layout(
        "VLP — Tubing Performance Curves by Size",
        "Flow Rate (STB/day)", "Flowing BHP at Perforations (psia)",
        height=height, y_range=p_y_range,
    )
    layout["updatemenus"] = []
    fig.update_layout(**layout)

    # Fix X-axis to full range from the start so animation is fully visible
    all_q_vlp = [q for key in tubing_keys for q in _clipped_vlp(key)[0]]
    all_q_ipr = list(_clipped_ipr()[0])
    all_q = all_q_vlp + all_q_ipr
    q_max = max(all_q) if all_q else 10000
    fig.update_layout(xaxis_range=[0, q_max * 1.08])

    return fig


# ══════════════════════════════════════════════════════════════════════════════ #
#  SIDEBAR — ALL INPUTS                                                         #
# ══════════════════════════════════════════════════════════════════════════════ #
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
      <div style="font-size:2rem;">🛢️</div>
      <div style="font-size:1rem; font-weight:700; color:#38bdf8;">Nodal Analysis</div>
      <div style="font-size:0.7rem; color:#475569; margin-top:0.2rem;">Petroleum Engineering Suite</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. RESERVOIR DATA ─────────────────────────────────────────────── #
    st.markdown("### 🏔️ Reservoir Data")
    Pr    = st.number_input("Reservoir Pressure (psia)", 100.0, 20000.0, 4200.0, 50.0)
    T_res = st.number_input("Reservoir Temperature (°F)", 60.0, 400.0, 180.0, 5.0)
    PI    = st.number_input("Productivity Index (STB/day/psi)", 0.01, 100.0, 2.5, 0.1,
                            format="%.3f")
    depth_res = st.number_input("Reservoir Depth / TVD (ft)", 1000.0, 30000.0, 9500.0, 100.0)

    # ── 2. FLUID DATA ─────────────────────────────────────────────────── #
    st.markdown("### 🧪 Fluid Properties")
    api_gravity = st.number_input("API Gravity (°API)", 10.0, 60.0, 35.0, 1.0)
    gas_sg      = st.number_input("Gas Specific Gravity (air=1)", 0.50, 1.20, 0.65, 0.01,
                                  format="%.3f")
    water_sg    = st.number_input("Water Specific Gravity (water=1)", 1.00, 1.25, 1.07, 0.01,
                                  format="%.3f")
    gor         = st.number_input("Producing GOR (scf/STB)", 0.0, 10000.0, 800.0, 50.0)
    water_cut   = st.slider("Water Cut (%)", 0, 100, 20, 1) / 100.0
    T_surf      = st.number_input("Surface Temperature (°F)", 40.0, 120.0, 75.0, 5.0)

    # ── 3. WELL & TUBING ──────────────────────────────────────────────── #
    st.markdown("### 🔧 Well & Tubing")
    tvd  = st.number_input("True Vertical Depth (ft)", 500.0, 30000.0, 9500.0, 100.0)
    md   = st.number_input("Measured Depth (ft)", 500.0, 35000.0, 9700.0, 100.0)
    whp  = st.number_input("Wellhead Pressure / THP (psia)", 14.7, 3000.0, 100.0, 10.0)

    tubing_names = list(STANDARD_TUBING_SIZES.keys())
    sel_tubing   = st.selectbox("Primary Tubing Size", tubing_names, index=1)
    tubing_id    = STANDARD_TUBING_SIZES[sel_tubing]
    roughness    = st.number_input("Pipe Roughness (ft)", 0.00001, 0.01, 0.0006, 0.0001,
                                   format="%.5f")

    # ── 4. COMPLETION ─────────────────────────────────────────────────── #
    st.markdown("### 🔩 Completion")
    perf_top    = st.number_input("Perforation Top (ft)", 100.0, 30000.0, 9350.0, 50.0)
    perf_bottom = st.number_input("Perforation Bottom (ft)", 100.0, 30000.0, 9500.0, 50.0)
    safety_dist = st.number_input("Packer Safety Distance (ft)", 50.0, 1000.0, 120.0, 10.0)

    # ── 5. ANALYSIS OPTIONS ───────────────────────────────────────────── #
    st.markdown("### ⚙️ Analysis Options")
    ipr_model_display = st.selectbox(
        "IPR Model",
        ["Vogel (1968)", "Linear PI", "Fetkovitch (1973)", "Standing (1970)", "Jones Composite"],
        index=0,
    )
    ipr_model_map = {
        "Vogel (1968)": "vogel",
        "Linear PI": "linear_pi",
        "Fetkovitch (1973)": "fetkovitch",
        "Standing (1970)": "standing",
        "Jones Composite": "jones",
    }
    ipr_model = ipr_model_map[ipr_model_display]

    vlp_corr_display = st.selectbox(
        "VLP Correlation",
        ["Beggs & Brill (1973)", "Hagedorn & Brown (1965)"],
        index=0,
    )
    vlp_corr_map = {
        "Beggs & Brill (1973)": "beggs_brill",
        "Hagedorn & Brown (1965)": "hagedorn_brown",
    }
    vlp_correlation = vlp_corr_map[vlp_corr_display]

    # Jones parameters (shown only if Jones selected)
    if ipr_model == "jones":
        st.markdown("**Jones Coefficients**")
        jones_a = st.number_input("Non-Darcy coeff A (psi/(STB/d)²)", 0.0, 1.0, 0.001, 0.0001,
                                  format="%.5f")
        jones_b = st.number_input("Darcy coeff B (psi/(STB/d))", 0.001, 10.0, 1.0/PI, 0.01,
                                  format="%.4f")
    else:
        jones_a = 0.0
        jones_b = 1.0 / PI if PI > 0 else 1.0

    if ipr_model == "fetkovitch":
        fetkovitch_n = st.slider("Fetkovitch n exponent", 0.5, 1.0, 0.85, 0.05)
    else:
        fetkovitch_n = 1.0

    if ipr_model == "standing":
        fe = st.slider("Completion Efficiency FE", 0.5, 1.5, 1.0, 0.05)
    else:
        fe = 1.0

    st.markdown("---")
    run_btn = st.button("🚀 Run Nodal Analysis", type="primary")


# ══════════════════════════════════════════════════════════════════════════════ #
#  HEADER                                                                       #
# ══════════════════════════════════════════════════════════════════════════════ #
st.markdown("""
<div class="header-banner">
  <div class="header-title">🛢️ Nodal Analysis Suite</div>
  <div class="header-sub">
    PIPESIM-grade production analysis · IPR × VLP intersection · 
    Multiphase flow · PVT correlations · Completion design
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════ #
#  SESSION STATE & COMPUTATION                                                  #
# ══════════════════════════════════════════════════════════════════════════════ #
@st.cache_data(show_spinner=False)
def run_analysis(
    Pr, T_res, PI, tvd, md, tubing_id, roughness, whp, T_surf,
    api_gravity, gas_sg, water_sg, gor, water_cut,
    perf_top, perf_bottom, safety_dist,
    ipr_model, vlp_correlation,
    jones_a, jones_b, fetkovitch_n, fe,
):
    fp = FluidProperties(
        api_gravity=api_gravity,
        gas_sg=gas_sg,
        water_sg=water_sg,
        gor=gor,
        water_cut=water_cut,
        reservoir_temp=T_res,
    )

    ipr_calc = IPRCalculator(
        reservoir_pressure=Pr,
        productivity_index=PI,
        bubble_point=fp.bubble_point,
        reservoir_temp=T_res,
        jones_a=jones_a,
        jones_b=jones_b,
        fetkovitch_n=fetkovitch_n,
        completion_efficiency=fe,
    )

    vlp_calc = VLPCalculator(
        fluid_props=fp,
        tvd=tvd,
        md=md,
        tubing_id=tubing_id,
        wellhead_pressure=whp,
        reservoir_temp=T_res,
        surface_temp=T_surf,
        roughness=roughness,
        n_segments=40,
    )

    solver = NodalAnalysis(ipr_calc=ipr_calc, vlp_calc=vlp_calc)
    results = solver.full_analysis(ipr_model, vlp_correlation, n_curve_points=80)

    # Pressure profile for the operating point
    if results["operating_point"]["q"] > 0:
        profile_data = vlp_calc.compute_fbhp(
            results["operating_point"]["q"], vlp_correlation
        )["profile"]
    else:
        profile_data = []

    # Multi-tubing VLP comparison
    multi_vlp = vlp_calc.multi_tubing_vlp(
        STANDARD_TUBING_SIZES,
        q_max=max(results["operating_point"]["aof"] * 1.1, 200.0),
        correlation=vlp_correlation,
        n_points=50,
    )

    # Completion design
    designer = CompletionDesigner(
        well_tvd=tvd,
        well_md=md,
        perf_top=perf_top,
        perf_bottom=perf_bottom,
        safety_margin=safety_dist,
    )

    # Find the primary tubing name from its ID
    primary_tubing_name = next(
        (k for k, v in STANDARD_TUBING_SIZES.items() if abs(v - tubing_id) < 0.01),
        "Custom"
    )
    completion_string = designer.design_string(primary_tubing_name, tubing_id)
    completion_summary = designer.summary_table(completion_string)

    # WHP sensitivity (quick)
    whp_vals = list(range(50, 851, 50))
    whp_sens = []
    for w in whp_vals:
        vlp_s = VLPCalculator(fp, tvd, md, tubing_id, w, T_res, T_surf, roughness, 25)
        s = NodalAnalysis(ipr_calc, vlp_s)
        op = s.operating_point(ipr_model, vlp_correlation)
        whp_sens.append({"WHP (psia)": w, "Flow Rate (STB/d)": op["q"], "FBHP (psia)": op["Pwf"]})

    return {
        "fp": fp,
        "ipr_calc": ipr_calc,
        "results": results,
        "profile_data": profile_data,
        "multi_vlp": multi_vlp,
        "completion_string": completion_string,
        "completion_summary": completion_summary,
        "whp_sensitivity": whp_sens,
        "designer": designer,
    }


# Run on button press (or if no previous run)
if run_btn or "last_results" not in st.session_state:
    with st.spinner("⚙️  Running nodal analysis..."):
        try:
            data = run_analysis(
                Pr, T_res, PI, tvd, md, tubing_id, roughness, whp, T_surf,
                api_gravity, gas_sg, water_sg, gor, water_cut,
                perf_top, perf_bottom, safety_dist,
                ipr_model, vlp_correlation,
                jones_a, jones_b, fetkovitch_n, fe,
            )
            st.session_state["last_results"] = data
            st.session_state["last_inputs"] = {
                "ipr_model": ipr_model,
                "vlp_correlation": vlp_correlation,
                "tubing_id": tubing_id,
                "sel_tubing": sel_tubing,
            }
        except Exception as e:
            st.error("❌ **Simulation Error:** The combination of input parameters (e.g. extreme flow rate, very small tubing size, or out-of-range pressures) resulted in unresolvable flow conditions or excessive pressure losses. Please adjust your parameters to within standard operating bounds.")
            st.stop()
else:
    data = st.session_state["last_results"]

results = data["results"]
op      = results["operating_point"]
fp      = data["fp"]

# ── Pressure axis ceiling: a bit above Pr, but never below 6000 psia ── #
P_AXIS_MAX = max(round(Pr * 1.15 / 1000) * 1000, 6000)
P_Y_RANGE  = [0, P_AXIS_MAX]


# ══════════════════════════════════════════════════════════════════════════════ #
#  KPI ROW                                                                      #
# ══════════════════════════════════════════════════════════════════════════════ #
drawdown = Pr - op["Pwf"]
drawdown_pct = (drawdown / Pr * 100) if Pr > 0 else 0

bubble_pt = fp.bubble_point
bo_at_op  = fp.oil_fvf(op["Pwf"], T_res) if op["Pwf"] > 0 else 1.0

kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-value">{op['q']:,.0f}</div>
    <div class="kpi-unit">STB/day</div>
    <div class="kpi-label">Operating Rate</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{op['Pwf']:,.0f}</div>
    <div class="kpi-unit">psia</div>
    <div class="kpi-label">FBHP</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{drawdown:,.0f}</div>
    <div class="kpi-unit">psi ({drawdown_pct:.1f}%)</div>
    <div class="kpi-label">Drawdown</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{op['aof']:,.0f}</div>
    <div class="kpi-unit">STB/day</div>
    <div class="kpi-label">AOF</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{bubble_pt:,.0f}</div>
    <div class="kpi-unit">psia</div>
    <div class="kpi-label">Bubble-Point</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{bo_at_op:.3f}</div>
    <div class="kpi-unit">RB/STB</div>
    <div class="kpi-label">Oil FVF (Bₒ)</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# Operating point status message
if op["found"]:
    st.success(f"✅ {op['message']}")
else:
    st.warning(f"⚠️ {op['message']}")

st.info("ℹ️ **Test Data Notice:** Internally calculated wellbore trajectories (MD, TVD, inclination), pressure-dependent PVT tables (P, Bo, Rs, viscosity, z-factor), and depth vs. temperature profiles are generated using standard benchmark test datasets.")



# ══════════════════════════════════════════════════════════════════════════════ #
#  MAIN TABS                                                                    #
# ══════════════════════════════════════════════════════════════════════════════ #
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Nodal Plot",
    "📈 IPR Curves",
    "📉 VLP / Tubing",
    "🔢 Results & Losses",
    "🛠️ Completion Design",
    "📋 Report",
])


# ════════════════════════════════════════════════════════════════════════ #
#  TAB 1 — NODAL PLOT (IPR + VLP + Operating Point)                       #
# ════════════════════════════════════════════════════════════════════════ #
with tab1:
    st.markdown('<div class="section-header">📊 Nodal Analysis — IPR ∩ VLP</div>',
                unsafe_allow_html=True)

    col_plot, col_info = st.columns([3, 1])

    q_ipr, pwf_ipr = results["ipr_curves"][ipr_model]

    with col_plot:
        fig = build_animated_nodal_fig(
            q_ipr=q_ipr,
            pwf_ipr=pwf_ipr,
            vlp_q=results["vlp_q"],
            vlp_fbhp=results["vlp_fbhp"],
            ipr_name=ipr_model_display,
            vlp_name=vlp_corr_display,
            op=op,
            Pr=Pr,
            whp=whp,
            bubble_pt=bubble_pt,
            p_y_range=P_Y_RANGE,
            p_axis_max=P_AXIS_MAX,
            height=580,
        )
        render_animated(fig, height=580)

    with col_info:
        st.markdown("**Analysis Settings**")
        st.markdown(f"""
        <div class="info-box">
          <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.5rem;">Configuration</div>
          <div style="font-size:0.85rem; color:#e2e8f0;">
            <b>IPR:</b> {ipr_model_display}<br>
            <b>VLP:</b> {vlp_corr_display}<br>
            <b>Tubing:</b> {sel_tubing}<br>
            <b>WHP:</b> {whp:.0f} psia<br>
            <b>WC:</b> {water_cut*100:.0f}%<br>
            <b>GOR:</b> {gor:.0f} scf/STB<br>
            <b>API:</b> {api_gravity:.0f}°<br>
            <b>Pb:</b> {bubble_pt:,.0f} psia
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Operating Point**")
        st.markdown(f"""
        <div class="info-box">
          <div style="font-size:0.85rem; color:#e2e8f0;">
            <b>q =</b> <span style="color:#38bdf8">{op['q']:,.0f} STB/d</span><br>
            <b>Pwf =</b> <span style="color:#fbbf24">{op['Pwf']:,.0f} psia</span><br>
            <b>AOF =</b> {op['aof']:,.0f} STB/d<br>
            <b>Drawdown =</b> {drawdown:,.0f} psi<br>
            <b>Eff. =</b> {(op['q']/op['aof']*100) if op['aof']>0 else 0:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

        # PVT summary
        st.markdown("**PVT @ Operating Pwf**")
        mu_o = fp.oil_viscosity(op["Pwf"], T_res) if op["Pwf"] > 0 else 0
        mu_g = fp.gas_viscosity(op["Pwf"], T_res) if op["Pwf"] > 0 else 0
        Rs_op = fp.solution_gor(op["Pwf"], T_res) if op["Pwf"] > 0 else gor
        z_op  = fp.gas_z_factor(op["Pwf"], T_res) if op["Pwf"] > 0 else 1
        st.markdown(f"""
        <div class="info-box">
          <div style="font-size:0.85rem; color:#e2e8f0;">
            <b>Bₒ =</b> {bo_at_op:.4f} RB/STB<br>
            <b>Rs =</b> {Rs_op:.0f} scf/STB<br>
            <b>μₒ =</b> {mu_o:.3f} cp<br>
            <b>μₘ =</b> {mu_g:.4f} cp (gas)<br>
            <b>z =</b> {z_op:.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════ #
#  TAB 2 — IPR CURVES (all models comparison)                             #
# ════════════════════════════════════════════════════════════════════════ #
with tab2:
    st.markdown('<div class="section-header">📈 IPR Curves — All Models Comparison</div>',
                unsafe_allow_html=True)

    model_display_map = {
        "vogel": "Vogel (1968)",
        "linear_pi": "Linear PI",
        "fetkovitch": "Fetkovitch (1973)",
        "standing": "Standing (1970)",
        "jones": "Jones Composite",
    }

    fig2 = build_animated_ipr_fig(
        ipr_curves=results["ipr_curves"],
        model_display_map=model_display_map,
        primary_model=ipr_model,
        op=op,
        bubble_pt=bubble_pt,
        p_y_range=P_Y_RANGE,
        p_axis_max=P_AXIS_MAX,
        height=520,
    )
    render_animated(fig2, height=520)

    # AOF comparison table
    st.markdown("**AOF Comparison by Model**")
    aof_table = []
    ipr_c = data["ipr_calc"]
    for mk, disp in model_display_map.items():
        try:
            aof_v = ipr_c.aof(mk)
            aof_table.append({"Model": disp, "AOF (STB/day)": f"{aof_v:,.0f}",
                               "Note": "⭐ Selected" if mk == ipr_model else ""})
        except:
            aof_table.append({"Model": disp, "AOF (STB/day)": "N/A", "Note": ""})
    st.dataframe(pd.DataFrame(aof_table), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════ #
#  TAB 3 — VLP / TUBING SIZE COMPARISON + PRESSURE PROFILE               #
# ════════════════════════════════════════════════════════════════════════ #
with tab3:
    st.markdown('<div class="section-header">📉 VLP Curves — Tubing Performance</div>',
                unsafe_allow_html=True)

    col_vlp1, col_vlp2 = st.columns([3, 2])

    with col_vlp1:
        st.markdown("**Multi-Tubing Size VLP Comparison**")
        fig3 = build_animated_vlp_fig(
            multi_vlp=data["multi_vlp"],
            ipr_q=results["ipr_curves"][ipr_model][0],
            ipr_pwf=results["ipr_curves"][ipr_model][1],
            ipr_name=ipr_model_display,
            primary_tubing_id=tubing_id,
            op=op,
            p_y_range=P_Y_RANGE,
            p_axis_max=P_AXIS_MAX,
            height=480,
        )
        render_animated(fig3, height=480)

    with col_vlp2:
        st.markdown("**Wellbore Pressure Profile**")
        st.caption("ℹ️ *Profile & trajectory (MD/TVD/inclination/temp) based on internal test dataset models.*")
        if data["profile_data"]:
            depths_p = [d for d, _ in data["profile_data"]]
            press_p  = [p for _, p in data["profile_data"]]

            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(
                x=press_p, y=depths_p,
                mode="lines+markers",
                name="Pressure Profile",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=4, color="#38bdf8"),
                hovertemplate="Depth: %{y:.0f} ft<br>Pressure: %{x:.0f} psia<extra></extra>",
            ))
            fig_prof.update_layout(
                **make_base_layout(
                    f"Pressure Profile @ {op['q']:.0f} STB/d",
                    "Pressure (psia)", "Depth (ft)", height=480
                )
            )
            fig_prof.update_yaxes(autorange="reversed", title_text="Depth (ft)")
            st.plotly_chart(fig_prof, use_container_width=True)
        else:
            st.info("Run analysis to see pressure profile.")

    # WHP Sensitivity
    st.markdown('<div class="section-header">🔎 WHP Sensitivity</div>',
                unsafe_allow_html=True)
    sens_df = pd.DataFrame(data["whp_sensitivity"])
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["WHP (psia)"], y=sens_df["Flow Rate (STB/d)"],
        mode="lines+markers",
        name="Flow Rate",
        line=dict(color=COLORS["ipr_primary"], width=2.5),
        marker=dict(size=6),
        hovertemplate="WHP: %{x:.0f} psia<br>Rate: %{y:.0f} STB/d<extra></extra>",
    ))
    # Current WHP marker
    fig_sens.add_vline(x=whp, line_dash="dot",
                       line_color="rgba(248,113,113,0.7)", line_width=1.5,
                       annotation_text=f"Current WHP = {whp:.0f} psia",
                       annotation_font_color="#f87171", annotation_font_size=10)
    fig_sens.update_layout(**make_base_layout(
        "Flow Rate vs. Wellhead Pressure (WHP Sensitivity)",
        "Wellhead Pressure (psia)", "Flow Rate (STB/day)", height=380
    ))
    st.plotly_chart(fig_sens, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════ #
#  TAB 4 — RESULTS & PRESSURE LOSSES                                      #
# ════════════════════════════════════════════════════════════════════════ #
with tab4:
    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        st.markdown('<div class="section-header">🔢 Reservoir & Well Summary</div>',
                    unsafe_allow_html=True)
        res_info = results["reservoir_info"]
        res_df = pd.DataFrame([
            {"Parameter": k, "Value": str(v)}
            for k, v in res_info.items()
        ])
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # PVT table at key pressures
        st.markdown('<div class="section-header">🧪 PVT Properties</div>',
                    unsafe_allow_html=True)
        st.caption("ℹ️ *Pressure-dependent table (P, Bo, Rs, viscosity, z-factor) calculated from internal test dataset models.*")
        pvt_pressures = [Pr, bubble_pt, op["Pwf"], whp] if op["Pwf"] > 0 else [Pr, bubble_pt, whp]
        pvt_pressures = sorted(set([max(14.7, p) for p in pvt_pressures]), reverse=True)
        pvt_rows = []
        for p in pvt_pressures:
            rs  = fp.solution_gor(p, T_res)
            bo  = fp.oil_fvf(p, T_res)
            muo = fp.oil_viscosity(p, T_res)
            mug = fp.gas_viscosity(p, T_res)
            z   = fp.gas_z_factor(p, T_res)
            bg  = fp.gas_fvf(p, T_res) * 1000  # mscf/mcf → show in 1e-3
            pvt_rows.append({
                "P (psia)": f"{p:,.0f}",
                "Rs (scf/STB)": f"{rs:.0f}",
                "Bₒ (RB/STB)": f"{bo:.4f}",
                "μₒ (cp)": f"{muo:.3f}",
                "z-factor": f"{z:.4f}",
                "μg (cp)": f"{mug:.4f}",
            })
        st.dataframe(pd.DataFrame(pvt_rows), use_container_width=True, hide_index=True)

    with col_res2:
        st.markdown('<div class="section-header">📉 Pressure Loss Breakdown</div>',
                    unsafe_allow_html=True)

        pl = results["pressure_losses"]
        grav = pl["Hydrostatic (Gravity)"]
        fric = pl["Friction"]
        acc  = pl["Acceleration"]
        tot  = pl["Total ΔP"]

        # Donut chart
        if tot > 0:
            fig_pl = go.Figure(data=[go.Pie(
                labels=["Hydrostatic (Gravity)", "Friction", "Acceleration"],
                values=[max(0, grav), max(0, fric), max(0, acc)],
                hole=0.55,
                marker=dict(colors=[COLORS["gravity"], COLORS["friction"], COLORS["accel"]],
                            line=dict(color="#0a0e1a", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#e2e8f0", size=11),
                hovertemplate="%{label}: %{value:.1f} psi (%{percent})<extra></extra>",
            )])
            fig_pl.update_layout(
                paper_bgcolor=PLOT_PAPER,
                plot_bgcolor=PLOT_BG,
                font=dict(family="Inter", color=FONT_COLOR),
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                title=dict(text=f"Total ΔP = {tot:.0f} psi", font=dict(color="#e2e8f0")),
                legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(10,14,26,0.8)"),
            )
            st.plotly_chart(fig_pl, use_container_width=True)

        # Table
        pl_table = [
            {"Component": "Hydrostatic (Gravity)", "Pressure Loss (psi)": f"{grav:.1f}",
             "% of Total": f"{grav/tot*100:.1f}%" if tot > 0 else "—"},
            {"Component": "Friction", "Pressure Loss (psi)": f"{fric:.1f}",
             "% of Total": f"{fric/tot*100:.1f}%" if tot > 0 else "—"},
            {"Component": "Acceleration", "Pressure Loss (psi)": f"{acc:.1f}",
             "% of Total": f"{acc/tot*100:.1f}%" if tot > 0 else "—"},
            {"Component": "Total ΔP (WHP→FBHP)", "Pressure Loss (psi)": f"{tot:.1f}",
             "% of Total": "100%"},
        ]
        st.dataframe(pd.DataFrame(pl_table), use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="info-box">
          <div style="font-size:0.85rem; color:#e2e8f0;">
            <b>WHP (THP):</b>  {whp:.0f} psia<br>
            <b>Total ΔP:</b>   {tot:.0f} psi<br>
            <b>FBHP:</b>       {pl['FBHP']:.0f} psia<br>
            <b>Correlation:</b> {vlp_corr_display}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Stacked bar for pressure buildup
        st.markdown("**Pressure Buildup (Wellhead → Perforations)**")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="WHP", x=["Pressure Buildup"],
            y=[whp], marker_color="rgba(148,163,184,0.5)",
        ))
        fig_bar.add_trace(go.Bar(
            name="Hydrostatic", x=["Pressure Buildup"],
            y=[grav], marker_color=COLORS["gravity"],
        ))
        fig_bar.add_trace(go.Bar(
            name="Friction", x=["Pressure Buildup"],
            y=[fric], marker_color=COLORS["friction"],
        ))
        fig_bar.add_trace(go.Bar(
            name="Acceleration", x=["Pressure Buildup"],
            y=[acc], marker_color=COLORS["accel"],
        ))
        fig_bar.update_layout(
            barmode="stack",
            paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
            font=dict(family="Inter", color=FONT_COLOR),
            height=260, margin=dict(l=60, r=10, t=10, b=30),
            legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(10,14,26,0.8)"),
            yaxis=dict(title="Pressure (psia)", gridcolor=GRID_COLOR,
                       tickfont=dict(color=FONT_COLOR)),
            xaxis=dict(tickfont=dict(color=FONT_COLOR)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════ #
#  TAB 5 — COMPLETION DESIGN                                              #
# ════════════════════════════════════════════════════════════════════════ #
with tab5:
    st.markdown('<div class="section-header">🛠️ Completion String Design</div>',
                unsafe_allow_html=True)
    st.caption("ℹ️ *Packer depth and tubing string components are computed from test wellbore trajectory assumptions.*")

    comp = data["completion_string"]
    summary = data["completion_summary"]
    designer = data["designer"]

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown("**Completion String Summary**")
        badge_cls = {
            "VALID": "badge-valid",
            "WARNING": "badge-warning",
            "INVALID": "badge-invalid",
        }.get(comp.status, "badge-warning")
        st.markdown(f'<span class="{badge_cls}">{comp.status}</span>', unsafe_allow_html=True)

        comp_df = pd.DataFrame([
            {"Item": k, "Value": v}
            for k, v in summary.items()
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col_c2:
        # Schematic diagram of completion string
        st.markdown("**Schematic (Depth vs. Component)**")
        fig_comp = go.Figure()

        components = [
            ("Surface", 0, "#94a3b8"),
            ("SCSSV", comp.scssv_depth, "#f472b6"),
            ("Landing Nipple", comp.landing_nipple_depth, "#34d399"),
            ("Seal Assembly", comp.seal_assembly_depth, "#fb923c"),
            ("Packer", comp.packer_depth, "#38bdf8"),
            ("Perf Top", perf_top, "#fbbf24"),
            ("Perf Bottom", perf_bottom, "#fbbf24"),
        ]

        for name, depth, color in components:
            fig_comp.add_trace(go.Scatter(
                x=[0], y=[depth],
                mode="markers+text",
                name=name,
                marker=dict(symbol="line-ew", size=24, color=color,
                            line=dict(color=color, width=3)),
                text=[f" {name} — {depth:.0f} ft"],
                textposition="middle right",
                textfont=dict(color=color, size=10),
                hovertemplate=f"{name}: {depth:.0f} ft<extra></extra>",
            ))

        # Tubing line
        fig_comp.add_shape(type="line",
            x0=0, x1=0, y0=0, y1=comp.packer_depth,
            line=dict(color="rgba(56,189,248,0.6)", width=4),
        )
        # Perforation interval
        fig_comp.add_hrect(
            y0=perf_top, y1=perf_bottom,
            fillcolor="rgba(251,191,36,0.1)",
            line_color="rgba(251,191,36,0.4)",
            annotation_text="Perforations",
            annotation_font_color="#fbbf24",
            annotation_font_size=9,
        )

        fig_comp.update_layout(
            **make_base_layout("Completion String Schematic", "", "Depth (ft)", height=520)
        )
        fig_comp.update_yaxes(autorange="reversed", title_text="Depth (ft)")
        fig_comp.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                               title_text="")
        fig_comp.update_layout(showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════ #
#  TAB 6 — FULL REPORT                                                    #
# ════════════════════════════════════════════════════════════════════════ #
with tab6:
    st.markdown('<div class="section-header">📋 Analysis Report</div>',
                unsafe_allow_html=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_md = f"""
# Nodal Analysis Report
**Generated:** {now}

> ℹ️ **Test Data Notice:** Wellbore trajectory (MD, TVD, inclination), pressure-dependent PVT tables (P, Bo, Rs, viscosity, z-factor), and depth vs. temperature profiles are generated using standard benchmark test datasets.

---

## Well & Reservoir Summary

| Parameter | Value |
|-----------|-------|
| Reservoir Pressure | {Pr:,.0f} psia |
| Reservoir Temperature | {T_res:.0f} °F |
| Productivity Index | {PI:.3f} STB/day/psi |
| Bubble-Point Pressure | {bubble_pt:,.0f} psia |
| TVD | {tvd:,.0f} ft |
| MD | {md:,.0f} ft |
| Tubing Size | {sel_tubing} |
| Wellhead Pressure | {whp:.0f} psia |

## Fluid Properties

| Parameter | Value |
|-----------|-------|
| API Gravity | {api_gravity:.0f} °API |
| Gas SG | {gas_sg:.3f} |
| Water SG | {water_sg:.3f} |
| Producing GOR | {gor:.0f} scf/STB |
| Water Cut | {water_cut*100:.0f}% |

## Analysis Settings

- **IPR Model:** {ipr_model_display}
- **VLP Correlation:** {vlp_corr_display}

## Operating Point Results

| Result | Value |
|--------|-------|
| **Operating Flow Rate** | **{op['q']:,.0f} STB/day** |
| **FBHP at Perforations** | **{op['Pwf']:,.0f} psia** |
| Drawdown | {drawdown:,.0f} psi ({drawdown_pct:.1f}%) |
| AOF | {op['aof']:,.0f} STB/day |
| Productivity Ratio | {(op['q']/op['aof']*100) if op['aof']>0 else 0:.1f}% |

## Pressure Loss Breakdown

| Component | Loss (psi) | % Total |
|-----------|-----------|---------|
| Hydrostatic (Gravity) | {results['pressure_losses']['Hydrostatic (Gravity)']:.1f} | {results['pressure_losses']['Hydrostatic (Gravity)']/max(results['pressure_losses']['Total ΔP'],1)*100:.1f}% |
| Friction | {results['pressure_losses']['Friction']:.1f} | {results['pressure_losses']['Friction']/max(results['pressure_losses']['Total ΔP'],1)*100:.1f}% |
| Acceleration | {results['pressure_losses']['Acceleration']:.1f} | {results['pressure_losses']['Acceleration']/max(results['pressure_losses']['Total ΔP'],1)*100:.1f}% |
| **Total ΔP** | **{results['pressure_losses']['Total ΔP']:.1f}** | **100%** |

## Completion Design

| Item | Value |
|------|-------|
| Packer Depth | {data['completion_string'].packer_depth:.0f} ft |
| SCSSV Depth | {data['completion_string'].scssv_depth:.0f} ft |
| Landing Nipple Depth | {data['completion_string'].landing_nipple_depth:.0f} ft |
| Seal Assembly Depth | {data['completion_string'].seal_assembly_depth:.0f} ft |
| Status | {data['completion_string'].status} |

---
*Generated by Nodal Analysis Suite — Field Units (psia, STB/day, ft, °F)*
"""

    st.markdown(report_md)

    # Download button
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=report_md.encode("utf-8"),
        file_name=f"nodal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )
