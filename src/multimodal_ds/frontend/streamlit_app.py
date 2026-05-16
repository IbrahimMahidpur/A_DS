import threading
import time
import sys
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add the repository root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from multimodal_ds.frontend.ui_bus_adapter import UIBusAdapter
from multimodal_ds.core.message_bus import MessageType
from multimodal_ds.graph import build_graph, make_initial_state
st.set_page_config(layout="wide", page_title="Multimodal DS Console", page_icon="🧬")
from multimodal_ds.config import OUTPUT_DIR

PRIMARY_GREEN = "#1D9E75"
BG_DARK = "#0F111A"
BG_SECONDARY = "#161B22"
BORDER_COLOR = "#30363D"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
TEXT_TERTIARY = "#6E7681"

# ── Session State Initialization ────────────────────────────────────
if "ui_adapter" not in st.session_state:
    st.session_state.ui_adapter = UIBusAdapter()
if "trace_log" not in st.session_state:
    st.session_state.trace_log = []
if "orchestrator_thread" not in st.session_state:
    st.session_state.orchestrator_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# ── Theme handling & global CSS injection ───────────────────────────
def apply_custom_styles():
    css = f"""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --color-background-primary: {BG_DARK};
        --color-background-secondary: {BG_SECONDARY};
        --color-border-tertiary: {BORDER_COLOR};
        --color-border-secondary: #444c56;
        --color-text-primary: {TEXT_PRIMARY};
        --color-text-secondary: {TEXT_SECONDARY};
        --color-text-tertiary: {TEXT_TERTIARY};
        --color-background-success: rgba(29, 158, 117, 0.15);
        --color-text-success: #1D9E75;
        --color-text-danger: #F85149;
        --font-sans: 'Inter', sans-serif;
        --border-radius-lg: 12px;
        --border-radius-md: 8px;
    }}

    /* Global Overrides */
    .stApp {{ background-color: var(--color-background-primary); color: var(--color-text-primary); font-family: var(--font-sans); }}
    [data-testid="stSidebar"] {{ background-color: var(--color-background-secondary); border-right: 0.5px solid var(--color-border-tertiary); }}
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}
    [data-testid="stHeader"] {{ display: none; }}

    /* Blueprint Layout Components */
    .topbar {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 0.5px solid var(--color-border-tertiary); background: var(--color-background-primary); position: sticky; top: 0; z-index: 100; }}
    .topbar-left {{ display: flex; align-items: center; gap: 12px; }}
    .topbar-title {{ font-size: 15px; font-weight: 500; }}
    .status-badge {{ display: inline-flex; align-items: center; gap: 5px; font-size: 11px; padding: 3px 8px; border-radius: 20px; background: var(--color-background-success); color: var(--color-text-success); font-weight: 500; }}
    .status-badge.idle {{ background: rgba(255,255,255,0.05); color: var(--color-text-tertiary); }}
    .dot {{ width: 6px; height: 6px; border-radius: 50%; }}
    .dot.live {{ background: #1D9E75; box-shadow: 0 0 8px #1D9E75; animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} 100% {{ opacity: 1; }} }}

    .topbar-actions {{ display: flex; gap: 8px; align-items: center; }}
    .btn {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: var(--border-radius-md); font-size: 13px; cursor: pointer; border: 0.5px solid var(--color-border-secondary); background: transparent; color: var(--color-text-primary); transition: 0.2s; text-decoration: none; }}
    .btn:hover {{ background: rgba(255,255,255,0.05); }}
    .btn.primary {{ background: #1D9E75; border-color: #1D9E75; color: #fff; }}
    .btn.danger {{ color: var(--color-text-danger); border-color: var(--color-text-danger); }}

    .content-area {{ padding: 20px; display: flex; flex-direction: column; gap: 16px; }}
    .metrics-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .metric {{ background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px 14px; border: 0.5px solid var(--color-border-tertiary); transition: 0.3s; }}
    .metric-label {{ font-size: 11px; color: var(--color-text-tertiary); margin-bottom: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric-value {{ font-size: 22px; font-weight: 500; color: var(--color-text-primary); }}
    .metric-delta {{ font-size: 11px; color: var(--color-text-success); margin-top: 2px; }}

    .pipeline-card {{ background: var(--color-background-primary); border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); padding: 16px 20px; }}
    .pipeline-stages {{ display: flex; align-items: center; gap: 0; margin: 20px 0; }}
    .stage {{ display: flex; flex-direction: column; align-items: center; gap: 8px; flex: 1; }}
    .stage-icon {{ width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; border: 0.5px solid var(--color-border-tertiary); }}
    .stage-icon.done {{ background: #E1F5EE; color: #0F6E56; border: none; }}
    .stage-icon.active {{ background: #E6F1FB; color: #185FA5; border: none; box-shadow: 0 0 10px rgba(24,95,165,0.2); }}
    .stage-icon.pending {{ background: var(--color-background-secondary); color: var(--color-text-tertiary); }}
    .stage-connector {{ flex: 1; height: 1px; background: var(--color-border-tertiary); margin-bottom: 25px; }}
    .stage-connector.done {{ background: #1D9E75; }}

    .panel {{ background: var(--color-background-primary); border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); overflow: hidden; }}
    .panel-header {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 0.5px solid var(--color-border-tertiary); background: rgba(255,255,255,0.02); }}
    .panel-header-title {{ font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 8px; }}
    .panel-body {{ padding: 12px 16px; }}
    
    .event-row {{ display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 0.5px solid var(--color-border-tertiary); }}
    .event-type-badge {{ font-size: 10px; padding: 2px 8px; border-radius: 20px; font-weight: 600; white-space: nowrap; }}
    .badge-viz {{ background: #E1F5EE; color: #085041; }}
    .badge-model {{ background: #EEEDFE; color: #3C3489; }}
    .badge-data {{ background: #E6F1FB; color: #0C447C; }}
    .badge-err {{ background: #FCEBEB; color: #791F1F; }}
    .badge-sys {{ background: #F1EFE8; color: #444441; }}
    .badge-agent {{ background: #FAEEDA; color: #633806; }}
    
    .viz-item {{ background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px; border: 0.5px solid transparent; transition: 0.2s; }}
    .viz-item:hover {{ border-color: var(--color-border-secondary); }}
    .chart-thumb {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .placeholder-card {{ background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 20px; text-align: center; border: 1px dashed var(--color-border-tertiary); opacity: 0.6; }}

    .sidebar-logo {{ font-size: 14px; font-weight: 600; padding: 10px 4px; border-bottom: 0.5px solid var(--color-border-tertiary); margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
    .nav-item {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: var(--border-radius-md); font-size: 13px; cursor: pointer; color: var(--color-text-secondary); transition: 0.15s; text-decoration: none; margin-bottom: 2px; }}
    .nav-item.active {{ background: var(--color-background-primary); color: var(--color-text-primary); font-weight: 500; border: 0.5px solid var(--color-border-tertiary); }}
    
    .progress-bar-wrap {{ height: 4px; background: var(--color-background-secondary); border-radius: 2px; overflow: hidden; }}
    .progress-bar-fill {{ height: 100%; background: #1D9E75; transition: width 0.5s ease; }}
    
    .dl-chip {{ display: inline-flex; align-items: center; gap: 6px; font-size: 11px; padding: 4px 12px; border-radius: 20px; border: 0.5px solid var(--color-border-secondary); color: var(--color-text-secondary); background: transparent; cursor: pointer; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ── Helper Components ──────────────────────────────────────────────
def render_metric(label, value, delta):
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline(trace_log):
    stages = [("Ingest", "ti ti-upload"), ("Clean", "ti ti-filter"), ("Model", "ti ti-brain"), ("Visualize", "ti ti-chart-dots"), ("Report", "ti ti-file-analytics")]
    # Mapping based on guide suggestions
    stage_idx = -1
    for e in trace_log:
        etype = e.get("type", "")
        if "INGEST" in etype or "DATA_LOADED" in etype: stage_idx = max(stage_idx, 0)
        if "CLEAN" in etype or "ANALYSIS" in etype: stage_idx = max(stage_idx, 1)
        if "MODEL" in etype or "PLANNER" in etype: stage_idx = max(stage_idx, 2)
        if "VIZ" in etype: stage_idx = max(stage_idx, 3)
        if "REPORT" in etype or "SESSION_END" in etype: stage_idx = max(stage_idx, 4)
    
    html = '<div class="pipeline-card"><div class="pipeline-title">Pipeline stages</div><div class="pipeline-stages">'
    for i, (label, icon) in enumerate(stages):
        status = "done" if i < stage_idx else ("active" if i == stage_idx else "pending")
        html += f'<div class="stage"><div class="stage-icon {status}"><i class="{icon}"></i></div><div class="stage-label">{label}</div></div>'
        if i < len(stages) - 1:
            conn_status = "done" if i < stage_idx else ""
            html += f'<div class="stage-connector {conn_status}"></div>'
    
    progress = (stage_idx + 1) / len(stages) * 100 if stage_idx >= 0 else 0
    html += f'</div><div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{progress}%"></div></div></div>'
    st.markdown(html, unsafe_allow_html=True)

def render_trace(trace_log):
    new_events = st.session_state.ui_adapter.drain_queue()
    if new_events: st.session_state.trace_log.extend(new_events)
    
    rows = ""
    for entry in reversed(st.session_state.trace_log[-50:]):
        etype = str(entry.get("type", "DATA")).upper()
        # FAANG Blueprint Badge Map
        tag = "DATA"
        b_class = "badge-data"
        if "VIZ" in etype: tag, b_class = "VIZ", "badge-viz"
        elif "MODEL" in etype: tag, b_class = "MODEL", "badge-model"
        elif "ERR" in etype or "FAIL" in etype: tag, b_class = "ERR", "badge-err"
        elif "SESSION" in etype or "SYS" in etype: tag, b_class = "SYS", "badge-sys"
        elif "AGENT" in etype or "PLAN" in etype: tag, b_class = "AGENT", "badge-agent"
        
        rows += f"""
        <div class="event-row">
            <div class="event-type-badge {b_class}">{tag}</div>
            <div class="ev-meta">
                <div class="ev-sender" style="font-size:12px; font-weight:500;">{entry.get('sender', 'Agent')}</div>
                <div class="ev-text" style="font-size:11px; color:var(--color-text-tertiary);">{str(entry.get('payload', ''))[:120]}</div>
            </div>
            <div class="ev-time" style="font-size:10px; color:var(--color-text-tertiary);">{entry.get('timestamp', datetime.now().strftime('%H:%M:%S'))}</div>
        </div>
        """
    
    st.markdown(f"""
    <div class="panel">
        <div class="panel-header"><div class="panel-header-title"><i class="ti ti-activity"></i>Live agent trace</div><div class="chip"><i class="ti ti-refresh"></i>Live</div></div>
        <div class="panel-body">{"No engine activity detected." if not rows else rows}</div>
    </div>
    """, unsafe_allow_html=True)

def _run_graph(file_paths: List[str], objective: str, session_id: str, stop_event: threading.Event) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        graph.invoke(make_initial_state(user_query=objective, uploaded_files=file_paths, session_id=session_id), config=config)
    finally:
        stop_event.set()

def create_zip(files: List[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in files: zf.write(f, f.name)
    return buf.getvalue()

# ── Main UI ─────────────────────────────────────────────────────────
apply_custom_styles()

with st.sidebar:
    st.markdown("""<div class="sidebar-logo"><i class="ti ti-brain"></i>Multimodal DS</div>""", unsafe_allow_html=True)
    if st.button("🏠 Dashboard", width="stretch", type="primary" if st.session_state.page=="dashboard" else "secondary"):
        st.session_state.page = "dashboard"; st.rerun()
    if st.button("📊 Visualizations", width="stretch", type="primary" if st.session_state.page=="gallery" else "secondary"):
        st.session_state.page = "gallery"; st.rerun()
    
    st.markdown('<div style="font-size:10px; text-transform:uppercase; color:var(--color-text-tertiary); padding:16px 4px 8px; font-weight:600;">Recent sessions</div>', unsafe_allow_html=True)
    for sid in sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()], reverse=True)[:5]:
        if st.button(f"● {sid[:15]}...", key=f"sid_{sid}", width="stretch"):
            st.session_state.session_id = sid; st.session_state.page = "dashboard"; st.rerun()
    st.write("")
    st.caption("v1.0.0 · DeepMind Agentic Data Engine")

# Main Header
cur_sid = st.session_state.get("session_id", "No Active Session")
is_running = st.session_state.orchestrator_thread and st.session_state.orchestrator_thread.is_alive()

st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-title">{cur_sid}</div>
        <div class="status-badge {'idle' if not is_running else ''}"><div class="dot {'live' if is_running else ''}"></div>{'Running' if is_running else 'Idle'}</div>
    </div>
    <div class="topbar-actions">
        <div class="btn"><i class="ti ti-download"></i>Export</div>
        <div class="btn danger" onclick="window.location.reload()"><i class="ti ti-player-stop"></i>Stop</div>
        <div class="btn primary" onclick="window.location.reload()"><i class="ti ti-plus"></i>New run</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.page == "dashboard":
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    
    # ── Dynamic Data ──────────────────────────────────────────────
    v_count = 0
    v_files = []
    if st.session_state.session_id:
        spath = OUTPUT_DIR / st.session_state.session_id
        if spath.exists():
            v_files = sorted(list(spath.glob("*.png")))
            v_count = len(v_files)
    
    elapsed = "0s"
    if is_running and st.session_state.start_time:
        d = int(time.time() - st.session_state.start_time)
        elapsed = f"{d//60}m {d%60}s"
    elif st.session_state.session_id: elapsed = "Complete"

    # Metrics
    m_cols = st.columns(4)
    with m_cols[0]: render_metric("Agents active", "1" if is_running else "0", "↑ 1 engine live" if is_running else "Ready")
    with m_cols[1]: render_metric("Events logged", str(len(st.session_state.trace_log)), f"+{len(st.session_state.ui_adapter.drain_queue())} new" if is_running else "Log cached")
    with m_cols[2]: render_metric("Charts generated", str(v_count), "Rendering..." if is_running else "Finished")
    with m_cols[3]: render_metric("Elapsed", elapsed, "~3m remaining" if is_running else "Closed")

    # Pipeline
    render_pipeline(st.session_state.trace_log)

    # Panels
    c1, c2 = st.columns(2)
    with c1: render_trace(st.session_state.trace_log)
    with c2:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header"><div class="panel-header-title"><i class="ti ti-chart-bar"></i>Chart gallery</div><div class="chip"><i class="ti ti-eye"></i>{v_count} charts</div></div>
            <div class="panel-body">
        """, unsafe_allow_html=True)
        if v_files:
            gc = st.columns(2)
            for i, vf in enumerate(v_files[:2]):
                with gc[i]: st.image(str(vf), width="stretch", caption=vf.name)
        elif is_running:
            st.markdown("""<div class="chart-thumb"><div class="placeholder-card"><i class="ti ti-hourglass" style="font-size:24px"></i><br>Rendering...</div><div class="placeholder-card"><i class="ti ti-hourglass" style="font-size:24px"></i><br>In queue</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; padding:30px; color:var(--color-text-tertiary);'>No visualizations found for this session.</div>", unsafe_allow_html=True)
        
        if v_files:
            st.write("")
            st.download_button("📥 All charts (.zip)", data=create_zip(v_files), file_name=f"{cur_sid}_viz.zip", width="stretch")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Executive Report Snippet
    st.markdown(f"""
    <div class="panel" style="margin-top:16px">
        <div class="panel-header"><div class="panel-header-title"><i class="ti ti-file-analytics"></i>Executive report</div><div class="chip" style="opacity:0.5">{'Pending' if is_running else 'Ready'}</div></div>
        <div class="panel-body">
            <div style="padding:16px; background:var(--color-background-secondary); border-radius:var(--border-radius-md);">
                <h4 style="font-size:14px; margin-bottom:8px;">{'Report Agent Status' if is_running else 'Analysis Executive Summary'}</h4>
                <p style="font-size:12px; color:var(--color-text-secondary);">{'The engine is currently cleaning data and training models. The final report will be synthesized automatically upon completion.' if is_running else 'The analysis is complete. Detailed insights and model performance metrics have been compiled into the full report.'}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Launch Panel (Modern)
    if not st.session_state.session_id and not is_running:
        with st.container():
            st.markdown("""<div style="background:var(--color-background-secondary); padding:20px; border-radius:var(--border-radius-lg); border:1px solid var(--color-border-tertiary); margin-top:20px;">
                <h4 style="margin-top:0;">🚀 Start New Analysis Session</h4>
            </div>""", unsafe_allow_html=True)
            with st.form("new_session_form"):
                obj = st.text_input("Objective", "Explain churn drivers and predict next-month risk.")
                files = st.file_uploader("Upload Data Files", accept_multiple_files=True)
                if st.form_submit_button("Launch Analysis Engine", type="primary", width="stretch"):
                    if not files: st.error("Please upload at least one dataset."); st.stop()
                    # Save files locally
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    saved_paths = []
                    for f in files:
                        p = data_dir / f.name
                        with open(p, "wb") as wb: wb.write(f.getvalue())
                        saved_paths.append(str(p))
                    
                    new_id = f"session_{datetime.now().strftime('%y%m%d_%H%M%S')}"
                    st.session_state.session_id = new_id
                    st.session_state.trace_log = []
                    st.session_state.start_time = time.time()
                    st.session_state.stop_event.clear()
                    thread = threading.Thread(target=_run_graph, args=(saved_paths, obj, new_id, st.session_state.stop_event), daemon=True)
                    st.session_state.orchestrator_thread = thread
                    thread.start()
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── Robust Auto-Refresh ─────────────────────────────────────────────
if is_running:
    time.sleep(4)
    st.rerun()
