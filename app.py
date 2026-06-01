import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ecg_generator import *

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ECG Atlas Pro",
    page_icon="🫀",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color:#0b0b0b;
}

[data-testid="stSidebar"]{
    background-color:#111111;
}

.metric-card{
    background:#151515;
    padding:15px;
    border-radius:12px;
    text-align:center;
    border:1px solid #222;
}

.hero{
    padding:25px;
    border-radius:20px;
    background: linear-gradient(
        90deg,
        #0f2027,
        #203a43,
        #2c5364
    );
    text-align:center;
}

.hero h1{
    color:white;
}

.hero p{
    color:#dddddd;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🫀 ECG Atlas Pro</h1>
<p>
Interactive Cardiology Learning & Research Platform
</p>
<p>
15 Cardiac Conditions • ECG Education • Dataset Generation
</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🫀 ECG Atlas")

st.sidebar.success("15 ECG Conditions")

waveform_map = {
    w["name"]: w
    for w in WAVEFORMS
}

condition = st.sidebar.selectbox(
    "Select ECG Condition",
    list(waveform_map.keys())
)

window = st.sidebar.slider(
    "Display Duration (seconds)",
    2,
    8,
    5
)

show_grid = st.sidebar.checkbox(
    "ECG Grid",
    value=True
)

# ---------------------------------------------------
# CURRENT WAVEFORM
# ---------------------------------------------------

wf = waveform_map[condition]

severity_icon = {
    "NORMAL":"🟢",
    "MILD":"🔵",
    "WARNING":"🟠",
    "ARRHYTHMIA":"🟣",
    "CRITICAL":"🔴",
    "FATAL":"⚫"
}

# ---------------------------------------------------
# TOP METRICS
# ---------------------------------------------------

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Heart Rate",
    wf["hr"]
)

c2.metric(
    "PR Interval",
    wf["pr_ms"]
)

c3.metric(
    "QRS Duration",
    wf["qrs_ms"]
)

c4.metric(
    "Severity",
    wf["severity"]
)

st.markdown(
    f"## {severity_icon.get(wf['severity'],'⚪')} {wf['severity']}"
)

# ---------------------------------------------------
# ECG SIGNAL
# ---------------------------------------------------

signal = wf["fn"]()

t = np.linspace(
    0,
    DURATION,
    len(signal)
)

mask = t <= window

t = t[mask]
signal = signal[mask]

# ---------------------------------------------------
# ECG PLOT
# ---------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=signal,
        mode="lines",
        line=dict(
            color="#00ff88",
            width=2
        ),
        name=condition
    )
)

fig.update_layout(
    title=condition,
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#071207",
    height=550,
    xaxis_title="Time (seconds)",
    yaxis_title="Amplitude (mV)",
    showlegend=False
)

if show_grid:

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#123412"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#123412"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1,tab2,tab3,tab4 = st.tabs(
[
    "🩺 Clinical Notes",
    "📚 ECG Features",
    "⚖️ Compare",
    "🔬 Research"
]
)

# ---------------------------------------------------
# CLINICAL
# ---------------------------------------------------

with tab1:

    st.subheader("Clinical Interpretation")

    st.warning(wf["severity"])

    st.write(wf["clinical"])

    st.info(wf["key"])

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

with tab2:

    st.subheader("Key ECG Features")

    for feat in wf["features"]:

        st.markdown(
            f"✅ {feat}"
        )

# ---------------------------------------------------
# COMPARE
# ---------------------------------------------------

with tab3:

    st.subheader("Compare Two Conditions")

    compare_condition = st.selectbox(
        "Comparison ECG",
        list(waveform_map.keys()),
        index=1
    )

    wf2 = waveform_map[compare_condition]

    sig1 = wf["fn"]()
    sig2 = wf2["fn"]()

    t2 = np.linspace(
        0,
        DURATION,
        len(sig1)
    )

    fig_compare = go.Figure()

    fig_compare.add_trace(
        go.Scatter(
            x=t2,
            y=sig1,
            mode="lines",
            name=wf["name"]
        )
    )

    fig_compare.add_trace(
        go.Scatter(
            x=t2,
            y=sig2,
            mode="lines",
            name=wf2["name"]
        )
    )

    fig_compare.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

# ---------------------------------------------------
# RESEARCH
# ---------------------------------------------------

with tab4:

    st.subheader("Research Dataset Export")

    df = pd.DataFrame(
        {
            "time":t,
            "amplitude":signal
        }
    )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download CSV",
        csv,
        file_name=f"{condition}.csv",
        mime="text/csv"
    )

    st.success(
        "Export ECG signal for AI/ML research."
    )

# ---------------------------------------------------
# ECG ATLAS
# ---------------------------------------------------

st.divider()

st.subheader("🫀 ECG Disease Atlas")

cols = st.columns(3)

for i,w in enumerate(WAVEFORMS):

    with cols[i % 3]:

        st.success(w["name"])

        st.caption(
            f"{w['category']} • {w['severity']}"
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
"""
Educational and research tool only.

Not intended for clinical diagnosis,
treatment or medical decision making.
"""
)
