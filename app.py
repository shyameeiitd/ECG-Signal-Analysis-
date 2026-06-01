import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ecg_generator import *

st.set_page_config(
    page_title="ECG Atlas",
    page_icon="🫀",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0b0b0b;
}

[data-testid="stSidebar"] {
    background-color: #111111;
}

h1,h2,h3 {
    color: #00ff88;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🫀 ECG Atlas")

waveform_map = {w["name"]: w for w in WAVEFORMS}

condition = st.sidebar.selectbox(
    "Select ECG Condition",
    list(waveform_map.keys())
)

window = st.sidebar.slider(
    "Display Duration (s)",
    2,
    8,
    5
)

# --------------------------------------------------
# CURRENT ECG
# --------------------------------------------------

wf = waveform_map[condition]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🫀 ECG Atlas")

st.markdown(
    "### Interactive ECG Learning and Research Platform"
)

# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Heart Rate", wf["hr"])
c2.metric("PR Interval", wf["pr_ms"])
c3.metric("QRS Duration", wf["qrs_ms"])
c4.metric("Severity", wf["severity"])

# --------------------------------------------------
# ECG GENERATION
# --------------------------------------------------

signal = wf["fn"]()

t = np.linspace(
    0,
    DURATION,
    len(signal)
)

mask = t <= window

t = t[mask]
signal = signal[mask]

# --------------------------------------------------
# PLOTLY ECG
# --------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=signal,
        mode="lines",
        name=condition
    )
)

fig.update_layout(
    title=condition,
    template="plotly_dark",
    height=500,
    xaxis_title="Time (seconds)",
    yaxis_title="Amplitude (mV)",
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🩺 Clinical Notes",
        "📚 ECG Features",
        "🔬 Research"
    ]
)

# --------------------------------------------------
# CLINICAL TAB
# --------------------------------------------------

with tab1:

    st.subheader("Clinical Interpretation")

    st.warning(wf["severity"])

    st.write(wf["clinical"])

    st.info(wf["key"])

# --------------------------------------------------
# FEATURES TAB
# --------------------------------------------------

with tab2:

    st.subheader("Key ECG Features")

    for feat in wf["features"]:
        st.markdown(f"✅ {feat}")

# --------------------------------------------------
# RESEARCH TAB
# --------------------------------------------------

with tab3:

    st.subheader("Download ECG Data")

    df = pd.DataFrame(
        {
            "time": t,
            "amplitude": signal
        }
    )

    csv = df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        file_name=f"{condition}.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "Educational and research tool only. "
    "Not intended for clinical diagnosis."
)
