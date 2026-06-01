# ============================================================
# 🫀 ECG Atlas Clinical — Premium Medical Software UI
# ============================================================
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
import warnings, io, json
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Atlas Clinical",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0C10 !important;
    color: #D8DCE6 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #0A0C10 !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D0F15 !important;
    border-right: 1px solid #1E2130 !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Radio buttons → Nav ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 0 !important;
    cursor: pointer !important;
    font-size: 0.82rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
    color: #707890 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #C8D0E0 !important;
    background: #141720 !important;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div + div,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    color: #00E676 !important;
    background: #0D1A14 !important;
    border-left: 2px solid #00E676 !important;
}
/* Hide radio circles */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { display: none !important; }

/* ── Selectbox / Dropdown ── */
.stSelectbox > div > div {
    background: #141720 !important;
    border: 1px solid #1E2334 !important;
    border-radius: 4px !important;
    color: #D8DCE6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div:hover {
    border-color: #00E676 !important;
}
[data-testid="stSelectboxVirtualDropdown"] {
    background: #141720 !important;
    border: 1px solid #1E2334 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #141720 !important;
    color: #00E676 !important;
    border: 1px solid #00E67640 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #00E67615 !important;
    border-color: #00E676 !important;
    box-shadow: 0 0 12px #00E67620 !important;
}

/* ── Toggle / Checkbox ── */
.stCheckbox label {
    color: #B0B8C8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #0D1A14 !important;
    color: #00E676 !important;
    border: 1px solid #00E67650 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #141720 !important;
    border: 1px solid #1E2334 !important;
    border-radius: 4px !important;
    color: #B0B8C8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: #111420 !important;
    border: 1px solid #1E2334 !important;
    border-top: none !important;
    border-radius: 0 0 4px 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E2334 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #606880 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: #00E676 !important;
    border-bottom: 2px solid #00E676 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.2rem 0 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0 !important; }
.stSlider [data-testid="stThumbValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #00E676 !important;
}

/* ── Metric ── */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #00E676 !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #707890 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0A0C10; }
::-webkit-scrollbar-thumb { background: #1E2334; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2E3350; }

/* ── Divider ── */
hr { border-color: #1E2334 !important; margin: 0.8rem 0 !important; }

/* ── Custom Plotly toolbar ── */
.js-plotly-plot .plotly .modebar {
    background: #141720 !important;
    border: 1px solid #1E2334 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ECG BACKEND (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────
FS = 500
DURATION = 8

def bandpass(sig, lo=0.5, hi=40):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype='band')
    return filtfilt(b, a, sig)

def add_noise(sig, snr=30):
    p = np.mean(sig**2)
    n = p / 10**(snr/10)
    return sig + np.random.normal(0, np.sqrt(n), len(sig))

def gauss(t, mu, sigma, amp):
    return amp * np.exp(-((t - mu)**2) / (2 * sigma**2))

def gen_normal(hr=72):
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0 / hr
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS * rr))
        beat = gauss(bt,0.12,0.022,0.15)+gauss(bt,0.22,0.008,-0.09)+gauss(bt,0.265,0.009,1.20)+gauss(bt,0.305,0.008,-0.30)+gauss(bt,0.50,0.055,0.28)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_bradycardia(hr=42):
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0 / hr
    for bs in np.arange(0.4, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS * rr))
        beat = gauss(bt,0.14,0.025,0.16)+gauss(bt,0.24,0.009,-0.10)+gauss(bt,0.28,0.009,1.10)+gauss(bt,0.32,0.008,-0.28)+gauss(bt,0.56,0.060,0.26)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_tachycardia(hr=148):
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0 / hr
    for bs in np.arange(0.1, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS * rr))
        beat = gauss(bt,0.08,0.018,0.17)+gauss(bt,0.15,0.007,-0.10)+gauss(bt,0.19,0.008,1.15)+gauss(bt,0.23,0.007,-0.32)+gauss(bt,0.38,0.038,0.24)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_afib():
    N = FS * DURATION; t_arr = np.linspace(0, DURATION, N); sig = np.zeros(N)
    for f in [4.2,5.8,7.1,9.3,11.5]: sig += 0.045*np.sin(2*np.pi*f*t_arr+np.random.uniform(0,2*np.pi))
    mean_rr = 60.0/110; cur = 0.3
    while cur < DURATION - 0.5:
        rr = mean_rr*np.random.uniform(0.55,1.7)
        bt = np.linspace(0,0.5,int(FS*0.5)); ph = bt/0.5
        beat = gauss(ph,0.28,0.008,-0.10)+gauss(ph,0.32,0.009,1.05)+gauss(ph,0.36,0.008,-0.27)+gauss(ph,0.65,0.050,0.20)
        idx = int(cur*FS)
        if idx+len(bt)<=N: sig[idx:idx+len(bt)] += beat
        cur += rr
    return add_noise(bandpass(sig))

def gen_flutter():
    N = FS * DURATION; t_arr = np.linspace(0, DURATION, N); sig = np.zeros(N)
    for harm in [1,2,3]: sig += (0.18/harm)*np.sin(2*np.pi*5*harm*t_arr)
    rr = 60.0/150
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        beat = gauss(ph,0.22,0.008,-0.08)+gauss(ph,0.26,0.008,1.0)+gauss(ph,0.30,0.007,-0.24)+gauss(ph,0.55,0.050,0.22)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_av_block_1():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/65
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        beat = gauss(bt,0.10,0.022,0.15)+gauss(bt,0.30,0.008,-0.09)+gauss(bt,0.34,0.009,1.18)+gauss(bt,0.38,0.008,-0.28)+gauss(bt,0.60,0.055,0.26)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_av_block_2():
    N = FS * DURATION; sig = np.zeros(N); short_rr = 60.0/75; long_rr = short_rr*2.2; cur = 0.3; beat_count = 0
    while cur < DURATION - long_rr:
        bt_len = long_rr if beat_count%3==2 else short_rr
        bt = np.linspace(0,bt_len,int(FS*bt_len)); ph = bt/bt_len
        if beat_count%3 != 2:
            beat = gauss(ph,0.12,0.022,0.15)+gauss(ph,0.22,0.008,-0.09)+gauss(ph,0.265,0.009,1.15)+gauss(ph,0.305,0.008,-0.28)+gauss(ph,0.50,0.055,0.26)
        else:
            beat = gauss(ph,0.08,0.020,0.14)
        i = int(cur*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
        cur += bt_len; beat_count += 1
    return add_noise(bandpass(sig))

def gen_stemi():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/88
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        p  = gauss(ph,0.12,0.022,0.13); q = gauss(ph,0.22,0.008,-0.22); r = gauss(ph,0.26,0.009,0.90); s = gauss(ph,0.30,0.007,-0.10)
        st_mask = (ph>=0.31)&(ph<=0.55)
        st = np.where(st_mask,0.40*np.exp(-((ph-0.33)**2)/(2*0.07**2)),0)
        tw = gauss(ph,0.65,0.042,0.48)
        beat = p+q+r+s+st+tw
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_nstemi():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/82
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        p = gauss(ph,0.12,0.022,0.14); q = gauss(ph,0.22,0.009,-0.12); r = gauss(ph,0.265,0.009,1.0); s = gauss(ph,0.305,0.008,-0.22)
        st_dep = np.where((ph>=0.31)&(ph<=0.50),-0.18*np.exp(-((ph-0.35)**2)/(2*0.06**2)),0)
        tw = gauss(ph,0.58,0.045,-0.30)
        beat = p+q+r+s+st_dep+tw
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_vtach():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/170
    for bs in np.arange(0.15, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        main = 1.20*np.exp(-((ph-0.22)**2)/(2*0.045**2))-0.55*np.exp(-((ph-0.38)**2)/(2*0.032**2))+0.25*np.exp(-((ph-0.50)**2)/(2*0.025**2))
        tw = -0.38*np.exp(-((ph-0.72)**2)/(2*0.042**2))
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += main+tw
    return add_noise(bandpass(sig),snr=22)

def gen_vfib():
    N = FS * DURATION; t_arr = np.linspace(0,DURATION,N); sig = np.zeros(N)
    for f in [2.1,3.4,5.2,7.8,11.3,15.6]:
        sig += np.random.uniform(0.25,0.70)*np.sin(2*np.pi*f*t_arr+np.random.uniform(0,2*np.pi))
    sig *= (0.6+0.4*np.sin(2*np.pi*1.1*t_arr))
    return add_noise(bandpass(sig,lo=1,hi=35),snr=18)

def gen_lbbb():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/70
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        beat = gauss(ph,0.12,0.024,0.16)+gauss(ph,0.24,0.014,0.65)+gauss(ph,0.32,0.014,0.90)+gauss(ph,0.40,0.010,-0.20)+gauss(ph,0.62,0.060,-0.32)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_rbbb():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/72
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        beat = gauss(ph,0.12,0.022,0.15)+gauss(ph,0.21,0.008,-0.08)+gauss(ph,0.25,0.010,1.10)+gauss(ph,0.31,0.012,-0.45)+gauss(ph,0.37,0.013,0.55)+gauss(ph,0.62,0.055,-0.28)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_wpw():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/80
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        delta = 0.35*np.where((ph>=0.17)&(ph<=0.23),np.sin(np.pi*(ph-0.17)/0.06),0)
        beat = gauss(ph,0.10,0.020,0.16)+delta+gauss(ph,0.24,0.010,1.05)+gauss(ph,0.28,0.009,-0.22)+gauss(ph,0.52,0.055,0.25)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

def gen_long_qt():
    N = FS * DURATION; sig = np.zeros(N); rr = 60.0/65
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr)); ph = bt/rr
        beat = gauss(ph,0.12,0.022,0.15)+gauss(ph,0.22,0.008,-0.09)+gauss(ph,0.265,0.009,1.18)+gauss(ph,0.305,0.008,-0.28)+gauss(ph,0.76,0.075,0.30)+gauss(ph,0.88,0.030,0.12)
        i = int(bs*FS)
        if i+len(bt)<=N: sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ─────────────────────────────────────────────────────────────
# WAVEFORM REGISTRY (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────
WAVEFORMS = [
    {"name":"Normal Sinus Rhythm","fn":gen_normal,"category":"Normal","cat_color":"#2ecc71","hr":"60–100","severity":"NORMAL","sev_color":"#00E676","pr_ms":"120–200","qrs_ms":"< 120","key":"Regular P before every QRS. Uniform P-P and R-R intervals.","clinical":"Healthy cardiac conduction. No treatment needed.","features":["Upright P wave in II","PR 120–200 ms","Narrow QRS < 120 ms","T wave concordant"]},
    {"name":"Sinus Bradycardia","fn":gen_bradycardia,"category":"Sinus","cat_color":"#3498db","hr":"< 60","severity":"MILD","sev_color":"#3498db","pr_ms":"120–200","qrs_ms":"< 120","key":"Normal morphology, rate < 60 bpm. Long R-R intervals.","clinical":"Common in athletes. Treat only if symptomatic (dizziness, syncope).","features":["Rate < 60 bpm","Normal P-QRS-T","Prolonged R-R","Possible escape beats"]},
    {"name":"Sinus Tachycardia","fn":gen_tachycardia,"category":"Sinus","cat_color":"#3498db","hr":"> 100","severity":"MILD","sev_color":"#3498db","pr_ms":"< 160","qrs_ms":"< 120","key":"Rate > 100 bpm. P waves may merge with preceding T waves.","clinical":"Physiological response (fever, pain, anxiety). Treat the cause.","features":["Rate > 100 bpm","P-T merging at high rates","Short R-R","Narrow QRS"]},
    {"name":"Atrial Fibrillation","fn":gen_afib,"category":"Atrial","cat_color":"#9b59b6","hr":"60–180 (irregular)","severity":"ARRHYTHMIA","sev_color":"#9b59b6","pr_ms":"Absent","qrs_ms":"< 120","key":"Irregularly irregular rhythm. No visible P waves. Fibrillatory baseline.","clinical":"Most common arrhythmia. Risk of stroke — anticoagulate. Rate/rhythm control.","features":["No P waves","Irregularly irregular RR","Fibrillatory baseline","Narrow QRS (unless aberrant)"]},
    {"name":"Atrial Flutter","fn":gen_flutter,"category":"Atrial","cat_color":"#9b59b6","hr":"150 (2:1 block)","severity":"ARRHYTHMIA","sev_color":"#9b59b6","pr_ms":"Absent","qrs_ms":"< 120","key":"Sawtooth flutter waves at 300 bpm. Regular ventricular rate (2:1 or 4:1).","clinical":"Cardioversion or ablation. Anticoagulation as in AF.","features":["Sawtooth F-waves 300 bpm","2:1 or 4:1 conduction","Regular ventricular rate","No isoelectric baseline"]},
    {"name":"1° AV Block","fn":gen_av_block_1,"category":"Conduction","cat_color":"#e67e22","hr":"60–100","severity":"WARNING","sev_color":"#F39C12","pr_ms":"> 200","qrs_ms":"< 120","key":"Prolonged PR interval (> 200 ms). Every P conducts to QRS.","clinical":"Usually benign. No specific treatment unless symptomatic.","features":["PR > 200 ms (constant)","Every P followed by QRS","Normal QRS morphology","Slowed AV node conduction"]},
    {"name":"2° AV Block (Mobitz II)","fn":gen_av_block_2,"category":"Conduction","cat_color":"#e67e22","hr":"45–60","severity":"WARNING","sev_color":"#F39C12","pr_ms":"Constant then dropped","qrs_ms":"± wide","key":"Constant PR then sudden non-conducted P wave. High risk of progression.","clinical":"High risk — may need pacemaker. Does not improve with atropine.","features":["Constant PR then dropped QRS","Infra-Hisian block","2:1 or 3:1 pattern","Pacemaker often indicated"]},
    {"name":"STEMI","fn":gen_stemi,"category":"Ischemia","cat_color":"#e74c3c","hr":"60–100","severity":"CRITICAL","sev_color":"#E74C3C","pr_ms":"120–200","qrs_ms":"≥ 100","key":"ST elevation ≥ 1 mm in ≥ 2 contiguous leads. Pathological Q waves.","clinical":"MEDICAL EMERGENCY. Activate cath lab. PCI within 90 min. Aspirin + heparin.","features":["ST elevation ≥ 1 mm","Pathological Q waves","Hyperacute T waves","Reciprocal ST depression"]},
    {"name":"NSTEMI / Ischaemia","fn":gen_nstemi,"category":"Ischemia","cat_color":"#e74c3c","hr":"70–100","severity":"CRITICAL","sev_color":"#E74C3C","pr_ms":"120–200","qrs_ms":"< 120","key":"ST depression + T-wave inversion without ST elevation. Troponin rise.","clinical":"Urgent: anticoagulate, antiplatelets. Angiography within 24–72 h.","features":["ST depression ≥ 0.5 mm","T-wave inversion","No ST elevation","Troponin elevation"]},
    {"name":"Ventricular Tachycardia","fn":gen_vtach,"category":"Ventricular","cat_color":"#c0392b","hr":"150–250","severity":"CRITICAL","sev_color":"#E74C3C","pr_ms":"Absent","qrs_ms":"> 120","key":"Wide complex tachycardia, QRS > 120 ms. AV dissociation. No P waves.","clinical":"EMERGENCY. Unstable → DC cardioversion. Stable → amiodarone IV.","features":["Rate > 150 bpm","QRS > 120 ms (bizarre)","AV dissociation","Discordant T waves"]},
    {"name":"Ventricular Fibrillation","fn":gen_vfib,"category":"Ventricular","cat_color":"#c0392b","hr":"N/A","severity":"FATAL","sev_color":"#C0392B","pr_ms":"None","qrs_ms":"None","key":"Completely chaotic baseline. No organised complexes. No cardiac output.","clinical":"CARDIAC ARREST. Immediate CPR + defibrillation. Adrenaline 1mg IV.","features":["No organised complexes","Chaotic baseline","No cardiac output","Immediate defibrillation needed"]},
    {"name":"Left BBB","fn":gen_lbbb,"category":"Bundle Branch","cat_color":"#1abc9c","hr":"60–100","severity":"WARNING","sev_color":"#F39C12","pr_ms":"Normal","qrs_ms":"> 120","key":"QRS > 120 ms. Broad notched (M-shaped) R in I, aVL, V5–V6. WiLLiaM pattern.","clinical":"New LBBB with chest pain = STEMI equivalent. Investigate for structural disease.","features":["QRS > 120 ms","Notched R in V5-V6 (M-shape)","Discordant ST/T","No septal Q in I, V5, V6"]},
    {"name":"Right BBB","fn":gen_rbbb,"category":"Bundle Branch","cat_color":"#1abc9c","hr":"60–100","severity":"MILD","sev_color":"#3498db","pr_ms":"Normal","qrs_ms":"> 120","key":"QRS > 120 ms. rsR' (M) pattern in V1. Deep slurred S in I, V6. MaRRoW pattern.","clinical":"May be normal variant. Investigate if new. Right heart disease workup.","features":["QRS > 120 ms","RSR' in V1 (rabbit ears)","Deep S in I, V6","Discordant T in V1-V2"]},
    {"name":"Wolff-Parkinson-White","fn":gen_wpw,"category":"Pre-excitation","cat_color":"#f39c12","hr":"60–100 (SVT burst)","severity":"WARNING","sev_color":"#F39C12","pr_ms":"< 120 (short)","qrs_ms":"Broad (slurred)","key":"Short PR < 120 ms. Delta wave. Broad QRS. Accessory conduction pathway.","clinical":"Risk of SVT and AF with rapid conduction. Catheter ablation curative.","features":["Short PR < 120 ms","Delta wave (slurred upstroke)","Broad QRS","Pseudo ST/T changes"]},
    {"name":"Long QT Syndrome","fn":gen_long_qt,"category":"Channelopathy","cat_color":"#e74c3c","hr":"60–80","severity":"WARNING","sev_color":"#F39C12","pr_ms":"Normal","qrs_ms":"Normal","key":"QTc > 500 ms. Prominent U wave. Risk of torsades de pointes → VF.","clinical":"Avoid QT-prolonging drugs. Beta-blockers. ICD if high risk. Genetic testing.","features":["QTc > 500 ms","Prominent U wave","T-wave morphology changes","Risk of Torsades de Pointes"]},
]

CONDITION_NAMES = [w["name"] for w in WAVEFORMS]
WAVEFORM_MAP   = {w["name"]: w for w in WAVEFORMS}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "page"        not in st.session_state: st.session_state.page = "ECG Viewer"
if "favorites"   not in st.session_state: st.session_state.favorites = []
if "recent"      not in st.session_state: st.session_state.recent = []
if "ecg_mode"    not in st.session_state: st.session_state.ecg_mode = "Monitor"
if "selected"    not in st.session_state: st.session_state.selected = "Normal Sinus Rhythm"
if "cmp_left"    not in st.session_state: st.session_state.cmp_left  = "Normal Sinus Rhythm"
if "cmp_right"   not in st.session_state: st.session_state.cmp_right = "Atrial Fibrillation"

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
SEV_COLORS = {
    "NORMAL":    "#00E676",
    "MILD":      "#3498DB",
    "WARNING":   "#F39C12",
    "ARRHYTHMIA":"#9B59B6",
    "CRITICAL":  "#E74C3C",
    "FATAL":     "#C0392B",
}

def sev_color(sev): return SEV_COLORS.get(sev, "#888")

def add_to_recent(name):
    r = st.session_state.recent
    if name in r: r.remove(name)
    r.insert(0, name)
    st.session_state.recent = r[:5]

def build_ecg_fig(sig, mode="Monitor", window=6.0, title=""):
    N  = len(sig)
    t  = np.linspace(0, DURATION, N)
    m  = t <= window
    tw, sw = t[m], sig[m]

    if mode == "Monitor":
        bg, grid_major, grid_minor, wave_color, paper = "#050D05", "#0A2A0A", "#061506", "#00E676", False
    else:
        bg, grid_major, grid_minor, wave_color, paper = "#FEFAF5", "#FFCCCC", "#FFE8E8", "#CC0000", True

    fig = go.Figure()

    # Grid lines
    for gx in np.arange(0, window+0.2, 0.2):
        fig.add_shape(type="line", x0=gx, x1=gx, y0=-1.8, y1=2.0,
                      line=dict(color=grid_minor, width=0.5))
    for gx in np.arange(0, window+1, 1.0):
        fig.add_shape(type="line", x0=gx, x1=gx, y0=-1.8, y1=2.0,
                      line=dict(color=grid_major, width=1.0))
    for gy in np.arange(-2, 2.2, 0.1):
        fig.add_shape(type="line", x0=0, x1=window, y0=gy, y1=gy,
                      line=dict(color=grid_minor, width=0.4))
    for gy in np.arange(-2, 2.2, 0.5):
        fig.add_shape(type="line", x0=0, x1=window, y0=gy, y1=gy,
                      line=dict(color=grid_major, width=0.9))

    # Baseline
    fig.add_shape(type="line", x0=0, x1=window, y0=0, y1=0,
                  line=dict(color=grid_major, width=1.2))

    # Glow (monitor only)
    if not paper:
        fig.add_trace(go.Scatter(x=tw, y=sw, mode="lines",
            line=dict(color=wave_color, width=8),
            opacity=0.07, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=tw, y=sw, mode="lines",
            line=dict(color=wave_color, width=3),
            opacity=0.25, hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Scatter(
        x=tw, y=sw, mode="lines",
        line=dict(color=wave_color, width=1.4),
        opacity=0.95,
        hovertemplate="<b>t:</b> %{x:.3f}s<br><b>mV:</b> %{y:.3f}<extra></extra>",
        name="ECG",
        showlegend=False,
    ))

    tc = "#303030" if paper else "#555"
    fig.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            range=[0, window], title="Time (s)", titlefont=dict(size=10, color=tc),
            tickfont=dict(size=9, color=tc, family="IBM Plex Mono"),
            gridcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False,
            tickmode="linear", dtick=1.0,
        ),
        yaxis=dict(
            range=[-1.6, 1.9], title="Amplitude (mV)", titlefont=dict(size=10, color=tc),
            tickfont=dict(size=9, color=tc, family="IBM Plex Mono"),
            gridcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False,
            tickmode="linear", dtick=0.5,
        ),
        height=320,
        dragmode="pan",
        font=dict(family="IBM Plex Mono"),
        modebar=dict(
            bgcolor="#141720", color="#555", activecolor="#00E676",
            orientation="v",
        ),
    )
    return fig

def build_anatomy_fig():
    sig = gen_normal(hr=65)
    t   = np.linspace(0, DURATION, len(sig))
    s, e = int(0.8*FS), int(2.2*FS)
    tw = t[s:e]; sw = sig[s:e]; tw = tw - tw[0]

    fig = go.Figure()
    # Grid
    for gx in np.arange(0, 1.4, 0.04):
        fig.add_shape(type="line", x0=gx, x1=gx, y0=-0.8, y1=1.6,
                      line=dict(color="#061506", width=0.4))
    for gx in np.arange(0, 1.4, 0.2):
        fig.add_shape(type="line", x0=gx, x1=gx, y0=-0.8, y1=1.6,
                      line=dict(color="#0A2A0A", width=0.9))
    for gy in np.arange(-0.5, 1.6, 0.1):
        fig.add_shape(type="line", x0=0, x1=1.35, y0=gy, y1=gy,
                      line=dict(color="#061506", width=0.4))
    for gy in np.arange(-0.5, 1.6, 0.5):
        fig.add_shape(type="line", x0=0, x1=1.35, y0=gy, y1=gy,
                      line=dict(color="#0A2A0A", width=0.9))
    fig.add_shape(type="line", x0=0, x1=1.35, y0=0, y1=0,
                  line=dict(color="#1a5c1a", width=1.2))

    # Glow
    fig.add_trace(go.Scatter(x=tw, y=sw, mode="lines",
        line=dict(color="#00E676", width=8), opacity=0.07,
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=tw, y=sw, mode="lines",
        line=dict(color="#00E676", width=1.6), opacity=0.9,
        hovertemplate="<b>t:</b> %{x:.3f}s  <b>mV:</b> %{y:.3f}<extra></extra>",
        name="ECG", showlegend=False))

    labels = [
        (0.12, 0.15, "P", 0.05, 0.35, "#FFD700"),
        (0.22, -0.09, "Q", -0.02, -0.30, "#FF8C00"),
        (0.265, 1.20, "R", 0.32, 1.38, "#00FF88"),
        (0.305, -0.30, "S", 0.37, -0.46, "#FF8C00"),
        (0.50, 0.28, "T", 0.57, 0.44, "#FFD700"),
    ]
    for (px, py, lbl, ax, ay, col) in labels:
        fig.add_annotation(
            x=px, y=py, ax=ax, ay=ay,
            text=f"<b>{lbl}</b>",
            font=dict(size=13, color=col, family="IBM Plex Mono"),
            arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor=col,
            showarrow=True,
        )

    # Interval brackets
    def bracket(x1, x2, y, label, color):
        fig.add_shape(type="line", x0=x1, x1=x2, y0=y, y1=y,
                      line=dict(color=color, width=1.5, dash="dot"))
        fig.add_shape(type="line", x0=x1, x1=x1, y0=y-0.04, y1=y+0.04,
                      line=dict(color=color, width=1.5))
        fig.add_shape(type="line", x0=x2, x1=x2, y0=y-0.04, y1=y+0.04,
                      line=dict(color=color, width=1.5))
        fig.add_annotation(x=(x1+x2)/2, y=y-0.09, text=label, showarrow=False,
                           font=dict(size=9, color=color, family="IBM Plex Mono"),
                           bgcolor="#0A0C10", borderpad=2)

    bracket(0.08, 0.23, -0.42, "PR  ~160 ms", "#88CCFF")
    bracket(0.21, 0.31, -0.57, "QRS ~80 ms",  "#FFAA44")
    bracket(0.21, 0.62, -0.72, "QT  ~380 ms", "#FF88AA")

    fig.add_annotation(x=0.41, y=0.06, text="ST segment", showarrow=False,
                       font=dict(size=9, color="#88FF88", family="IBM Plex Mono"),
                       bgcolor="#071207", borderpad=2)

    fig.update_layout(
        paper_bgcolor="#050D05", plot_bgcolor="#050D05",
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(range=[0, 1.35], title="Time (s)",
                   titlefont=dict(size=10, color="#555"),
                   tickfont=dict(size=9, color="#555", family="IBM Plex Mono"),
                   showgrid=False, zeroline=False, tickmode="linear", dtick=0.2),
        yaxis=dict(range=[-0.82, 1.62], title="Amplitude (mV)",
                   titlefont=dict(size=10, color="#555"),
                   tickfont=dict(size=9, color="#555", family="IBM Plex Mono"),
                   showgrid=False, zeroline=False, tickmode="linear", dtick=0.5),
        height=380,
        font=dict(family="IBM Plex Mono"),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:1.6rem 1.2rem 0.8rem 1.2rem; border-bottom:1px solid #1E2130;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.15rem; font-weight:600;
                    color:#00E676; letter-spacing:0.02em; line-height:1.3;">
            🫀 ECG Atlas
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:400;
                    color:#3A4560; letter-spacing:0.12em; text-transform:uppercase;
                    margin-top:0.25rem;">
            Clinical Edition
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.8rem 1.2rem 0.4rem; font-family:'IBM Plex Mono',monospace;
                font-size:0.62rem; color:#3A4560; letter-spacing:0.14em; text-transform:uppercase;">
        Navigation
    </div>
    """, unsafe_allow_html=True)

    NAV_OPTIONS = [
        "⬡  ECG Viewer",
        "⬡  Disease Atlas",
        "⬡  ECG Anatomy",
        "⬡  Compare ECGs",
        "⬡  Research Export",
        "⬡  About",
    ]
    nav = st.radio("nav", NAV_OPTIONS, label_visibility="collapsed")
    st.session_state.page = nav.replace("⬡  ", "")

    st.markdown("""<div style="border-top:1px solid #1E2130; margin:0.8rem 0;"></div>""",
                unsafe_allow_html=True)

    # Favorites
    if st.session_state.favorites:
        st.markdown("""
        <div style="padding:0.4rem 1.2rem 0.3rem; font-family:'IBM Plex Mono',monospace;
                    font-size:0.62rem; color:#3A4560; letter-spacing:0.14em; text-transform:uppercase;">
            Favorites
        </div>
        """, unsafe_allow_html=True)
        for fav in st.session_state.favorites[:4]:
            w = WAVEFORM_MAP[fav]
            st.markdown(f"""
            <div style="padding:0.3rem 1.2rem; font-family:'IBM Plex Mono',monospace;
                        font-size:0.75rem; color:#507060; cursor:pointer;">
                <span style="color:{w['sev_color']};">●</span>  {fav}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""<div style="border-top:1px solid #1E2130; margin:0.6rem 0;"></div>""",
                    unsafe_allow_html=True)

    # Recent
    if st.session_state.recent:
        st.markdown("""
        <div style="padding:0.4rem 1.2rem 0.3rem; font-family:'IBM Plex Mono',monospace;
                    font-size:0.62rem; color:#3A4560; letter-spacing:0.14em; text-transform:uppercase;">
            Recent
        </div>
        """, unsafe_allow_html=True)
        for rec in st.session_state.recent[:3]:
            w = WAVEFORM_MAP[rec]
            st.markdown(f"""
            <div style="padding:0.3rem 1.2rem; font-family:'IBM Plex Mono',monospace;
                        font-size:0.72rem; color:#404858;">
                <span style="color:{w['sev_color']}80;">◆</span>  {rec}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute; bottom:0; left:0; right:0;
                padding:0.8rem 1.2rem; border-top:1px solid #1E2130;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                    color:#2A3040; letter-spacing:0.08em;">
            v2.0.0  ·  15 Conditions  ·  500 Hz
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="display:flex; align-items:flex-end; justify-content:space-between;
                margin-bottom:1.4rem; padding-bottom:0.8rem;
                border-bottom:1px solid #1A1D26;">
        <div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                        color:#3A4560; letter-spacing:0.16em; text-transform:uppercase;
                        margin-bottom:0.3rem;">
                ECG Atlas Clinical  /  {title}
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1.4rem;
                        font-weight:600; color:#C8D0E0; letter-spacing:-0.01em;">
                {title}
            </div>
            {"<div style='font-family:IBM Plex Sans,sans-serif; font-size:0.8rem; color:#505870; margin-top:0.2rem;'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#303848; text-align:right;">
            Lead II · 25 mm/s · 10 mm/mV
        </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_cards(w):
    sc = sev_color(w["severity"])
    cols = st.columns(4)
    items = [
        ("HEART RATE", w["hr"], "bpm", "#00E676"),
        ("PR INTERVAL", w["pr_ms"], "ms", "#3498DB"),
        ("QRS DURATION", w["qrs_ms"], "ms", "#F39C12"),
        ("SEVERITY", w["severity"], "", sc),
    ]
    for col, (label, val, unit, color) in zip(cols, items):
        col.markdown(f"""
        <div style="background:#10131A; border:1px solid #1A1D26; border-top:2px solid {color};
                    border-radius:4px; padding:0.9rem 1.1rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                        color:#404858; letter-spacing:0.14em; text-transform:uppercase;
                        margin-bottom:0.4rem;">{label}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:1.25rem;
                        font-weight:600; color:{color}; letter-spacing:-0.02em;">
                {val}
                <span style="font-size:0.7rem; font-weight:400; color:#404858; margin-left:0.2rem;">{unit}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def info_card(w):
    sc = sev_color(w["severity"])
    st.markdown(f"""
    <div style="background:#10131A; border:1px solid #1A1D26; border-radius:4px;
                padding:1rem 1.2rem; margin-bottom:0.8rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:0.8rem; padding-bottom:0.6rem; border-bottom:1px solid #1A1D26;">
            <div>
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1rem;
                            font-weight:600; color:#C8D0E0;">{w["name"]}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                            color:#505870; margin-top:0.2rem;">{w["category"]}</div>
            </div>
            <div style="background:{sc}20; border:1px solid {sc}60; border-radius:3px;
                        padding:0.25rem 0.65rem; font-family:'IBM Plex Mono',monospace;
                        font-size:0.7rem; font-weight:600; color:{sc}; letter-spacing:0.08em;">
                {w["severity"]}
            </div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                    color:#505870; font-style:italic; line-height:1.6;">
            {w["key"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

def feature_cards(w):
    sc = sev_color(w["severity"])
    st.markdown(f"""
    <div style="background:#10131A; border:1px solid #1A1D26; border-radius:4px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3A4560;
                    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.7rem;">
            Key ECG Features
        </div>
        {"".join([f'<div style="padding:0.3rem 0; border-bottom:1px solid #141720; font-family:IBM Plex Mono,monospace; font-size:0.8rem; color:#909CB0;"><span style="color:{sc}; margin-right:0.5rem;">▸</span>{feat}</div>' for feat in w["features"]])}
    </div>
    """, unsafe_allow_html=True)

def clinical_card(w):
    sc = sev_color(w["severity"])
    st.markdown(f"""
    <div style="background:#10131A; border:1px solid #1A1D26; border-left:3px solid {sc};
                border-radius:4px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3A4560;
                    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.6rem;">
            Clinical Action
        </div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem;
                    color:#A0AAB8; line-height:1.7;">
            {w["clinical"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: ECG VIEWER
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "ECG Viewer":
    page_header("ECG Viewer", "Interactive ECG Learning, Analysis and Research Platform")

    # Search + select
    col_sel, col_fav, col_mode, col_window = st.columns([3, 1, 1.5, 2])

    with col_sel:
        search = st.text_input("", placeholder="Search condition…", label_visibility="collapsed", key="search_input")
        filtered = [n for n in CONDITION_NAMES if search.lower() in n.lower()] if search else CONDITION_NAMES
        if filtered:
            sel = st.selectbox("Condition", filtered, index=filtered.index(st.session_state.selected) if st.session_state.selected in filtered else 0, label_visibility="collapsed")
            st.session_state.selected = sel
        else:
            st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#E74C3C;padding:0.5rem;'>No match found</div>", unsafe_allow_html=True)
            sel = st.session_state.selected

    with col_fav:
        favs = st.session_state.favorites
        fav_label = "★ Saved" if sel in favs else "☆ Save"
        if st.button(fav_label, key="fav_btn"):
            if sel in favs: favs.remove(sel)
            else: favs.append(sel)
            st.session_state.favorites = favs
            st.rerun()

    with col_mode:
        mode = st.selectbox("Mode", ["Monitor", "Paper"], key="mode_sel", label_visibility="collapsed")
        st.session_state.ecg_mode = mode

    with col_window:
        window = st.slider("Window (s)", 2.0, 8.0, 6.0, 0.5, label_visibility="collapsed")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    w = WAVEFORM_MAP[sel]
    add_to_recent(sel)

    # KPI cards
    kpi_cards(w)
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # ECG Plot
    sig = w["fn"]()
    fig = build_ecg_fig(sig, mode=st.session_state.ecg_mode, window=window)

    mode_badge = "MONITOR" if st.session_state.ecg_mode == "Monitor" else "PAPER"
    badge_bg   = "#0D2010" if st.session_state.ecg_mode == "Monitor" else "#2D0808"
    badge_fg   = "#00E676" if st.session_state.ecg_mode == "Monitor" else "#E74C3C"

    st.markdown(f"""
    <div style="background:#0D1018; border:1px solid #1A1D26; border-radius:4px 4px 0 0;
                padding:0.5rem 1rem; display:flex; justify-content:space-between; align-items:center;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#404858; letter-spacing:0.1em;">
            {sel.upper()}  ·  Lead II
        </div>
        <div style="display:flex; gap:0.8rem; align-items:center;">
            <span style="background:{badge_bg}; border:1px solid {badge_fg}40; border-radius:2px;
                         padding:0.18rem 0.55rem; font-family:'IBM Plex Mono',monospace;
                         font-size:0.65rem; color:{badge_fg}; letter-spacing:0.1em;">{mode_badge}</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#303848;">25 mm/s · 10 mm/mV</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True, "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath"],
        "toImageButtonOptions": {"format": "png", "filename": f"ecg_{sel}", "scale": 2},
    })

    # Bottom panels
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1: info_card(w)
    with c2: feature_cards(w)
    with c3: clinical_card(w)

# ─────────────────────────────────────────────────────────────
# PAGE: DISEASE ATLAS
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Disease Atlas":
    page_header("Disease Atlas", "Medical Reference Library — 15 Cardiac Conditions")

    # Category filter
    cats = ["All"] + sorted(set(w["category"] for w in WAVEFORMS))
    sev_filters = ["All"] + list(SEV_COLORS.keys())
    fc1, fc2, _ = st.columns([2, 2, 4])
    with fc1:
        cat_filter = st.selectbox("Category", cats, label_visibility="collapsed")
    with fc2:
        sev_filter = st.selectbox("Severity", sev_filters, label_visibility="collapsed")

    filtered_waveforms = [
        w for w in WAVEFORMS
        if (cat_filter == "All" or w["category"] == cat_filter)
        and (sev_filter == "All" or w["severity"] == sev_filter)
    ]

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#404858;
                margin-bottom:1rem;">{len(filtered_waveforms)} condition(s) shown</div>
    """, unsafe_allow_html=True)

    COLS = 3
    for i in range(0, len(filtered_waveforms), COLS):
        row = filtered_waveforms[i:i+COLS]
        cols = st.columns(COLS)
        for col, w in zip(cols, row):
            sc = sev_color(w["severity"])
            with col:
                with st.expander(f"", expanded=False):
                    sig = w["fn"]()
                    fig = build_ecg_fig(sig, mode="Monitor", window=4.0)
                    fig.update_layout(height=160, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    feature_cards(w)
                    clinical_card(w)

                st.markdown(f"""
                <div style="background:#10131A; border:1px solid #1A1D26; border-radius:4px;
                            padding:0.75rem 1rem; margin-bottom:0.6rem; cursor:pointer;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.88rem;
                                        font-weight:600; color:#C0C8D8; margin-bottom:0.25rem;">{w["name"]}</div>
                            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                                        color:#404858;">{w["category"]}</div>
                        </div>
                        <div style="background:{sc}20; border:1px solid {sc}50; border-radius:2px;
                                    padding:0.18rem 0.5rem; font-family:'IBM Plex Mono',monospace;
                                    font-size:0.65rem; color:{sc}; white-space:nowrap;">{w["severity"]}</div>
                    </div>
                    <div style="margin-top:0.5rem; display:flex; gap:1.2rem;">
                        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#505870;">
                            HR: <span style="color:#808898;">{w["hr"]}</span>
                        </span>
                        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#505870;">
                            QRS: <span style="color:#808898;">{w["qrs_ms"]}</span>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: ECG ANATOMY
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "ECG Anatomy":
    page_header("ECG Anatomy", "Interactive Waveform Education — Normal Sinus Rhythm")

    st.plotly_chart(build_anatomy_fig(), use_container_width=True, config={
        "displaylogo": False, "scrollZoom": True,
        "toImageButtonOptions": {"format": "png", "filename": "ecg_anatomy", "scale": 2},
    })

    # Component cards
    components = [
        ("P Wave", "#3498DB", "Atrial depolarisation. Normally upright in lead II. Duration 80–100 ms, amplitude < 2.5 mm.", "Absent in AF. Peaked in right atrial hypertrophy. Notched in left atrial hypertrophy."),
        ("PR Interval", "#9B59B6", "From start of P wave to start of QRS. Normal 120–200 ms. Represents AV node conduction.", "Prolonged (> 200 ms) in 1° AV block. Short (< 120 ms) in WPW and accessory pathways."),
        ("QRS Complex", "#00E676", "Ventricular depolarisation. Normal < 120 ms. Narrow = normal. Wide = BBB or ventricular origin.", "Wide bizarre QRS in LBBB, RBBB, V-tach. Q waves indicate prior myocardial infarction."),
        ("ST Segment", "#F39C12", "Between QRS end (J-point) and T-wave onset. Normally isoelectric. Represents plateau phase.", "Elevated in STEMI, pericarditis. Depressed in ischaemia (NSTEMI), digoxin effect."),
        ("T Wave", "#FFD700", "Ventricular repolarisation. Normally upright in most leads. Amplitude < 5 mm (limb), < 10 mm (precordial).", "Inverted in ischaemia, RVH, LVH, BBB. Peaked/hyperacute in hyperkalaemia, early STEMI."),
        ("QT Interval", "#E74C3C", "From start of Q to end of T. Corrected QTc normal < 440 ms (men), < 460 ms (women).", "Prolonged in Long QT syndrome, electrolyte abnormalities, drugs. Risk of Torsades de Pointes."),
    ]

    cols = st.columns(3)
    for i, (name, color, desc, pathology) in enumerate(components):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#10131A; border:1px solid #1A1D26; border-top:2px solid {color};
                        border-radius:4px; padding:0.9rem 1rem; margin-bottom:0.8rem;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                            font-weight:600; color:{color}; margin-bottom:0.5rem;">{name}</div>
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.8rem;
                            color:#909CB0; line-height:1.65; margin-bottom:0.6rem;">{desc}</div>
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.75rem;
                            color:#505870; line-height:1.6; padding-top:0.5rem;
                            border-top:1px solid #1A1D26;">⚠ {pathology}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: COMPARE ECGS
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Compare ECGs":
    page_header("Compare ECGs", "Side-by-Side Clinical ECG Comparison")

    # Preset buttons
    presets = [
        ("Normal vs AFib", "Normal Sinus Rhythm", "Atrial Fibrillation"),
        ("STEMI vs NSTEMI", "STEMI", "NSTEMI / Ischaemia"),
        ("LBBB vs RBBB", "Left BBB", "Right BBB"),
        ("Brady vs Tachy", "Sinus Bradycardia", "Sinus Tachycardia"),
        ("V-Tach vs V-Fib", "Ventricular Tachycardia", "Ventricular Fibrillation"),
    ]
    preset_cols = st.columns(len(presets))
    for col, (label, l, r) in zip(preset_cols, presets):
        if col.button(label, key=f"preset_{label}"):
            st.session_state.cmp_left  = l
            st.session_state.cmp_right = r
            st.rerun()

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    # Selectors
    lc, rc = st.columns(2)
    with lc:
        left_sel = st.selectbox("Left ECG", CONDITION_NAMES,
                                index=CONDITION_NAMES.index(st.session_state.cmp_left),
                                key="cmp_l", label_visibility="collapsed")
        st.session_state.cmp_left = left_sel
    with rc:
        right_sel = st.selectbox("Right ECG", CONDITION_NAMES,
                                 index=CONDITION_NAMES.index(st.session_state.cmp_right),
                                 key="cmp_r", label_visibility="collapsed")
        st.session_state.cmp_right = right_sel

    wL = WAVEFORM_MAP[left_sel]
    wR = WAVEFORM_MAP[right_sel]
    sigL = wL["fn"](); sigR = wR["fn"]()
    figL = build_ecg_fig(sigL, window=5.0)
    figR = build_ecg_fig(sigR, window=5.0)

    lc2, rc2 = st.columns(2)
    with lc2:
        scL = sev_color(wL["severity"])
        st.markdown(f"""
        <div style="background:#0D1018; border:1px solid #1A1D26; border-top:2px solid {scL};
                    border-radius:4px 4px 0 0; padding:0.5rem 1rem; margin-bottom:-1px;
                    display:flex; justify-content:space-between; align-items:center;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#C0C8D8;
                         font-weight:600;">{left_sel}</span>
            <span style="background:{scL}20; border:1px solid {scL}50; border-radius:2px;
                         padding:0.18rem 0.5rem; font-family:'IBM Plex Mono',monospace;
                         font-size:0.65rem; color:{scL};">{wL["severity"]}</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(figL, use_container_width=True, config={"displayModeBar": False})

    with rc2:
        scR = sev_color(wR["severity"])
        st.markdown(f"""
        <div style="background:#0D1018; border:1px solid #1A1D26; border-top:2px solid {scR};
                    border-radius:4px 4px 0 0; padding:0.5rem 1rem; margin-bottom:-1px;
                    display:flex; justify-content:space-between; align-items:center;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#C0C8D8;
                         font-weight:600;">{right_sel}</span>
            <span style="background:{scR}20; border:1px solid {scR}50; border-radius:2px;
                         padding:0.18rem 0.5rem; font-family:'IBM Plex Mono',monospace;
                         font-size:0.65rem; color:{scR};">{wR["severity"]}</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(figR, use_container_width=True, config={"displayModeBar": False})

    # Comparison table
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3A4560;
                letter-spacing:0.14em; text-transform:uppercase; margin:0.6rem 0 0.5rem;">
        Parameter Comparison
    </div>
    """, unsafe_allow_html=True)

    params = [
        ("Category",     wL["category"],  wR["category"]),
        ("Heart Rate",   wL["hr"],        wR["hr"]),
        ("PR Interval",  wL["pr_ms"],     wR["pr_ms"]),
        ("QRS Duration", wL["qrs_ms"],    wR["qrs_ms"]),
        ("Severity",     wL["severity"],  wR["severity"]),
    ]
    rows_html = ""
    for label, vL, vR in params:
        sc_vL = sev_color(vL) if label == "Severity" else "#909CB0"
        sc_vR = sev_color(vR) if label == "Severity" else "#909CB0"
        rows_html += f"""
        <tr style="border-bottom:1px solid #1A1D26;">
            <td style="padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace;
                       font-size:0.75rem; color:#505870; white-space:nowrap;">{label}</td>
            <td style="padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace;
                       font-size:0.75rem; color:{sc_vL}; text-align:center;">{vL}</td>
            <td style="padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace;
                       font-size:0.75rem; color:{sc_vR}; text-align:center;">{vR}</td>
        </tr>
        """
    st.markdown(f"""
    <table style="width:100%; background:#10131A; border:1px solid #1A1D26; border-radius:4px;
                  border-collapse:collapse;">
        <thead>
            <tr style="border-bottom:1px solid #252838;">
                <th style="padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace;
                           font-size:0.65rem; color:#3A4560; letter-spacing:0.12em;
                           text-transform:uppercase; text-align:left;">Parameter</th>
                <th style="padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace;
                           font-size:0.65rem; color:{sev_color(wL["severity"])}; letter-spacing:0.12em;
                           text-transform:uppercase; text-align:center;">{left_sel}</th>
                <th style="padding:0.5rem 1rem; font-family:'IBM Plex Mono',monospace;
                           font-size:0.65rem; color:{sev_color(wR["severity"])}; letter-spacing:0.12em;
                           text-transform:uppercase; text-align:center;">{right_sel}</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Feature diff
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    fl, fr = st.columns(2)
    with fl:
        feature_cards(wL)
        clinical_card(wL)
    with fr:
        feature_cards(wR)
        clinical_card(wR)

# ─────────────────────────────────────────────────────────────
# PAGE: RESEARCH EXPORT
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Research Export":
    page_header("Research Export", "Dataset Generation & Export for Clinical Research")

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        res_cond = st.selectbox("Select Condition", ["All Conditions"] + CONDITION_NAMES,
                                label_visibility="collapsed")
    with rc2:
        n_samples = st.selectbox("Samples", [100, 500, 1000, 5000], label_visibility="collapsed")

    # Stats table
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3A4560;
                letter-spacing:0.14em; text-transform:uppercase; margin:0.8rem 0 0.5rem;">
        Dataset Summary
    </div>
    """, unsafe_allow_html=True)

    target_waveforms = WAVEFORMS if res_cond == "All Conditions" else [WAVEFORM_MAP[res_cond]]
    rows = ""
    for w in target_waveforms:
        sc = sev_color(w["severity"])
        count = n_samples if res_cond != "All Conditions" else n_samples // len(WAVEFORMS)
        rows += f"""
        <tr style="border-bottom:1px solid #1A1D26;">
            <td style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#909CB0;">{w["name"]}</td>
            <td style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#505870; text-align:center;">{w["category"]}</td>
            <td style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:{sc}; text-align:center;">{w["severity"]}</td>
            <td style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#404858; text-align:center;">{w["hr"]}</td>
            <td style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#00E676; text-align:center;">{count}</td>
        </tr>
        """
    st.markdown(f"""
    <table style="width:100%; background:#10131A; border:1px solid #1A1D26; border-radius:4px; border-collapse:collapse;">
        <thead><tr style="border-bottom:1px solid #252838;">
            <th style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3A4560; letter-spacing:0.12em; text-transform:uppercase; text-align:left;">Condition</th>
            <th style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3A4560; letter-spacing:0.12em; text-transform:uppercase; text-align:center;">Category</th>
            <th style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3A4560; letter-spacing:0.12em; text-transform:uppercase; text-align:center;">Severity</th>
            <th style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3A4560; letter-spacing:0.12em; text-transform:uppercase; text-align:center;">Heart Rate</th>
            <th style="padding:0.45rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3A4560; letter-spacing:0.12em; text-transform:uppercase; text-align:center;">Samples</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Export buttons
    ec1, ec2, ec3 = st.columns(3)

    # CSV export
    with ec1:
        @st.cache_data
        def gen_csv(cond_name, n):
            import csv, io
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["sample_id","condition","category","severity","heart_rate","pr_ms","qrs_ms","signal_json"])
            wlist = WAVEFORMS if cond_name == "All Conditions" else [WAVEFORM_MAP[cond_name]]
            per = n if cond_name != "All Conditions" else max(1, n // len(WAVEFORMS))
            for w in wlist:
                for i in range(per):
                    sig = w["fn"]()
                    writer.writerow([i, w["name"], w["category"], w["severity"],
                                     w["hr"], w["pr_ms"], w["qrs_ms"],
                                     json.dumps(sig[:500].tolist())])
            return buf.getvalue().encode()

        csv_data = gen_csv(res_cond, n_samples)
        st.download_button("⬇ Download CSV", csv_data,
                           f"ecg_dataset_{res_cond.replace(' ','_').replace('/','_')}_{n_samples}.csv",
                           "text/csv", key="dl_csv")

    # NumPy export
    with ec2:
        @st.cache_data
        def gen_npy(cond_name, n):
            wlist = WAVEFORMS if cond_name == "All Conditions" else [WAVEFORM_MAP[cond_name]]
            per = n if cond_name != "All Conditions" else max(1, n // len(WAVEFORMS))
            arrays = []
            for w in wlist:
                for _ in range(per):
                    arrays.append(w["fn"]()[:1000])
            arr = np.stack(arrays)
            buf = io.BytesIO(); np.save(buf, arr); buf.seek(0)
            return buf.getvalue()

        npy_data = gen_npy(res_cond, n_samples)
        st.download_button("⬇ Download .npy", npy_data,
                           f"ecg_{res_cond.replace(' ','_').replace('/','_')}_{n_samples}.npy",
                           "application/octet-stream", key="dl_npy")

    # PNG export of ECG strip
    with ec3:
        if res_cond != "All Conditions":
            w = WAVEFORM_MAP[res_cond]
            sig = w["fn"]()
            fig = build_ecg_fig(sig, window=6.0)
            img_bytes = fig.to_image(format="png", width=1400, height=400, scale=2)
            st.download_button("⬇ Download PNG", img_bytes,
                               f"ecg_{res_cond.replace(' ','_').replace('/','_')}.png",
                               "image/png", key="dl_png")
        else:
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#404858;
                        padding:0.5rem 0; text-align:center;">
                Select a condition for PNG export
            </div>
            """, unsafe_allow_html=True)

    # Metadata card
    total_samples = n_samples
    total_duration = (total_samples * DURATION) / 3600
    st.markdown(f"""
    <div style="background:#10131A; border:1px solid #1A1D26; border-radius:4px;
                padding:1rem 1.2rem; margin-top:1rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3A4560;
                    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.7rem;">
            Dataset Metadata
        </div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem;">
            {"".join([f'<div><div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#404858;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">{k}</div><div style="font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600;color:#00E676;">{v}</div></div>' for k,v in [("Total Samples", f"{total_samples:,}"), ("Sampling Rate", "500 Hz"), ("Duration/Sample", f"{DURATION}s"), ("Total Hours", f"{total_duration:.2f}h")]])}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "About":
    page_header("About", "ECG Atlas Clinical — Platform Information")

    st.markdown("""
    <div style="background:#10131A; border:1px solid #1A1D26; border-radius:4px;
                padding:1.5rem 2rem; max-width:780px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.2rem; font-weight:600;
                    color:#00E676; margin-bottom:0.3rem;">🫀 ECG Atlas Clinical</div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.8rem; color:#505870;
                    letter-spacing:0.08em; margin-bottom:1.2rem;">
            Interactive ECG Learning, Analysis and Research Platform
        </div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem; color:#909CB0;
                    line-height:1.8; margin-bottom:1.2rem;">
            ECG Atlas Clinical is a professional-grade electrocardiography education and research tool
            designed for cardiologists, medical students, biomedical engineers, and clinical researchers.
            The platform provides synthetic ECG waveform generation, clinical interpretation guidance,
            comparative analysis tools, and research dataset export capabilities.
        </div>
        <div style="border-top:1px solid #1A1D26; padding-top:1rem; display:grid;
                    grid-template-columns:repeat(3,1fr); gap:1rem;">
    """, unsafe_allow_html=True)

    stats = [
        ("Conditions", "15", "Cardiac arrhythmias & pathologies"),
        ("Sample Rate", "500 Hz", "Clinical-grade sampling"),
        ("ECG Duration", "8 s", "Per generated strip"),
        ("Signal Model", "Gaussian", "P-QRS-T decomposition"),
        ("Noise Model", "AWGN", "SNR 18–30 dB"),
        ("Filter", "Bandpass", "0.5–40 Hz, 4th order"),
    ]
    stat_html = ""
    for label, val, sub in stats:
        stat_html += f"""
        <div style="background:#141820; border:1px solid #1A1D26; border-radius:3px;
                    padding:0.7rem 0.9rem; margin-bottom:0.6rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#404858;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;">{label}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; font-weight:600;
                        color:#00E676;">{val}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.72rem; color:#505870;
                        margin-top:0.2rem;">{sub}</div>
        </div>
        """
    st.markdown(f"""
    <div style="background:#10131A; border:1px solid #1A1D26; border-radius:4px; padding:1.2rem;
                display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin-top:0.5rem;">
        {stat_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#10131A; border:1px solid #1A1D26; border-left:2px solid #F39C12;
                border-radius:4px; padding:1rem 1.2rem; margin-top:1rem; max-width:780px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#F39C12;
                    letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;">
            ⚠ Clinical Disclaimer
        </div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.8rem; color:#707888;
                    line-height:1.7;">
            This platform is intended solely for educational, training, and research purposes.
            All ECG waveforms are synthetically generated and do not represent actual patient data.
            This tool must not be used for clinical diagnosis, patient management, or any
            medical decision-making. Always consult a qualified cardiologist for clinical ECG interpretation.
        </div>
    </div>
    """, unsafe_allow_html=True)
