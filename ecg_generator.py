# ============================================================
# 🫀 ECG Waveform Visualization — Google Colab
# NO API KEY REQUIRED — 100% local, runs offline
# ============================================================
# Just run: Runtime > Run All
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install / import
# ─────────────────────────────────────────────────────────────
# !pip install matplotlib numpy scipy --quiet

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries ready — no API key needed!")

# ─────────────────────────────────────────────────────────────
# CELL 2 — Core ECG Waveform Generator
# ─────────────────────────────────────────────────────────────

FS = 500        # Sampling rate (Hz)
DURATION = 8    # Seconds per strip

def bandpass(sig, lo=0.5, hi=40):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype='band')
    return filtfilt(b, a, sig)

def add_noise(sig, snr=30):
    p = np.mean(sig**2)
    n = p / 10**(snr/10)
    return sig + np.random.normal(0, np.sqrt(n), len(sig))

def gauss(t, mu, sigma, amp):
    return amp * np.exp(-((t - mu)**2) / (2 * sigma**2))

# ── 1. Normal Sinus Rhythm ───────────────────────────────────
def gen_normal(hr=72):
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / hr
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS * rr))
        p = gauss(bt, 0.12, 0.022, 0.15)
        q = gauss(bt, 0.22, 0.008, -0.09)
        r = gauss(bt, 0.265, 0.009, 1.20)
        s = gauss(bt, 0.305, 0.008, -0.30)
        t = gauss(bt, 0.50,  0.055, 0.28)
        beat = p + q + r + s + t
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 2. Sinus Bradycardia ─────────────────────────────────────
def gen_bradycardia(hr=42):
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / hr
    for bs in np.arange(0.4, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS * rr))
        p = gauss(bt, 0.14, 0.025, 0.16)
        q = gauss(bt, 0.24, 0.009, -0.10)
        r = gauss(bt, 0.28, 0.009, 1.10)
        s = gauss(bt, 0.32, 0.008, -0.28)
        t = gauss(bt, 0.56, 0.060, 0.26)
        beat = p + q + r + s + t
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 3. Sinus Tachycardia ─────────────────────────────────────
def gen_tachycardia(hr=148):
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / hr
    for bs in np.arange(0.1, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS * rr))
        p = gauss(bt, 0.08, 0.018, 0.17)
        q = gauss(bt, 0.15, 0.007, -0.10)
        r = gauss(bt, 0.19, 0.008, 1.15)
        s = gauss(bt, 0.23, 0.007, -0.32)
        # P merges into T at high rates
        t = gauss(bt, 0.38, 0.038, 0.24)
        beat = p + q + r + s + t
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 4. Atrial Fibrillation ───────────────────────────────────
def gen_afib():
    N = FS * DURATION
    t_arr = np.linspace(0, DURATION, N)
    sig = np.zeros(N)
    # Fibrillatory baseline
    for f in [4.2, 5.8, 7.1, 9.3, 11.5]:
        sig += 0.045 * np.sin(2*np.pi*f*t_arr + np.random.uniform(0, 2*np.pi))
    # Irregular ventricular response
    mean_rr = 60.0 / 110
    cur = 0.3
    while cur < DURATION - 0.5:
        rr = mean_rr * np.random.uniform(0.55, 1.7)
        bt = np.linspace(0, 0.5, int(FS*0.5))
        ph = bt / 0.5
        q = gauss(ph, 0.28, 0.008, -0.10)
        r = gauss(ph, 0.32, 0.009, 1.05)
        s = gauss(ph, 0.36, 0.008, -0.27)
        tw = gauss(ph, 0.65, 0.050, 0.20)
        beat = q + r + s + tw
        idx = int(cur * FS)
        if idx + len(bt) <= N:
            sig[idx:idx+len(bt)] += beat
        cur += rr
    return add_noise(bandpass(sig))

# ── 5. Atrial Flutter ────────────────────────────────────────
def gen_flutter():
    N = FS * DURATION
    t_arr = np.linspace(0, DURATION, N)
    sig = np.zeros(N)
    # Sawtooth flutter waves at ~300 bpm (5 Hz)
    for harm in [1, 2, 3]:
        sig += (0.18 / harm) * np.sin(2*np.pi * 5 * harm * t_arr) # + np.random.uniform(0, 2*np.pi)) -- removed phase noise
    # Regular ventricular response — 2:1 block → ~150 bpm
    rr = 60.0 / 150
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        q = gauss(ph, 0.22, 0.008, -0.08)
        r = gauss(ph, 0.26, 0.008, 1.0)
        s = gauss(ph, 0.30, 0.007, -0.24)
        tw = gauss(ph, 0.55, 0.050, 0.22)
        beat = q + r + s + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 6. First-Degree AV Block ─────────────────────────────────
def gen_av_block_1():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 65
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        # PR interval prolonged to 280 ms
        p = gauss(bt, 0.10, 0.022, 0.15)
        q = gauss(bt, 0.30, 0.008, -0.09)  # shifted right
        r = gauss(bt, 0.34, 0.009, 1.18)
        s = gauss(bt, 0.38, 0.008, -0.28)
        t = gauss(bt, 0.60, 0.055, 0.26)
        beat = p + q + r + s + t
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 7. Second-Degree AV Block (Mobitz II) ────────────────────
def gen_av_block_2():
    N = FS * DURATION
    sig = np.zeros(N)
    short_rr = 60.0 / 75
    long_rr  = short_rr * 2.2  # dropped beat
    cur = 0.3
    beat_count = 0
    while cur < DURATION - long_rr:
        bt_len = long_rr if beat_count % 3 == 2 else short_rr
        bt = np.linspace(0, bt_len, int(FS*bt_len))
        ph = bt / bt_len
        if beat_count % 3 != 2:  # conducted beat
            p = gauss(ph, 0.12, 0.022, 0.15)
            q = gauss(ph, 0.22, 0.008, -0.09)
            r = gauss(ph, 0.265, 0.009, 1.15)
            s = gauss(ph, 0.305, 0.008, -0.28)
            t = gauss(ph, 0.50, 0.055, 0.26)
            beat = p + q + r + s + t
        else:  # dropped beat — only P wave
            beat = gauss(ph, 0.08, 0.020, 0.14)
        i = int(cur * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
        cur += bt_len
        beat_count += 1
    return add_noise(bandpass(sig))

# ── 8. STEMI ─────────────────────────────────────────────────
def gen_stemi():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 88
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        p  = gauss(ph, 0.12, 0.022, 0.13)
        q  = gauss(ph, 0.22, 0.008, -0.22)   # pathological Q
        r  = gauss(ph, 0.26, 0.009, 0.90)
        s  = gauss(ph, 0.30, 0.007, -0.10)
        # ST elevation dome
        st_mask = (ph >= 0.31) & (ph <= 0.55)
        st = np.where(st_mask, 0.40 * np.exp(-((ph-0.33)**2)/(2*0.07**2)), 0)
        # Hyperacute T
        tw = gauss(ph, 0.65, 0.042, 0.48)
        beat = p + q + r + s + st + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 9. NSTEMI / T-wave inversion ─────────────────────────────
def gen_nstemi():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 82
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        p  = gauss(ph, 0.12, 0.022, 0.14)
        q  = gauss(ph, 0.22, 0.009, -0.12)
        r  = gauss(ph, 0.265, 0.009, 1.0)
        s  = gauss(ph, 0.305, 0.008, -0.22)
        # ST depression + inverted T
        st_dep = np.where((ph>=0.31)&(ph<=0.50), -0.18*np.exp(-((ph-0.35)**2)/(2*0.06**2)), 0)
        tw = gauss(ph, 0.58, 0.045, -0.30)   # inverted T
        beat = p + q + r + s + st_dep + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 10. Ventricular Tachycardia ──────────────────────────────
def gen_vtach():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 170
    for bs in np.arange(0.15, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        # Wide bizarre QRS
        main  = 1.20 * np.exp(-((ph-0.22)**2)/(2*0.045**2))
        main -= 0.55 * np.exp(-((ph-0.38)**2)/(2*0.032**2))
        main += 0.25 * np.exp(-((ph-0.50)**2)/(2*0.025**2))
        tw    = -0.38 * np.exp(-((ph-0.72)**2)/(2*0.042**2))  # discordant
        beat = main + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig), snr=22)

# ── 11. Ventricular Fibrillation ─────────────────────────────
def gen_vfib():
    N = FS * DURATION
    t_arr = np.linspace(0, DURATION, N)
    sig = np.zeros(N)
    # Chaotic multi-frequency noise bursts
    for f in [2.1, 3.4, 5.2, 7.8, 11.3, 15.6]:
        amp = np.random.uniform(0.25, 0.70)
        phase = np.random.uniform(0, 2*np.pi)
        sig += amp * np.sin(2*np.pi*f*t_arr + phase)
    # Amplitude modulation for realistic coarse VF
    envelope = 0.6 + 0.4*np.sin(2*np.pi*1.1*t_arr)
    sig *= envelope
    return add_noise(bandpass(sig, lo=1, hi=35), snr=18)

# ── 12. Left Bundle Branch Block ─────────────────────────────
def gen_lbbb():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 70
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        p = gauss(ph, 0.12, 0.024, 0.16)
        # Broad notched QRS (M-shaped)
        r1 = gauss(ph, 0.24, 0.014, 0.65)
        r2 = gauss(ph, 0.32, 0.014, 0.90)  # second peak (notched)
        s  = gauss(ph, 0.40, 0.010, -0.20)
        # Discordant T wave
        tw = gauss(ph, 0.62, 0.060, -0.32)
        beat = p + r1 + r2 + s + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 13. Right Bundle Branch Block ────────────────────────────
def gen_rbbb():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 72
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        p  = gauss(ph, 0.12, 0.022, 0.15)
        q  = gauss(ph, 0.21, 0.008, -0.08)
        r  = gauss(ph, 0.25, 0.010, 1.10)
        s  = gauss(ph, 0.31, 0.012, -0.45)  # deep broad S
        rp = gauss(ph, 0.37, 0.013, 0.55)   # R' (rsR' pattern)
        tw = gauss(ph, 0.62, 0.055, -0.28)  # inverted T
        beat = p + q + r + s + rp + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 14. Wolff-Parkinson-White (WPW) ──────────────────────────
def gen_wpw():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 80
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        p  = gauss(ph, 0.10, 0.020, 0.16)
        # Delta wave — slurred QRS upstroke
        delta = 0.35 * np.where((ph>=0.17)&(ph<=0.23), np.sin(np.pi*(ph-0.17)/0.06), 0)
        r  = gauss(ph, 0.24, 0.010, 1.05)
        s  = gauss(ph, 0.28, 0.009, -0.22)
        # Short PR due to accessory pathway
        tw = gauss(ph, 0.52, 0.055, 0.25)
        beat = p + delta + r + s + tw
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

# ── 15. Long QT Syndrome ─────────────────────────────────────
def gen_long_qt():
    N = FS * DURATION
    sig = np.zeros(N)
    rr = 60.0 / 65
    for bs in np.arange(0.3, DURATION - rr, rr):
        bt = np.linspace(0, rr, int(FS*rr))
        ph = bt / rr
        p  = gauss(ph, 0.12, 0.022, 0.15)
        q  = gauss(ph, 0.22, 0.008, -0.09)
        r  = gauss(ph, 0.265, 0.009, 1.18)
        s  = gauss(ph, 0.305, 0.008, -0.28)
        # Very prolonged QT — T wave stretched far right
        tw = gauss(ph, 0.76, 0.075, 0.30)   # QTc > 500 ms
        u  = gauss(ph, 0.88, 0.030, 0.12)   # prominent U wave
        beat = p + q + r + s + tw + u
        i = int(bs * FS)
        if i + len(bt) <= N:
            sig[i:i+len(bt)] += beat
    return add_noise(bandpass(sig))

print("✅ 15 ECG waveform generators ready")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Waveform Registry
# ─────────────────────────────────────────────────────────────

WAVEFORMS = [
    {
        "name":     "Normal Sinus Rhythm",
        "fn":       gen_normal,
        "category": "Normal",
        "cat_color":"#2ecc71",
        "hr":       "60–100",
        "severity": "NORMAL",
        "sev_color":"#2ecc71",
        "pr_ms":    "120–200",
        "qrs_ms":   "< 120",
        "key":      "Regular P before every QRS. Uniform P-P and R-R intervals.",
        "clinical": "Healthy cardiac conduction. No treatment needed.",
        "features": ["Upright P wave in II", "PR 120–200 ms", "Narrow QRS < 120 ms", "T wave concordant"],
    },
    {
        "name":     "Sinus Bradycardia",
        "fn":       gen_bradycardia,
        "category": "Sinus",
        "cat_color":"#3498db",
        "hr":       "< 60",
        "severity": "MILD",
        "sev_color":"#3498db",
        "pr_ms":    "120–200",
        "qrs_ms":   "< 120",
        "key":      "Normal morphology, rate < 60 bpm. Long R-R intervals.",
        "clinical": "Common in athletes. Treat only if symptomatic (dizziness, syncope).",
        "features": ["Rate < 60 bpm", "Normal P-QRS-T", "Prolonged R-R", "Possible escape beats"],
    },
    {
        "name":     "Sinus Tachycardia",
        "fn":       gen_tachycardia,
        "category": "Sinus",
        "cat_color":"#3498db",
        "hr":       "> 100",
        "severity": "MILD",
        "sev_color":"#3498db",
        "pr_ms":    "< 160",
        "qrs_ms":   "< 120",
        "key":      "Rate > 100 bpm. P waves may merge with preceding T waves.",
        "clinical": "Physiological response (fever, pain, anxiety). Treat the cause.",
        "features": ["Rate > 100 bpm", "P-T merging at high rates", "Short R-R", "Narrow QRS"],
    },
    {
        "name":     "Atrial Fibrillation",
        "fn":       gen_afib,
        "category": "Atrial",
        "cat_color":"#9b59b6",
        "hr":       "60–180 (irregular)",
        "severity": "ARRHYTHMIA",
        "sev_color":"#9b59b6",
        "pr_ms":    "Absent",
        "qrs_ms":   "< 120",
        "key":      "Irregularly irregular rhythm. No visible P waves. Fibrillatory baseline.",
        "clinical": "Most common arrhythmia. Risk of stroke — anticoagulate. Rate/rhythm control.",
        "features": ["No P waves", "Irregularly irregular RR", "Fibrillatory baseline", "Narrow QRS (unless aberrant)"],
    },
    {
        "name":     "Atrial Flutter",
        "fn":       gen_flutter,
        "category": "Atrial",
        "cat_color":"#9b59b6",
        "hr":       "150 (2:1 block)",
        "severity": "ARRHYTHMIA",
        "sev_color":"#9b59b6",
        "pr_ms":    "Absent",
        "qrs_ms":   "< 120",
        "key":      "Sawtooth flutter waves at 300 bpm. Regular ventricular rate (2:1 or 4:1).",
        "clinical": "Cardioversion or ablation. Anticoagulation as in AF.",
        "features": ["Sawtooth F-waves 300 bpm", "2:1 or 4:1 conduction", "Regular ventricular rate", "No isoelectric baseline"],
    },
    {
        "name":     "1° AV Block",
        "fn":       gen_av_block_1,
        "category": "Conduction",
        "cat_color":"#e67e22",
        "hr":       "60–100",
        "severity": "WARNING",
        "sev_color":"#e67e22",
        "pr_ms":    "> 200",
        "qrs_ms":   "< 120",
        "key":      "Prolonged PR interval (> 200 ms). Every P conducts to QRS.",
        "clinical": "Usually benign. No specific treatment unless symptomatic.",
        "features": ["PR > 200 ms (constant)", "Every P followed by QRS", "Normal QRS morphology", "Slowed AV node conduction"],
    },
    {
        "name":     "2° AV Block (Mobitz II)",
        "fn":       gen_av_block_2,
        "category": "Conduction",
        "cat_color":"#e67e22",
        "hr":       "45–60",
        "severity": "WARNING",
        "sev_color":"#e67e22",
        "pr_ms":    "Constant then dropped",
        "qrs_ms":   "± wide",
        "key":      "Constant PR then sudden non-conducted P wave. High risk of progression.",
        "clinical": "High risk — may need pacemaker. Does not improve with atropine.",
        "features": ["Constant PR then dropped QRS", "Infra-Hisian block", "2:1 or 3:1 pattern", "Pacemaker often indicated"],
    },
    {
        "name":     "STEMI",
        "fn":       gen_stemi,
        "category": "Ischemia",
        "cat_color":"#e74c3c",
        "hr":       "60–100",
        "severity": "CRITICAL",
        "sev_color":"#e74c3c",
        "pr_ms":    "120–200",
        "qrs_ms":   "≥ 100",
        "key":      "ST elevation ≥ 1 mm in ≥ 2 contiguous leads. Pathological Q waves.",
        "clinical": "MEDICAL EMERGENCY. Activate cath lab. PCI within 90 min. Aspirin + heparin.",
        "features": ["ST elevation ≥ 1 mm", "Pathological Q waves", "Hyperacute T waves", "Reciprocal ST depression"],
    },
    {
        "name":     "NSTEMI / Ischaemia",
        "fn":       gen_nstemi,
        "category": "Ischemia",
        "cat_color":"#e74c3c",
        "hr":       "70–100",
        "severity": "CRITICAL",
        "sev_color":"#e74c3c",
        "pr_ms":    "120–200",
        "qrs_ms":   "< 120",
        "key":      "ST depression + T-wave inversion without ST elevation. Troponin rise.",
        "clinical": "Urgent: anticoagulate, antiplatelets. Angiography within 24–72 h.",
        "features": ["ST depression ≥ 0.5 mm", "T-wave inversion", "No ST elevation", "Troponin elevation"],
    },
    {
        "name":     "Ventricular Tachycardia",
        "fn":       gen_vtach,
        "category": "Ventricular",
        "cat_color":"#c0392b",
        "hr":       "150–250",
        "severity": "CRITICAL",
        "sev_color":"#c0392b",
        "pr_ms":    "Absent",
        "qrs_ms":   "> 120",
        "key":      "Wide complex tachycardia, QRS > 120 ms. AV dissociation. No P waves.",
        "clinical": "EMERGENCY. Unstable → DC cardioversion. Stable → amiodarone IV.",
        "features": ["Rate > 150 bpm", "QRS > 120 ms (bizarre)", "AV dissociation", "Discordant T waves"],
    },
    {
        "name":     "Ventricular Fibrillation",
        "fn":       gen_vfib,
        "category": "Ventricular",
        "cat_color":"#c0392b",
        "hr":       "N/A",
        "severity": "FATAL",
        "sev_color":"#c0392b",
        "pr_ms":    "None",
        "qrs_ms":   "None",
        "key":      "Completely chaotic baseline. No organised complexes. No cardiac output.",
        "clinical": "CARDIAC ARREST. Immediate CPR + defibrillation. Adrenaline 1mg IV.",
        "features": ["No organised complexes", "Chaotic baseline", "No cardiac output", "Immediate defibrillation needed"],
    },
    {
        "name":     "Left BBB",
        "fn":       gen_lbbb,
        "category": "Bundle Branch",
        "cat_color":"#1abc9c",
        "hr":       "60–100",
        "severity": "WARNING",
        "sev_color":"#e67e22",
        "pr_ms":    "Normal",
        "qrs_ms":   "> 120",
        "key":      "QRS > 120 ms. Broad notched (M-shaped) R in I, aVL, V5–V6. WiLLiaM pattern.",
        "clinical": "New LBBB with chest pain = STEMI equivalent. Investigate for structural disease.",
        "features": ["QRS > 120 ms", "Notched R in V5-V6 (M-shape)", "Discordant ST/T", "No septal Q in I, V5, V6"],
    },
    {
        "name":     "Right BBB",
        "fn":       gen_rbbb,
        "category": "Bundle Branch",
        "cat_color":"#1abc9c",
        "hr":       "60–100",
        "severity": "MILD",
        "sev_color":"#3498db",
        "pr_ms":    "Normal",
        "qrs_ms":   "> 120",
        "key":      "QRS > 120 ms. rsR' (M) pattern in V1. Deep slurred S in I, V6. MaRRoW pattern.",
        "clinical": "May be normal variant. Investigate if new. Right heart disease workup.",
        "features": ["QRS > 120 ms", "RSR' in V1 (rabbit ears)", "Deep S in I, V6", "Discordant T in V1-V2"],
    },
    {
        "name":     "Wolff-Parkinson-White",
        "fn":       gen_wpw,
        "category": "Pre-excitation",
        "cat_color":"#f39c12",
        "hr":       "60–100 (SVT burst)",
        "severity": "WARNING",
        "sev_color":"#e67e22",
        "pr_ms":    "< 120 (short)",
        "qrs_ms":   "Broad (slurred)",
        "key":      "Short PR < 120 ms. Delta wave. Broad QRS. Accessory conduction pathway.",
        "clinical": "Risk of SVT and AF with rapid conduction. Catheter ablation curative.",
        "features": ["Short PR < 120 ms", "Delta wave (slurred upstroke)", "Broad QRS", "Pseudo ST/T changes"],
    },
    {
        "name":     "Long QT Syndrome",
        "fn":       gen_long_qt,
        "category": "Channelopathy",
        "cat_color":"#e74c3c",
        "hr":       "60–80",
        "severity": "WARNING",
        "sev_color":"#e67e22",
        "pr_ms":    "Normal",
        "qrs_ms":   "Normal",
        "key":      "QTc > 500 ms. Prominent U wave. Risk of torsades de pointes → VF.",
        "clinical": "Avoid QT-prolonging drugs. Beta-blockers. ICD if high risk. Genetic testing.",
        "features": ["QTc > 500 ms", "Prominent U wave", "T-wave morphology changes", "Risk of Torsades de Pointes"],
    },
]

print(f"✅ Registry loaded: {len(WAVEFORMS)} ECG conditions")
for w in WAVEFORMS:
    print(f"   [{w['severity']:10s}]  {w['name']}")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Single Condition Deep-Dive Plot
# ─────────────────────────────────────────────────────────────

def plot_single(wf_dict, window=5.0):
    """
    Full clinical deep-dive for one waveform:
    large ECG trace + metrics + feature callouts + mini RR histogram.
    """
    w = wf_dict
    sig = w["fn"]()
    N = len(sig)
    t = np.linspace(0, DURATION, N)
    mask = t <= window
    tw, sw = t[mask], sig[mask]

    fig = plt.figure(figsize=(17, 9), facecolor="#0c0c0c")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.55, wspace=0.30,
                           top=0.88, bottom=0.07,
                           left=0.05, right=0.97)

    sc = w["sev_color"]
    cc = w["cat_color"]

    # Title
    fig.text(0.5, 0.95, w["name"], ha="center", fontsize=20,
             fontweight="bold", color="white", fontfamily="monospace")
    fig.text(0.5, 0.91, f"Category: {w['category']}   |   HR: {w['hr']} bpm",
             ha="center", fontsize=11, color=cc)
    sbox = dict(boxstyle="round,pad=0.35", fc=sc+"33", ec=sc, lw=1.2)
    fig.text(0.91, 0.935, f" {w['severity']} ", ha="center", va="center",
             fontsize=10, fontweight="bold", color=sc,
             bbox=sbox, fontfamily="monospace")

    # ── ECG Trace ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor("#071207")
    for gx in np.arange(0, window+0.2, 0.2):
        ax.axvline(gx, color="#0a2a0a", lw=0.4)
    for gy in np.arange(-2, 2.2, 0.2):
        ax.axhline(gy, color="#0a2a0a", lw=0.4)
    for gx in np.arange(0, window+1, 1.0):
        ax.axvline(gx, color="#0d3d0d", lw=0.8)
    for gy in np.arange(-2, 2.2, 1.0):
        ax.axhline(gy, color="#0d3d0d", lw=0.8)
    ax.axhline(0, color="#1a5c1a", lw=0.8)

    ax.plot(tw, sw, color=cc, alpha=0.2, lw=6)
    ax.plot(tw, sw, color=cc, alpha=0.4, lw=2.5)
    ax.plot(tw, sw, color=cc, alpha=0.92, lw=1.2)

    ax.set_xlim(0, window)
    ax.set_ylim(-1.5, 1.8)
    ax.set_xlabel("Time (seconds)", color="#888", fontsize=9)
    ax.set_ylabel("Amplitude (mV)", color="#888", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#222")
    ax.text(0.005, 0.97, "Lead II  •  25 mm/s  •  10 mm/mV",
            transform=ax.transAxes, color="#333",
            fontsize=8, va="top", fontfamily="monospace")

    # ── Metrics card ─────────────────────────────────────────
    ax_m = fig.add_subplot(gs[0, 2])
    ax_m.set_facecolor("#111")
    ax_m.axis("off")
    ax_m.text(0.5, 0.97, "ECG PARAMETERS", ha="center", va="top",
              color="#666", fontsize=8, fontweight="bold",
              fontfamily="monospace", transform=ax_m.transAxes)

    rows = [
        ("Heart Rate",   w["hr"] + " bpm"),
        ("PR Interval",  w["pr_ms"] + " ms" if "ms" not in w["pr_ms"] else w["pr_ms"]),
        ("QRS Duration", w["qrs_ms"] + " ms" if "ms" not in w["qrs_ms"] else w["qrs_ms"]),
        ("Category",     w["category"]),
        ("Severity",     w["severity"]),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = 0.85 - i * 0.14
        vc = sc if lbl == "Severity" else cc if lbl == "Category" else "#dddddd"
        ax_m.text(0.05, y, lbl, transform=ax_m.transAxes,
                  color="#888", fontsize=9, va="center", fontfamily="monospace")
        ax_m.text(0.97, y, val, transform=ax_m.transAxes,
                  color=vc, fontsize=9.5, va="center", ha="right",
                  fontweight="bold", fontfamily="monospace")
        ax_m.axhline(y=y-0.05, xmin=0.03, xmax=0.97,
                     color="#222", lw=0.5)

    # ── Features panel ───────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 0])
    ax_f.set_facecolor("#0e0e0e")
    ax_f.axis("off")
    ax_f.text(0.5, 0.97, "KEY ECG FEATURES", ha="center", va="top",
              color="#666", fontsize=8, fontweight="bold",
              fontfamily="monospace", transform=ax_f.transAxes)
    for i, feat in enumerate(w["features"]):
        ax_f.text(0.06, 0.82 - i*0.17, f"✦  {feat}",
                  transform=ax_f.transAxes,
                  color="#cccccc", fontsize=9.5, va="top")

    # ── Clinical note ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_facecolor("#0e0e0e")
    ax_c.axis("off")
    ax_c.text(0.5, 0.97, "CLINICAL ACTION", ha="center", va="top",
              color="#666", fontsize=8, fontweight="bold",
              fontfamily="monospace", transform=ax_c.transAxes)
    wrapped = _wrap(w["clinical"], 32)
    ax_c.text(0.5, 0.75, wrapped, ha="center", va="top",
              transform=ax_c.transAxes,
              color="#aaaaaa", fontsize=9, linespacing=1.6)
    ax_c.text(0.5, 0.25, w["key"], ha="center", va="top",
              transform=ax_c.transAxes,
              color="#666666", fontsize=8, linespacing=1.5,
              style="italic", wrap=True)

    # ── RR Histogram ─────────────────────────────────────────
    ax_rr = fig.add_subplot(gs[1, 2])
    ax_rr.set_facecolor("#0a0a0a")
    base_rr = 600
    try:
        hr_val = float(w["hr"].split("–")[0].replace(">","").replace("<","").strip())
        base_rr = 60000 / hr_val
    except:
        pass
    if w["name"] == "Atrial Fibrillation":
        rr_v = np.random.uniform(base_rr*0.4, base_rr*2.0, 50)
    elif w["name"] == "Ventricular Fibrillation":
        rr_v = np.random.uniform(100, 400, 60)
    else:
        rr_v = np.random.normal(base_rr, base_rr*0.04, 40)
    ax_rr.hist(rr_v, bins=14, color=cc, alpha=0.7, edgecolor="#0a0a0a")
    ax_rr.axvline(np.mean(rr_v), color="white", lw=1, ls="--", alpha=0.5,
                  label=f"μ = {np.mean(rr_v):.0f} ms")
    ax_rr.set_title("R-R Interval Distribution", color="#aaa",
                    fontsize=9, pad=5, fontfamily="monospace")
    ax_rr.set_xlabel("R-R (ms)", color="#666", fontsize=8)
    ax_rr.set_ylabel("Count",    color="#666", fontsize=8)
    ax_rr.tick_params(colors="#555", labelsize=7)
    for sp in ax_rr.spines.values(): sp.set_edgecolor("#222")
    ax_rr.legend(fontsize=7, labelcolor="#888",
                 facecolor="#111", edgecolor="#333")

    plt.savefig(f"ecg_{w['name'].replace(' ','_').replace('/','_')}.png",
                dpi=140, bbox_inches="tight", facecolor="#0c0c0c")
    plt.show()
    print(f"   💾 Saved: ecg_{w['name'].replace(' ','_')}.png")


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur)+len(word)+1 > width:
            lines.append(cur.strip()); cur = word+" "
        else:
            cur += word+" "
    if cur: lines.append(cur.strip())
    return "\n".join(lines)

print("✅ Single-condition plotter ready")


# ─────────────────────────────────────────────────────────────
# CELL 5 — Master Overview: All 15 on one canvas
# ─────────────────────────────────────────────────────────────

def plot_all_overview(window=4.0):
    """
    5-column × 3-row grid of all 15 ECG conditions,
    colour-coded by severity.
    """
    COLS, ROWS = 5, 3
    fig, axes = plt.subplots(ROWS, COLS, figsize=(22, 11),
                             facecolor="#080808")
    fig.suptitle("ECG Waveform Atlas — 15 Cardiac Conditions",
                 fontsize=16, fontweight="bold", color="white",
                 fontfamily="monospace", y=0.97)

    axes_flat = axes.flatten()

    for idx, w in enumerate(WAVEFORMS):
        ax = axes_flat[idx]
        ax.set_facecolor("#071207")

        sig = w["fn"]()
        N   = len(sig)
        t   = np.linspace(0, DURATION, N)
        mask = t <= window
        tw, sw = t[mask], sig[mask]

        ax.plot(tw, sw, color=w["cat_color"], alpha=0.15, lw=4)
        ax.plot(tw, sw, color=w["cat_color"], alpha=0.9,  lw=0.9)

        # Grid
        for gx in np.arange(0, window+1, 1):
            ax.axvline(gx, color="#0d3d0d", lw=0.4)
        ax.axhline(0, color="#1a5c1a", lw=0.5)

        ax.set_xlim(0, window)
        ax.set_ylim(-1.6, 1.8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#1a1a1a")

        # Severity dot
        sc = w["sev_color"]
        ax.add_patch(plt.Circle((0.07, 0.88), 0.055,
                                transform=ax.transAxes,
                                color=sc, zorder=5, clip_on=False))

        ax.set_title(w["name"], color="white", fontsize=7.5,
                     fontfamily="monospace", pad=4, fontweight="bold")
        ax.text(0.5, -0.02, f"HR: {w['hr']}",
                transform=ax.transAxes, ha="center",
                color="#555", fontsize=6.5, fontfamily="monospace")

    # Hide unused axes
    for idx in range(len(WAVEFORMS), ROWS*COLS):
        axes_flat[idx].set_visible(False)

    # Legend
    sev_map = [
        ("NORMAL",    "#2ecc71"),
        ("MILD",      "#3498db"),
        ("WARNING",   "#e67e22"),
        ("ARRHYTHMIA","#9b59b6"),
        ("CRITICAL",  "#e74c3c"),
        ("FATAL",     "#c0392b"),
    ]
    handles = [mpatches.Patch(facecolor=c, label=s, edgecolor="none")
               for s, c in sev_map]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               frameon=False, labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, 0.005))

    plt.savefig("ecg_atlas_all15.png", dpi=160,
                bbox_inches="tight", facecolor="#080808")
    plt.show()
    print("💾 Saved: ecg_atlas_all15.png")

print("✅ Overview atlas plotter ready")


# ─────────────────────────────────────────────────────────────
# CELL 6 — Category Comparison Strips
# ─────────────────────────────────────────────────────────────

def plot_category(category_name, window=5.0):
    """
    Side-by-side ECG strips for all conditions in a category.
    """
    group = [w for w in WAVEFORMS if w["category"] == category_name]
    if not group:
        print(f"Category '{category_name}' not found.")
        return

    n = len(group)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.2 * n),
                              facecolor="#0a0a0a", squeeze=False)
    fig.suptitle(f"Category: {category_name}", fontsize=15,
                 fontweight="bold", color="white",
                 fontfamily="monospace", y=1.01)

    for i, w in enumerate(group):
        ax = axes[i][0]
        ax.set_facecolor("#071207")

        sig = w["fn"]()
        t   = np.linspace(0, DURATION, len(sig))
        msk = t <= window
        tw, sw = t[msk], sig[msk]

        # Grid
        for gx in np.arange(0, window+0.2, 0.2):
            ax.axvline(gx, color="#0a2a0a", lw=0.35)
        for gx in np.arange(0, window+1, 1.0):
            ax.axvline(gx, color="#0d3d0d", lw=0.7)
        ax.axhline(0, color="#1a5c1a", lw=0.7)

        ax.plot(tw, sw, color=w["cat_color"], alpha=0.18, lw=5)
        ax.plot(tw, sw, color=w["cat_color"], alpha=0.90, lw=1.1)

        ax.set_xlim(0, window); ax.set_ylim(-1.6, 1.8)
        ax.set_ylabel(w["name"], color="white", fontsize=9.5,
                      fontfamily="monospace", fontweight="bold")
        ax.tick_params(colors="#444", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#222")

        # HR label
        ax.text(0.99, 0.93, f"HR: {w['hr']}  |  {w['severity']}",
                transform=ax.transAxes, ha="right", va="top",
                color=w["sev_color"], fontsize=8, fontfamily="monospace")
        ax.text(0.01, 0.08, w["key"],
                transform=ax.transAxes, va="bottom",
                color="#555", fontsize=7.5, style="italic")

    axes[-1][0].set_xlabel("Time (s)", color="#888", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"ecg_category_{category_name.replace(' ','_')}.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.show()
    print(f"💾 Saved: ecg_category_{category_name}.png")


CATEGORIES = sorted(set(w["category"] for w in WAVEFORMS))
print("✅ Category plotter ready")
print("   Available categories:", CATEGORIES)


# ─────────────────────────────────────────────────────────────
# CELL 7 — Annotated Waveform Anatomy (Normal SR)
# ─────────────────────────────────────────────────────────────

def plot_anatomy():
    """
    Zoomed-in labelled diagram of one ECG beat,
    showing P, Q, R, S, T, PR interval, QRS complex, QT interval.
    """
    sig = gen_normal(hr=65)
    t   = np.linspace(0, DURATION, len(sig))
    # Pick the beat at ~1.2 s
    s, e = int(0.8*FS), int(2.2*FS)
    tw, sw = t[s:e], sig[s:e]
    tw = tw - tw[0]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#071207")
    ax.set_facecolor("#071207")

    # Grid
    for gx in np.arange(0, 1.4, 0.04): ax.axvline(gx, color="#0a2a0a", lw=0.4)
    for gx in np.arange(0, 1.4, 0.2):  ax.axvline(gx, color="#0d3d0d", lw=0.8)
    for gy in np.arange(-0.5,1.6,0.1): ax.axhline(gy, color="#0a2a0a", lw=0.4)
    for gy in np.arange(-0.5,1.6,0.5): ax.axhline(gy, color="#0d3d0d", lw=0.8)
    ax.axhline(0, color="#1a5c1a", lw=1.0)

    ax.plot(tw, sw, "#00e676", alpha=0.2, lw=6)
    ax.plot(tw, sw, "#00e676", alpha=0.9, lw=1.5)

    # Approximate key points from gaussian params at hr=65
    rr = 60/65
    P_t, P_v   = 0.12,  0.15
    Q_t, Q_v   = 0.22, -0.09
    R_t, R_v   = 0.265, 1.20
    S_t, S_v   = 0.305,-0.30
    T_t, T_v   = 0.50,  0.28
    Tend_t      = 0.62

    lc = "#ffd700"
    arrowprops = dict(arrowstyle="-|>", color=lc, lw=0.8,
                      mutation_scale=10)

    def ann(ax, label, xy, xytext, valign="center"):
        ax.annotate(label, xy=xy, xytext=xytext,
                    color=lc, fontsize=10.5, fontfamily="monospace",
                    fontweight="bold", va=valign,
                    arrowprops=arrowprops)

    ann(ax, "P",  (P_t, P_v),   (P_t-0.07, P_v+0.22))
    ann(ax, "Q",  (Q_t, Q_v),   (Q_t-0.06, Q_v-0.28))
    ann(ax, "R",  (R_t, R_v),   (R_t+0.05, R_v+0.18))
    ann(ax, "S",  (S_t, S_v),   (S_t+0.05, S_v-0.24))
    ann(ax, "T",  (T_t, T_v),   (T_t+0.06, T_v+0.20))

    # Interval brackets (below baseline)
    def bracket(ax, x1, x2, y, label, color="#88ccff"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
        ax.text((x1+x2)/2, y-0.08, label, ha="center",
                color=color, fontsize=8.5, fontfamily="monospace")

    bracket(ax, P_t-0.04, Q_t-0.01, -0.40, "PR interval\n160 ms", "#88ccff")
    bracket(ax, Q_t-0.01, S_t+0.01, -0.55, "QRS\n80 ms",           "#ffaa44")
    bracket(ax, Q_t-0.01, Tend_t,   -0.70, "QT interval  380 ms",  "#ff88aa")

    # ST segment label
    st_mid = (S_t + T_t) / 2
    ax.text(st_mid, 0.06, "ST segment", ha="center",
            color="#aaffaa", fontsize=8.5, fontfamily="monospace")

    # Isoelectric line label
    ax.text(0.02, 0.025, "Isoelectric\nbaseline",
            color="#1a5c1a", fontsize=8, fontfamily="monospace")

    ax.set_xlim(0.0, 1.30)
    ax.set_ylim(-0.9, 1.55)
    ax.set_xlabel("Time (s)", color="#888", fontsize=10)
    ax.set_ylabel("Amplitude (mV)", color="#888", fontsize=10)
    ax.tick_params(colors="#555", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#222")
    ax.set_title("ECG Waveform Anatomy — Lead II, Normal Sinus Rhythm",
                 color="white", fontsize=13, fontfamily="monospace",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig("ecg_anatomy_labelled.png", dpi=160,
                bbox_inches="tight", facecolor="#071207")
    plt.show()
    print("💾 Saved: ecg_anatomy_labelled.png")

print("✅ Anatomy diagram ready")



