import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from ecg_generator import *

st.set_page_config(
    page_title="ECG Atlas",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 ECG Atlas")
st.markdown("### Interactive ECG Learning Platform")

conditions = {
    "Normal Sinus Rhythm": gen_normal,
    "Sinus Bradycardia": gen_bradycardia,
    "Sinus Tachycardia": gen_tachycardia,
    "Atrial Fibrillation": gen_afib,
    "Atrial Flutter": gen_flutter,
    "1° AV Block": gen_av_block_1,
    "2° AV Block (Mobitz II)": gen_av_block_2,
    "STEMI": gen_stemi,
    "NSTEMI": gen_nstemi,
    "Ventricular Tachycardia": gen_vtach,
    "Ventricular Fibrillation": gen_vfib,
    "Left BBB": gen_lbbb,
    "Right BBB": gen_rbbb,
    "Wolff-Parkinson-White": gen_wpw,
    "Long QT Syndrome": gen_long_qt
}

condition = st.selectbox(
    "Select ECG Condition",
    list(conditions.keys())
)

if st.button("Generate ECG"):

    signal = conditions[condition]()

    t = np.linspace(
        0,
        DURATION,
        len(signal)
    )

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(t, signal, linewidth=1.2)

    ax.set_title(condition)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")

    ax.grid(True)

    st.pyplot(fig)

st.info(
    "Educational and research tool only. "
    "Not intended for clinical diagnosis."
)
