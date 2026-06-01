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
st.subheader("Interactive ECG Learning Platform")

# Mapping of conditions to generator functions
conditions = {
    "Normal Sinus Rhythm": gen_normal,
    "Sinus Bradycardia": gen_bradycardia,
    "Sinus Tachycardia": gen_tachycardia,
    "Atrial Fibrillation": gen_afib,
    "Atrial Flutter": gen_flutter,
    "1° AV Block": gen_av_block_1,
    "2° AV Block (Mobitz II)": gen_av_block_2,
    "STEMI": gen_stemi,
    "NSTEMI / Ischaemia": gen_nstemi,
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

st.write("### Selected Condition")
st.success(condition)

# Debug information
st.write("Generator Function:", conditions[condition].__name__)

if st.button("Generate ECG"):

    # Call the selected ECG generator
    signal = conditions[condition]()

    # Time axis
    t = np.linspace(0, DURATION, len(signal))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t, signal, linewidth=1.2)

    ax.set_title(condition)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True)

    st.pyplot(fig)

    # Download CSV
    csv_data = "\n".join(
        [f"{x},{y}" for x, y in zip(t, signal)]
    )

    st.download_button(
        label="Download ECG CSV",
        data=csv_data,
        file_name=f"{condition}.csv",
        mime="text/csv"
    )

st.info(
    "Educational and research tool only. "
    "Not intended for medical diagnosis."
)
