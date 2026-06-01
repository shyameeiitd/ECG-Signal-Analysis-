import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ECG Atlas",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 ECG Atlas")
st.subheader("Interactive ECG Learning Platform")

condition = st.selectbox(
    "Select ECG Condition",
    [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation",
        "STEMI",
        "Ventricular Tachycardia"
    ]
)

duration = st.slider(
    "Duration (seconds)",
    2,
    8,
    5
)

if st.button("Generate ECG"):

    t = np.linspace(0,duration,1000)

    signal = np.sin(2*np.pi*2*t)

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(t,signal)

    ax.set_title(condition)

    ax.grid(True)

    st.pyplot(fig)

    st.success(f"Generated: {condition}")
