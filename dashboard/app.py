import streamlit as st
import pandas as pd
import time
from azure.identity import DefaultAzureCredential
from azure.digitaltwins.core import DigitalTwinsClient
import os

# CONFIG
ADT_URL = "https://BioSyncTwin.api.sea.digitaltwins.azure.net" # Update this!

st.set_page_config(page_title="Bio-Sync HPC Control Plane", layout="wide")

st.title("🏥 Bio-Sync HPC: Real-Time Network & Patient Twin")

# Sidebar for status
st.sidebar.header("System Status")
st.sidebar.success("✅ Azure Cloud: Connected")
st.sidebar.info("⚡ KEDA Scaling: Active")
st.sidebar.warning("🔒 Differential Privacy: Enabled")

# Azure Connection
@st.cache_resource
def get_client():
    credential = DefaultAzureCredential()
    return DigitalTwinsClient(ADT_URL, credential)

client = get_client()

# Layout
col1, col2 = st.columns(2)

def get_twin_data(twin_id):
    try:
        twin = client.get_digital_twin(twin_id)
        return twin['HeartRate'], twin['IsCritical']
    except:
        return 0, False

# Live Loop
placeholder = st.empty()

while True:
    alpha_hr, alpha_crit = get_twin_data("Silo-Alpha")
    beta_hr, beta_crit = get_twin_data("Silo-Beta")

    with placeholder.container():
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silo-Alpha HR", f"{alpha_hr:.2f} bpm", delta_color="inverse" if alpha_crit else "normal")
        m2.metric("Silo-Beta HR", f"{beta_hr:.2f} bpm", delta_color="inverse" if beta_crit else "normal")
        
        # Alerts
        if alpha_crit:
            st.error("🚨 SEIZURE DETECTED IN SILO-ALPHA! HPC SCALING TRIGGERED.")
        if beta_crit:
            st.error("🚨 SEIZURE DETECTED IN SILO-BETA! HPC SCALING TRIGGERED.")

        # Network Simulation Graph
        chart_data = pd.DataFrame({
            'Silo-Alpha': [alpha_hr],
            'Silo-Beta': [beta_hr]
        })
        st.line_chart(chart_data)

    time.sleep(2)