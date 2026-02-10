import streamlit as st
import json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")

# Auto refresh every 500ms
st_autorefresh(interval=500, key="refresh")

st.title("Real-Time Crowd Safety Dashboard")

# Load data
try:
    with open("live_data.json") as f:
        data = json.load(f)
except:
    data = {"count": 0, "density": 0, "status": "LOADING..."}

col1, col2 = st.columns([3,1])

with col1:
    st.image("latest_frame.jpg", channels="BGR", width='stretch')

with col2:
    st.metric("People Count", data["count"])
    st.metric("Density", f'{data["density"]:.5f}')
    
    if data["status"] == "SAFE":
        st.success("SAFE")
    elif data["status"] == "MODERATE":
        st.warning("MODERATE")
    else:
        st.error("DANGEROUS")

