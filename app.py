import streamlit as st
import pandas as pd
from dd0102_database import get_all_segment_data
from pacing_lead_curvature_webapp import run_web_analysis

st.title("Segment Comparison Tool")

mode=st.radio("Mode",["Comparison"])

patient=st.selectbox("Patient",["1011"])

if st.button("Run Comparison"):
    calc=run_web_analysis({})['table']
    calc['Segment']=range(1,len(calc)+1)
    dd=get_all_segment_data()
    merged=calc.merge(dd,left_on='Segment',right_on='Seg_Reindexed')
    st.dataframe(merged)
