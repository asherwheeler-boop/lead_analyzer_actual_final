import streamlit as st
import io
import plotly.graph_objects as go
import pandas as pd

from pacing_lead_curvature_webapp import run_web_analysis
from dd0102_database import get_all_segment_data

st.set_page_config(layout="wide")
st.title("Pacing Lead Analyzer")

mode = st.radio("Mode",["Single","Segment Comparison vs DD-0102"])

# ===================
# SINGLE MODE
# ===================
if mode == "Single":

    fi = st.file_uploader("Front Inhale")
    si = st.file_uploader("Side Inhale")
    fe = st.file_uploader("Front Exhale")
    se = st.file_uploader("Side Exhale")

    if all([fi,si,fe,se]):
        if st.button("Run Analysis"):
            out = run_web_analysis({
                "front_inhale":io.BytesIO(fi.getvalue()),
                "side_inhale":io.BytesIO(si.getvalue()),
                "front_exhale":io.BytesIO(fe.getvalue()),
                "side_exhale":io.BytesIO(se.getvalue())
            })
            st.dataframe(out['table'])

# ===================
# COMPARISON MODE
# ===================
elif mode == "Segment Comparison vs DD-0102":

    dd = get_all_segment_data()
    patient = st.selectbox("Patient",sorted(dd['Patient'].unique()))

    fi = st.file_uploader("Front Inhale",key='c1')
    si = st.file_uploader("Side Inhale",key='c2')
    fe = st.file_uploader("Front Exhale",key='c3')
    se = st.file_uploader("Side Exhale",key='c4')

    if all([fi,si,fe,se]):
        if st.button("Run Comparison"):

            out = run_web_analysis({
                "front_inhale":io.BytesIO(fi.getvalue()),
                "side_inhale":io.BytesIO(si.getvalue()),
                "front_exhale":io.BytesIO(fe.getvalue()),
                "side_exhale":io.BytesIO(se.getvalue())
            })

            df_calc = out['table'].copy()
            df_calc['Segment']=range(1,len(df_calc)+1)

            df_dd = dd[dd['Patient']==patient]

            merged = df_calc.merge(df_dd,left_on='Segment',right_on='Seg_Reindexed',how='left')
            merged['Difference']=merged['Max_Ca']-merged['Ca_reported']

            st.dataframe(merged)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=merged['Max_Ca'],mode='lines+markers',name='Calculated'))
            fig.add_trace(go.Scatter(y=merged['Ca_reported'],mode='lines+markers',name='DD0102'))
            st.plotly_chart(fig, use_container_width=True)
