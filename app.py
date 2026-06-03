
import io
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_engine import (
    infer_coordinate_columns,
    normalize_dataframe,
    summarize_dataset,
    curvature_profile,
    pairwise_alignment_summary,
    export_report_workbook,
)

st.set_page_config(page_title="4-File Graph & Diagram Builder", layout="wide")

st.title("4-File Excel Upload Graph & Diagram Builder")
st.caption(
    "Upload up to four Excel or CSV files. The app automatically infers coordinate columns, builds multiple graphs/diagrams, and lets you download a consolidated Excel report."
)

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader(
        "Choose exactly 4 Excel/CSV files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )
    pair_mode = st.selectbox(
        "Comparison mode",
        [
            "Automatic sequential pairs (1-2 and 3-4)",
            "All pairwise comparisons",
        ],
        index=0,
    )
    smooth_window = st.slider("Curvature smoothing window", 1, 15, 3, 2)
    show_markers = st.checkbox("Show point markers on line plots", value=True)


def read_table(file_obj):
    name = file_obj.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file_obj)
    return pd.read_excel(file_obj, engine='openpyxl' if name.endswith('xlsx') else None)


if not uploaded_files:
    st.info("Upload four files to generate the webpage visuals and downloadable report.")
    st.stop()

if len(uploaded_files) != 4:
    st.warning(f"You uploaded {len(uploaded_files)} file(s). This app is designed for 4 files so the full comparison dashboard can be built.")

raw_tables = {}
normalized_tables = {}
metadata = []

for up in uploaded_files:
    df = read_table(up)
    coords = infer_coordinate_columns(df)
    norm = normalize_dataframe(df, source_name=up.name, coord_cols=coords)
    raw_tables[up.name] = df
    normalized_tables[up.name] = norm
    metadata.append({
        'file': up.name,
        'row_count': len(df),
        'column_count': len(df.columns),
        'inferred_coordinates': ', '.join(coords) if coords else 'Not found',
    })

meta_df = pd.DataFrame(metadata)

st.subheader("Upload Summary")
st.dataframe(meta_df, use_container_width=True)

# ------ overview metrics ------
summary_frames = []
for name, df in normalized_tables.items():
    summary_frames.append(summarize_dataset(df, smooth_window=smooth_window))
summary_df = pd.DataFrame(summary_frames)

st.subheader("Dataset Metrics")
st.dataframe(summary_df, use_container_width=True)

# ------ overview charts ------
color_map = {name: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)] for i, name in enumerate(normalized_tables.keys())}

left, right = st.columns(2)
with left:
    fig2d = go.Figure()
    for name, df in normalized_tables.items():
        if {'x', 'y'}.issubset(df.columns):
            mode = 'lines+markers' if show_markers else 'lines'
            fig2d.add_trace(go.Scatter(
                x=df['x'], y=df['y'], mode=mode, name=name,
                line=dict(width=3, color=color_map[name]),
                marker=dict(size=5),
                text=[f"Index {i}" for i in range(len(df))],
                hovertemplate=f"{name}<br>x=%{{x}}<br>y=%{{y}}<extra></extra>"
            ))
    fig2d.update_layout(title='2D Overlay Trajectory Diagram', xaxis_title='X', yaxis_title='Y', height=520)
    st.plotly_chart(fig2d, use_container_width=True)

with right:
    has_z = any('z' in df.columns for df in normalized_tables.values())
    if has_z:
        fig3d = go.Figure()
        for name, df in normalized_tables.items():
            if {'x', 'y', 'z'}.issubset(df.columns):
                mode = 'lines+markers' if show_markers else 'lines'
                fig3d.add_trace(go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['z'], mode=mode, name=name,
                    line=dict(width=6, color=color_map[name]),
                    marker=dict(size=3),
                    hovertemplate=f"{name}<br>x=%{{x}}<br>y=%{{y}}<br>z=%{{z}}<extra></extra>"
                ))
        fig3d.update_layout(title='3D Overlay Trajectory Diagram', height=520)
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("No Z column was detected, so the 3D overlay diagram is skipped.")

# ------ per-file graphs ------
st.subheader("Per-File Graphs & Diagrams")
for name, df in normalized_tables.items():
    s1, s2 = st.columns(2)
    with s1:
        sub = go.Figure()
        mode = 'lines+markers' if show_markers else 'lines'
        sub.add_trace(go.Scatter(x=df['point_index'], y=df['x'], name='X', mode=mode))
        sub.add_trace(go.Scatter(x=df['point_index'], y=df['y'], name='Y', mode=mode))
        if 'z' in df.columns:
            sub.add_trace(go.Scatter(x=df['point_index'], y=df['z'], name='Z', mode=mode))
        sub.update_layout(title=f'{name} Coordinate Profile', xaxis_title='Point Index', yaxis_title='Coordinate Value', height=420)
        st.plotly_chart(sub, use_container_width=True)
    with s2:
        curv = curvature_profile(df, smooth_window=smooth_window)
        cfig = go.Figure()
        cfig.add_trace(go.Scatter(x=curv['point_index'], y=curv['curvature'], mode=mode, name='Curvature'))
        cfig.add_trace(go.Scatter(x=curv['point_index'], y=curv['step_length'], mode=mode, name='Step Length'))
        cfig.update_layout(title=f'{name} Curvature / Step-Length Profile', xaxis_title='Point Index', height=420)
        st.plotly_chart(cfig, use_container_width=True)

# ------ pair comparisons ------
st.subheader("Comparisons")
file_names = list(normalized_tables.keys())
if pair_mode.startswith('Automatic') and len(file_names) >= 4:
    pairs = [(file_names[0], file_names[1]), (file_names[2], file_names[3])]
else:
    pairs = list(combinations(file_names, 2))

pair_summaries = []
for a, b in pairs:
    comp = pairwise_alignment_summary(normalized_tables[a], normalized_tables[b], a, b)
    pair_summaries.append(comp)

pair_df = pd.DataFrame(pair_summaries)
st.dataframe(pair_df, use_container_width=True)

for pair in pair_summaries:
    a, b = pair['file_a'], pair['file_b']
    st.markdown(f"#### {a} vs {b}")
    ca, cb = st.columns(2)
    with ca:
        fig = go.Figure()
        dfa, dfb = normalized_tables[a], normalized_tables[b]
        mode = 'lines+markers' if show_markers else 'lines'
        fig.add_trace(go.Scatter(x=dfa['x'], y=dfa['y'], mode=mode, name=a, line=dict(width=3)))
        fig.add_trace(go.Scatter(x=dfb['x'], y=dfb['y'], mode=mode, name=b, line=dict(width=3)))
        fig.update_layout(title='Overlay XY Comparison', xaxis_title='X', yaxis_title='Y', height=420)
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Scatter(
            x=pair['distance_profile']['point_index'],
            y=pair['distance_profile']['distance'],
            mode=mode,
            name='Point-to-point distance'
        ))
        dist_fig.update_layout(title='Point-to-Point Distance Profile', xaxis_title='Aligned Point Index', yaxis_title='Distance', height=420)
        st.plotly_chart(dist_fig, use_container_width=True)

# ------ downloads ------
report_buffer = export_report_workbook(normalized_tables, summary_df, pair_df)

st.subheader("Download")
st.download_button(
    label="Download consolidated Excel report",
    data=report_buffer.getvalue(),
    file_name="four_file_visual_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

csv_buffer = io.StringIO()
summary_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download summary metrics CSV",
    data=csv_buffer.getvalue(),
    file_name="four_file_summary_metrics.csv",
    mime="text/csv",
)
