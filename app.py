import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import run_engine
from io import BytesIO

st.title("DD-0102 Curvature Validation System")

FrontInhale_file = st.file_uploader("Front Inhale", type=["csv","xlsx"])
FrontExhale_file = st.file_uploader("Front Exhale", type=["csv","xlsx"])
SideInhale_file = st.file_uploader("Side Inhale", type=["csv","xlsx"])
SideExhale_file = st.file_uploader("Side Exhale", type=["csv","xlsx"])

if st.button("Run Analysis"):
    output = run_engine(
        FrontInhale_file,
        FrontExhale_file,
        SideInhale_file,
        SideExhale_file
    )

    st.success("Analysis Complete")

    # Raw curvature plot
    st.subheader("Raw Point Curvature")
    def load_raw(file):
        if file is None: return None
        if file.name.endswith(("xlsx","xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        if "Point Curvature (um-1)" not in df.columns:
            return None
        df["Index"] = np.arange(len(df))
        return df

    fig_raw = go.Figure()
    raw_data = {}
    datasets = {
        "Front Inhale": FrontInhale_file,
        "Front Exhale": FrontExhale_file,
        "Side Inhale": SideInhale_file,
        "Side Exhale": SideExhale_file
    }

    for label, file in datasets.items():
        df = load_raw(file)
        if df is not None:
            raw_data[label] = df
            fig_raw.add_trace(go.Scatter(x=df["Index"], y=df["Point Curvature (um-1)"], mode="lines", name=label))

    st.plotly_chart(fig_raw, use_container_width=True)

    # Overlay
    st.subheader("Computed vs Raw Overlay")
    fig_overlay = go.Figure()

    for label, df in raw_data.items():
        fig_overlay.add_trace(go.Scatter(x=df["Index"], y=df["Point Curvature (um-1)"], mode="lines", name=f"{label} Raw", opacity=0.6))

        if label in output["plot_ca"]:
            fig = output["plot_ca"][label]
            for trace in fig.data:
                fig_overlay.add_trace(trace)

    st.plotly_chart(fig_overlay, use_container_width=True)

    # Error + threshold
    st.subheader("Error + Threshold")
    threshold = 0.88
    fig_err = go.Figure()

    validation_rows = []

    for label, df in raw_data.items():
        if label not in output["plot_ca"]: continue

        comp = output["plot_ca"][label].data[0].y
        x_comp = np.arange(len(comp))
        x_interp = np.linspace(0, len(comp)-1, len(df))
        comp_interp = np.interp(x_interp, x_comp, comp)

        error = np.abs(comp_interp - df["Point Curvature (um-1)"])

        fig_err.add_trace(go.Scatter(x=df["Index"], y=error, mode="lines", name=f"{label} Error"))

        for i in range(len(error)):
            val = error[i]
            if val <= 0.02: status = "PASS"
            elif val <= 0.05: status = "WARNING"
            else: status = "FAIL"

            validation_rows.append({
                "Dataset": label,
                "Index": i,
                "Computed": comp_interp[i],
                "Raw": df["Point Curvature (um-1)"].iloc[i],
                "Error": val,
                "Status": status
            })

    fig_err.add_hline(y=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig_err, use_container_width=True)

    # Export
    df_report = pd.DataFrame(validation_rows)

    if not df_report.empty:
        st.subheader("Download Validation Report")
        st.download_button("Download CSV", df_report.to_csv(index=False), "validation.csv")

        from openpyxl import Workbook
        from openpyxl.styles import PatternFill

        wb = Workbook()
        ws = wb.active
        ws.append(list(df_report.columns))

        for row in df_report.itertuples(index=False):
            ws.append(list(row))

        bio = BytesIO()
        wb.save(bio)

        st.download_button("Download Excel", bio.getvalue(), "validation.xlsx")
