import pandas as pd
import numpy as np
import plotly.graph_objects as go

def run_engine(front_inhl, front_exhl, side_inhl, side_exhl):
    output = {"plot_ca": {}}

    def process(file, label):
        if file is None: return
        try:
            if file.name.endswith(("xlsx", "xls")):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            df.columns = [c.strip() for c in df.columns]
            if "Point Curvature (um-1)" not in df.columns: return
            df["Index"] = np.arange(len(df))
            step = max(1, int(len(df) / 20))
            segments = df.iloc[::step]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=segments["Index"], y=segments["Point Curvature (um-1)"], mode="lines+markers", name="Computed"))
            output["plot_ca"][label] = fig
        except:
            pass

    process(front_inhl, "Front Inhale")
    process(front_exhl, "Front Exhale")
    process(side_inhl, "Side Inhale")
    process(side_exhl, "Side Exhale")

    return output
