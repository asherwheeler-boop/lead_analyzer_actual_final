"""
3D Pacing Lead Curvature Amplitude Analyzer - Web App Engine v2
"""
from __future__ import annotations
import io, re, os, zipfile, tempfile, datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks

N_INTERP = 1000
SG_WINDOW = 21
SG_POLYORDER = 3
UNIT_FACTORS = {"um": 1e4, "mm": 1e1, "cm": 1.0, "pixels": 1.0}
UNIT_LABELS = {"um": "\u00b5m", "mm": "mm", "cm": "cm", "pixels": "px"}
DEFAULT_SAFETY = {"VisONE Stimulation": 0.88, "Respiration": 0.91}
COLOR_PAIRS = [
    ("#0077BB", "#33BBEE"), ("#009988", "#EE7733"),
    ("#AA3377", "#BBBBBB"), ("#332288", "#88CCEE"),
    ("#117733", "#999933"),
]
PEAK_COLOR = "#FF0000"

MATERIAL_PROPERTIES = {
    "MP35N": {"description": "Cobalt-Nickel alloy", "fatigue_coeff": 0.59, "fatigue_exp": -0.12, "youngs_modulus_gpa": 233, "yield_strain": 0.008},
    "DFT": {"description": "Drawn Filled Tube", "fatigue_coeff": 0.45, "fatigue_exp": -0.11, "youngs_modulus_gpa": 186, "yield_strain": 0.006},
    "Elgiloy": {"description": "Cobalt-Chromium-Nickel alloy", "fatigue_coeff": 0.52, "fatigue_exp": -0.115, "youngs_modulus_gpa": 221, "yield_strain": 0.007},
    "Nitinol": {"description": "Nickel-Titanium shape memory alloy", "fatigue_coeff": 0.80, "fatigue_exp": -0.09, "youngs_modulus_gpa": 75, "yield_strain": 0.010},
}


def _hex_to_rgba(hx, alpha=0.15):
    h = hx.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _arc_length_2d(pts):
    d = np.diff(pts, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])


def _drop_duplicate_points(pts, tol=1e-9, log=None):
    if len(pts) < 2:
        return pts
    keep = [True] * len(pts)
    prev = pts[0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - prev) <= tol:
            keep[i] = False
        else:
            prev = pts[i]
    cleaned = pts[np.array(keep)]
    nd = len(pts) - len(cleaned)
    if nd and log is not None:
        log.append(f"  [clean] Dropped {nd} duplicate point(s).")
    return cleaned


def _read_file_to_dataframe(source, log):
    if isinstance(source, (str, Path)):
        p = Path(source)
        ext = p.suffix.lower()
        log.append(f"  Reading file: {p.name}")
        if ext in (".xlsx", ".xls"):
            eng = "openpyxl" if ext == ".xlsx" else "xlrd"
            return pd.read_excel(p, engine=eng)
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(p, encoding=enc, skipinitialspace=True)
            except Exception:
                continue
        raise ValueError(f"Cannot read {p.name}")
    buf = source
    buf.seek(0)
    raw = buf.read()
    buf.seek(0)
    try:
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception:
        pass
    txt = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            pass
    if txt is None:
        raise ValueError("Cannot decode uploaded file.")
    for sep in (",", "\t", ";"):
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep, skipinitialspace=True)
            if len(df.columns) >= 3:
                return df
        except Exception:
            pass
    raise ValueError("Cannot parse uploaded CSV.")


def _find_coordinate_columns(df, curve_col, log):
    cands = [c for c in df.columns if c != curve_col]
    xc = [c for c in cands if "x-coordinate" in c.lower()]
    yc = [c for c in cands if "y-coordinate" in c.lower()]
    if xc and yc:
        log.append(f"  Coord cols (P1): {xc[0]}, {yc[0]}")
        return xc[0], yc[0]
    xp = re.compile(r"x[\s_-]?coord", re.I)
    yp = re.compile(r"y[\s_-]?coord", re.I)
    xc = [c for c in cands if xp.search(c)]
    yc = [c for c in cands if yp.search(c)]
    if xc and yc:
        log.append(f"  Coord cols (P2): {xc[0]}, {yc[0]}")
        return xc[0], yc[0]
    xc = [c for c in cands if c.strip().upper() == "X"]
    yc = [c for c in cands if c.strip().upper() == "Y"]
    if xc and yc:
        log.append(f"  Coord cols (P3): {xc[0]}, {yc[0]}")
        return xc[0], yc[0]
    num_cols = []
    for c in cands:
        try:
            v = pd.to_numeric(df[c], errors="raise")
            if v.nunique() > 10:
                num_cols.append(c)
        except Exception:
            pass
        if len(num_cols) == 2:
            break
    if len(num_cols) >= 2:
        log.append(f"  Coord cols (P4): {num_cols[0]}, {num_cols[1]}")
        return num_cols[0], num_cols[1]
    raise ValueError(f"Cannot find coordinate columns. Candidates: {cands}")


def parse_input_file(source, log=None):
    if log is None:
        log = []
    df = _read_file_to_dataframe(source, log)
    df.columns = [c.strip() for c in df.columns]
    curve_col = None
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str).str.strip().str.upper()
            if "curvature" in c.lower():
                continue
            if s.str.startswith("CURVE").any():
                curve_col = c
                break
    if curve_col is None:
        for c in df.columns:
            if df[c].dtype == object and "curvature" not in c.lower():
                curve_col = c
                break
    if curve_col is None:
        curve_col = df.columns[0]
    log.append(f"  Curve-name column: '{curve_col}'")
    x_col, y_col = _find_coordinate_columns(df, curve_col, log)
    df[curve_col] = df[curve_col].astype(str).str.strip().str.upper()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])
    curves = {}
    for name, grp in df.groupby(curve_col, sort=False):
        pts = grp[[x_col, y_col]].to_numpy(dtype=np.float64)
        pts = _drop_duplicate_points(pts, log=log)
        if len(pts) < 4:
            log.append(f"  Curve '{name}': only {len(pts)} pts - skipped.")
            continue
        curves[name] = pts
        log.append(f"  Curve '{name}': {len(pts)} clean points")
    return curves


def reconstruct_3d_wire(front_pts, side_pts, n=N_INTERP):
    sf = _arc_length_2d(front_pts); sf /= sf[-1]
    ss = _arc_length_2d(side_pts); ss /= ss[-1]
    t = np.linspace(0, 1, n)
    return np.column_stack([CubicSpline(sf, front_pts[:, 0])(t),
                            CubicSpline(sf, front_pts[:, 1])(t),
                            CubicSpline(ss, side_pts[:, 0])(t)])


def _smooth_wire(wire, window=SG_WINDOW, polyorder=SG_POLYORDER):
    win = min(window, len(wire) - 1)
    if win % 2 == 0:
        win -= 1
    win = max(win, polyorder + 1)
    return np.column_stack([savgol_filter(wire[:, i], win, polyorder) for i in range(3)])


def compute_curvature_3d(wire, dt=1.0):
    rp = np.gradient(wire, dt, axis=0)
    rpp = np.gradient(rp, dt, axis=0)
    cross = np.cross(rp, rpp)
    num = np.linalg.norm(cross, axis=1)
    den = np.linalg.norm(rp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(den > 1e-12, num / den**3, 0.0)
    return k


def analyse_wire(front_pts, side_pts, label="", log=None):
    if log and label:
        log.append(f"  Reconstructing {label} ...")
    wire_raw = reconstruct_3d_wire(front_pts, side_pts)
    wire = _smooth_wire(wire_raw)
    k = compute_curvature_3d(wire)
    diffs = np.diff(wire, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    return wire, k, arc


def compute_curvature_amplitude(k_in, arc_in, k_ex, arc_ex, n=N_INTERP):
    s_min = max(arc_in[0], arc_ex[0])
    s_max = min(arc_in[-1], arc_ex[-1])
    if s_max <= s_min:
        raise ValueError("Inhale/Exhale arc-length ranges do not overlap.")
    s = np.linspace(s_min, s_max, n)
    ki = np.nan_to_num(CubicSpline(arc_in, k_in, extrapolate=False)(s), nan=0.0)
    ke = np.nan_to_num(CubicSpline(arc_ex, k_ex, extrapolate=False)(s), nan=0.0)
    return s, np.abs(ke - ki) / 2.0


def _interpolate_3d_point(wire, arc, s_target):
    xyz = []
    for axis in range(3):
        cs = CubicSpline(arc, wire[:, axis], extrapolate=True)
        xyz.append(float(cs(s_target)))
    return np.array(xyz)


def find_ca_peaks(Ca, s_common, prominence_frac=0.10):
    ca_range = float(Ca.max() - Ca.min())
    prom = max(prominence_frac * ca_range, 1e-12)
    idxs, _ = find_peaks(Ca, prominence=prom)
    idx_global = int(np.argmax(Ca))
    idx_set = set(idxs.tolist())
    idx_set.add(idx_global)
    return [(i, float(s_common[i]), float(Ca[i])) for i in sorted(idx_set)]


def get_safety_status(max_ca, safety_thresholds=None):
    if safety_thresholds is None:
        safety_thresholds = DEFAULT_SAFETY
    thresh_sorted = sorted(safety_thresholds.values())
    if len(thresh_sorted) >= 2:
        if max_ca >= thresh_sorted[1]:
            return "FAIL"
        elif max_ca >= thresh_sorted[0]:
            return "WARNING"
    elif len(thresh_sorted) == 1:
        if max_ca >= thresh_sorted[0]:
            return "FAIL"
    return "PASS"


def estimate_fatigue_life(ca_cm, wire_od_mm, material_name="MP35N"):
    mat = MATERIAL_PROPERTIES.get(material_name)
    if mat is None:
        return {"error": "Unknown material: " + material_name}
    od_cm = wire_od_mm / 10.0
    bending_strain = ca_cm * (od_cm / 2.0)
    eps_f = mat["fatigue_coeff"]
    c = mat["fatigue_exp"]
    if bending_strain <= 0:
        cycles = float("inf")
        fatigue_status = "NO STRAIN"
    elif bending_strain >= eps_f:
        cycles = 0.0
        fatigue_status = "IMMEDIATE FAILURE"
    else:
        two_n = (bending_strain / eps_f) ** (1.0 / c)
        cycles = two_n / 2.0
        if cycles < 4e5:
            fatigue_status = "HIGH RISK (< 400k cycles)"
        elif cycles < 4e7:
            fatigue_status = f"MODERATE ({round(cycles / 1e6, 1)}M cycles)"
        else:
            fatigue_status = "LOW RISK (> 40M cycles)"
    return {"material": material_name, "wire_od_mm": wire_od_mm, "ca_cm_inv": ca_cm,
            "bending_strain": round(bending_strain, 6), "estimated_cycles": cycles,
            "fatigue_status": fatigue_status, "formula": "N = 0.5 * (eps_a / eps_f')^(1/c)",
            "eps_f": eps_f, "c_exp": c}


def generate_calculation_breakdown(cname, res, unit_label, input_unit):
    ca_u = "cm\u207b\u00b9"
    ul = unit_label
    scale = UNIT_FACTORS.get(input_unit, 1.0)
    Ca = res["Ca_cm"]
    ts = res.get("trim_start", 0)
    te = res.get("trim_end", len(Ca))
    Ca_t = Ca[ts:te]
    max_v = float(Ca_t.max()) if len(Ca_t) > 0 else 0.0
    mean_v = float(Ca_t.mean()) if len(Ca_t) > 0 else 0.0
    return [
        {"step": "1. Input Parsing", "desc": "Read 2D coordinate traces from front and side X-ray views.",
         "detail": f"Inhale points: {res['n_pts_inhale']} | Exhale points: {res['n_pts_exhale']}"},
        {"step": "2. Arc-length Parameterization", "desc": "Compute cumulative arc length for each 2D trace.",
         "formula": "s_i = sum sqrt(dx^2 + dy^2)", "detail": "Normalized to [0,1] for cubic spline fitting."},
        {"step": "3. 3D Wire Reconstruction", "desc": "Combine front (X,Y) and side (Z) via cubic spline on shared parameter t.",
         "formula": "wire(t) = [X_front(t), Y_front(t), X_side(t)]",
         "detail": f"Inhale 3D len: {round(res['length_inhale_3d'],2)} {ul} | Exhale 3D len: {round(res['length_exhale_3d'],2)} {ul} | {N_INTERP} pts"},
        {"step": "4. Savitzky-Golay Smoothing", "desc": "Reduce digitization noise.",
         "formula": f"SavGol(wire, window={SG_WINDOW}, order={SG_POLYORDER})", "detail": "Applied to X, Y, Z independently."},
        {"step": "5. 3D Curvature (Frenet-Serret)", "desc": "Local curvature via cross-product formula.",
         "formula": "kappa = |r' x r''| / |r'|^3", "detail": f"Units: {ul}^-1"},
        {"step": "6. Curvature Amplitude (Ca)", "desc": "Half-range of curvature change between respiratory states.",
         "formula": "Ca(s) = |kappa_exhale(s) - kappa_inhale(s)| / 2",
         "detail": f"Resampled onto common {N_INTERP}-point arc grid."},
        {"step": "7. Unit Conversion", "desc": f"Scale from {input_unit} to cm^-1.",
         "formula": f"Ca [cm^-1] = Ca [{ul}^-1] * {scale}", "detail": f"Scale: {scale}"},
        {"step": "8. Endpoint Trimming", "desc": "Exclude noisy electrode tips.",
         "detail": f"Active range: [{ts}:{te}] of {len(Ca)} total."},
        {"step": "9. Peak Detection", "desc": "scipy find_peaks with 10% prominence.",
         "detail": f"Peaks: {len(res.get('all_peaks',[]))} | Max Ca={round(max_v,4)} {ca_u} | Mean Ca={round(mean_v,4)} {ca_u}"},
    ]


def generate_3d_plotly(results, unit_label="\u00b5m"):
    import plotly.graph_objects as go
    fig = go.Figure()
    for i, (cname, res) in enumerate(sorted(results.items())):
        c_in, c_ex = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        w_in, w_ex = res["wire_inhale"], res["wire_exhale"]
        fig.add_trace(go.Scatter3d(x=w_in[:,0], y=w_in[:,1], z=w_in[:,2],
            mode="lines", name=cname+" Inhale", line=dict(width=5, color=c_in)))
        fig.add_trace(go.Scatter3d(x=w_ex[:,0], y=w_ex[:,1], z=w_ex[:,2],
            mode="lines", name=cname+" Exhale", line=dict(width=5, color=c_ex, dash="dash")))
        all_peaks = res.get("all_peaks", [])
        if all_peaks:
            px_in = [p["xyz_inhale"][0] for p in all_peaks]
            py_in = [p["xyz_inhale"][1] for p in all_peaks]
            pz_in = [p["xyz_inhale"][2] for p in all_peaks]
            ht = [f"Ca={round(p['ca'],2)} cm^-1<br>Arc={round(p['s'],2)} {unit_label}" for p in all_peaks]
            fig.add_trace(go.Scatter3d(x=px_in, y=py_in, z=pz_in, mode="markers",
                marker=dict(size=6, color=PEAK_COLOR, line=dict(width=1, color="white")),
                hovertext=ht, hoverinfo="text", name=cname+" Peaks"))
    fig.update_layout(title="3-D Pacing Lead Reconstruction",
        scene=dict(aspectmode="data", xaxis_title=f"X ({unit_label})", yaxis_title=f"Y ({unit_label})", zaxis_title=f"Z ({unit_label})"),
        height=750, margin=dict(l=0,r=0,t=50,b=0))
    return fig


def generate_3d_heatmap(results, unit_label="\u00b5m"):
    import plotly.graph_objects as go
    fig = go.Figure()
    for cname, res in sorted(results.items()):
        for tag, wk, kk in [("Inhale","wire_inhale","k_inhale"),("Exhale","wire_exhale","k_exhale")]:
            w, k = res[wk], res[kk]
            fig.add_trace(go.Scatter3d(x=w[:,0], y=w[:,1], z=w[:,2], mode="lines",
                line=dict(color=k, colorscale="Jet", width=6, cmin=0, cmax=float(np.percentile(k,99)),
                          showscale=(tag=="Inhale")), name=f"{cname} {tag}"))
    fig.update_layout(title="Curvature Heatmap", scene=dict(aspectmode="data"),
        height=750, margin=dict(l=0,r=0,t=50,b=0))
    return fig


def generate_morph_animation(results, unit_label="\u00b5m", n_frames=30):
    import plotly.graph_objects as go
    figs = {}
    for cname, res in sorted(results.items()):
        w_in, w_ex = res["wire_inhale"], res["wire_exhale"]
        n = min(len(w_in), len(w_ex))
        w_in, w_ex = w_in[:n], w_ex[:n]
        alphas = np.linspace(0, 1, n_frames)
        alphas = np.concatenate([alphas, alphas[::-1]])
        frames = []
        for idx, a in enumerate(alphas):
            w = w_in*(1-a) + w_ex*a
            lab = "Inhale" if a < 0.5 else "Exhale"
            frames.append(go.Frame(
                data=[go.Scatter3d(x=w[:,0],y=w[:,1],z=w[:,2],mode="lines",line=dict(width=5,color="#0077BB"))],
                name=str(idx), layout=go.Layout(title_text=f"{cname} - {lab} (t={round(a,2)})")))
        fig = go.Figure(
            data=[go.Scatter3d(x=w_in[:,0],y=w_in[:,1],z=w_in[:,2],mode="lines",line=dict(width=5,color="#0077BB"),name="Wire")],
            frames=frames)
        fig.update_layout(scene=dict(aspectmode="data"), height=700, margin=dict(l=0,r=0,t=50,b=0),
            updatemenus=[dict(type="buttons", showactive=False, y=0.05, x=0.05, buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=80,redraw=True),fromcurrent=True)]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0,redraw=False),mode="immediate")])])])
        figs[cname] = fig
    return figs


def generate_ca_plotly(results, unit_label="\u00b5m", safety_thresholds=None):
    if safety_thresholds is None:
        safety_thresholds = DEFAULT_SAFETY
    import plotly.graph_objects as go
    figs = {}
    thresh_list = sorted(safety_thresholds.items(), key=lambda x: x[1])
    for i, (cname, res) in enumerate(sorted(results.items())):
        c_in, _ = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        s, Ca = res["s_common"], res["Ca_cm"]
        ts, te = res.get("trim_start",0), res.get("trim_end",len(Ca))
        Ca_act = Ca[ts:te]
        mean_ca = float(Ca_act.mean()) if te>ts else 0.0
        max_ca = float(Ca_act.max()) if te>ts else 0.0
        idx_max = ts + int(np.argmax(Ca_act)) if te>ts else 0
        fig = go.Figure()
        if ts > 0:
            fig.add_trace(go.Scatter(x=s[:ts+1], y=Ca[:ts+1], mode="lines",
                line=dict(color="rgba(180,180,180,0.4)", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(200,200,200,0.08)", name="Trimmed", hoverinfo="skip"))
        if te < len(Ca):
            fig.add_trace(go.Scatter(x=s[te-1:], y=Ca[te-1:], mode="lines",
                line=dict(color="rgba(180,180,180,0.4)", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(200,200,200,0.08)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s[ts:te], y=Ca[ts:te], mode="lines",
            line=dict(color=c_in, width=3), fill="tozeroy", fillcolor=_hex_to_rgba(c_in, 0.12),
            name=f"Ca ({cname})"))
        all_peaks = res.get("all_peaks", [])
        if all_peaks:
            fig.add_trace(go.Scatter(x=[p["s"] for p in all_peaks], y=[p["ca"] for p in all_peaks],
                mode="markers", marker=dict(size=10, color=PEAK_COLOR, symbol="diamond"), name="Peaks"))
        colors_t = ["#FF0000","#FF8C00","#FFD700","#999999"]
        for ti, (tn, tv) in enumerate(thresh_list):
            fig.add_hline(y=tv, line_dash="dash", line_color=colors_t[ti%len(colors_t)], line_width=1.5,
                annotation_text=f"{tn} ({tv})", annotation_position="top right")
        fig.add_hline(y=mean_ca, line_dash="dot", line_color="rgba(100,100,100,0.5)", line_width=1.5,
            annotation_text=f"Mean={round(mean_ca,4)}", annotation_position="top left")
        fig.add_annotation(x=float(s[idx_max]), y=max_ca, text=f"<b>Peak={round(max_ca,4)}</b>",
            showarrow=True, arrowhead=2, arrowcolor=PEAK_COLOR, ax=40, ay=-35,
            font=dict(size=11, color=PEAK_COLOR), bgcolor="rgba(255,255,255,0.85)")
        fig.update_layout(title=f"Curvature Amplitude - {cname}",
            xaxis_title=f"Arc Length ({unit_label})", yaxis_title="Ca (cm^-1)",
            plot_bgcolor="white", height=450, hovermode="x unified")
        figs[cname] = fig
    return figs


def generate_results_table(results, unit_label="\u00b5m", safety_thresholds=None):
    rows = []
    for cname, res in results.items():
        Ca = res["Ca_cm"]; s = res["s_common"]
        ts, te = res.get("trim_start",0), res.get("trim_end",len(Ca))
        Ca_t = Ca[ts:te]; s_t = s[ts:te]
        idx = int(np.argmax(Ca_t)); mx = float(Ca_t.max())
        rows.append({"Curve": cname, "Status": get_safety_status(mx, safety_thresholds),
            "Max Ca (cm^-1)": round(mx,6), "Mean Ca (cm^-1)": round(float(Ca_t.mean()),6),
            f"Arc at Max ({unit_label})": round(float(s_t[idx]),2),
            f"Total Arc ({unit_label})": round(float(s[-1]),2),
            f"Inhale Len ({unit_label})": round(res["length_inhale_3d"],2),
            f"Exhale Len ({unit_label})": round(res["length_exhale_3d"],2),
            "Peaks": len(res.get("all_peaks",[]))})
    return pd.DataFrame(rows)


def generate_html_report(output):
    ul = output.get("unit_label", "um")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = output.get("summary_cards", [])
    notes = output.get("patient_notes", "")
    safety = output.get("safety_thresholds", DEFAULT_SAFETY)
    html = ["<!DOCTYPE html><html><head><title>Pacing Lead Report</title>"]
    html.append("<style>body{font-family:Arial;max-width:960px;margin:40px auto;color:#333}")
    html.append("h1{color:#0077BB;border-bottom:2px solid #0077BB;padding-bottom:8px}")
    html.append("table{border-collapse:collapse;width:100%;margin:15px 0}")
    html.append("th,td{border:1px solid #ddd;padding:8px 12px;text-align:left}")
    html.append("th{background:#f0f0f0}.pass{color:#009988;font-weight:bold}")
    html.append(".warning{color:#EE7733;font-weight:bold}.fail{color:#FF0000;font-weight:bold}")
    html.append(".notes{background:#fafafa;border-left:4px solid #0077BB;padding:12px;margin:20px 0;font-style:italic}")
    html.append("</style></head><body>")
    html.append(f"<h1>Pacing Lead Curvature Analysis Report</h1>")
    html.append(f"<p>Generated: {now}<br>Units: {ul}</p>")
    thresh_str = "<br>".join([f"{k} = {v} cm^-1" for k, v in safety.items()])
    html.append(f"<p><b>Safety Thresholds:</b><br>{thresh_str}</p>")
    if notes:
        html.append(f'<div class="notes"><b>Patient Notes:</b> {notes}</div>')
    for card in cards:
        st_css = card["status"].lower()
        html.append(f'<h2>{card["name"]} <span class="{st_css}">[{card["status"]}]</span></h2>')
        html.append("<table><tr><th>Metric</th><th>Value</th></tr>")
        html.append(f"<tr><td>Peak Ca</td><td>{round(card['max_ca'],4)} cm^-1</td></tr>")
        html.append(f"<tr><td>Mean Ca</td><td>{round(card['mean_ca'],4)} cm^-1</td></tr>")
        html.append(f"<tr><td>Inhale 3D Length</td><td>{round(card['inhale_length'],2)} {ul}</td></tr>")
        html.append(f"<tr><td>Exhale 3D Length</td><td>{round(card['exhale_length'],2)} {ul}</td></tr>")
        html.append(f"<tr><td>Arc at Peak</td><td>{round(card['arc_at_max'],2)} {ul}</td></tr>")
        html.append(f"<tr><td>Total Arc</td><td>{round(card['total_arc'],2)} {ul}</td></tr>")
        html.append(f"<tr><td>Peaks</td><td>{card['n_peaks']}</td></tr>")
        fat = card.get("fatigue")
        if fat and "estimated_cycles" in fat:
            cyc = fat["estimated_cycles"]
            cyc_s = "inf" if cyc == float("inf") else f"{int(cyc):,}"
            html.append(f"<tr><td>Strain</td><td>{fat['bending_strain']}</td></tr>")
            html.append(f"<tr><td>Fatigue Life</td><td>{cyc_s} cycles ({fat['fatigue_status']})</td></tr>")
        html.append("</table>")
    html.append("</body></html>")
    return "\n".join(html)


def run_web_analysis(files_dict, input_unit="um", trim_start_pct=0.0, trim_end_pct=0.0,
                     safety_thresholds=None, patient_notes="",
                     wire_od_mm=2.0, material_name="MP35N"):
    if safety_thresholds is None:
        safety_thresholds = dict(DEFAULT_SAFETY)
    log = []
    results = {}
    ul = UNIT_LABELS.get(input_unit.lower(), input_unit)
    try:
        log.append("=" * 60)
        log.append("STEP 1 - Parsing input files")
        log.append("=" * 60)
        cfi = parse_input_file(files_dict["front_inhale"], log)
        csi = parse_input_file(files_dict["side_inhale"], log)
        cfe = parse_input_file(files_dict["front_exhale"], log)
        cse = parse_input_file(files_dict["side_exhale"], log)
        common = cfi.keys() & csi.keys() & cfe.keys() & cse.keys()
        if not common:
            raise ValueError("No matching curve names across the four files.")
        log.append(f"  Matched curves: {sorted(common)}")
        for cname in sorted(common):
            log.append("-" * 60)
            log.append(f"  Processing: {cname}")
            fi, si, fe, se = cfi[cname], csi[cname], cfe[cname], cse[cname]
            wire_in, k_in, arc_in = analyse_wire(fi, si, "Inhale", log)
            wire_ex, k_ex, arc_ex = analyse_wire(fe, se, "Exhale", log)
            len_in, len_ex = float(arc_in[-1]), float(arc_ex[-1])
            scale = UNIT_FACTORS.get(input_unit.lower(), 1.0)
            s_common, Ca_raw = compute_curvature_amplitude(k_in, arc_in, k_ex, arc_ex)
            Ca_cm = Ca_raw * scale
            n = len(Ca_cm)
            trim_s = int(n * trim_start_pct / 100.0)
            trim_e = n - int(n * trim_end_pct / 100.0)
            trim_e = max(trim_e, trim_s + 1)
            Ca_trimmed = Ca_cm[trim_s:trim_e]
            max_ca = float(Ca_trimmed.max())
            mean_ca = float(Ca_trimmed.mean())
            idx_max_t = int(np.argmax(Ca_trimmed))
            s_at_max = float(s_common[trim_s + idx_max_t])
            status = get_safety_status(max_ca, safety_thresholds)
            log.append(f"  Max Ca: {round(max_ca,6)} cm^-1 [{status}]")
            peaks_raw = find_ca_peaks(Ca_trimmed, s_common[trim_s:trim_e])
            all_peaks = []
            for pidx, ps, pca in peaks_raw:
                pxyz_in = _interpolate_3d_point(wire_in, arc_in, ps)
                pxyz_ex = _interpolate_3d_point(wire_ex, arc_ex, ps)
                all_peaks.append({"idx": pidx+trim_s, "s": ps, "ca": pca,
                                  "xyz_inhale": pxyz_in, "xyz_exhale": pxyz_ex})
            fat = estimate_fatigue_life(max_ca, wire_od_mm, material_name)
            results[cname] = {
                "wire_inhale": wire_in, "wire_exhale": wire_ex,
                "k_inhale": k_in, "k_exhale": k_ex,
                "s_common": s_common, "Ca_cm": Ca_cm,
                "length_inhale_3d": len_in, "length_exhale_3d": len_ex,
                "n_pts_inhale": len(fi), "n_pts_exhale": len(fe),
                "all_peaks": all_peaks, "peak_s": s_at_max, "peak_ca": max_ca,
                "trim_start": trim_s, "trim_end": trim_e, "fatigue": fat}
        table = generate_results_table(results, ul, safety_thresholds)
        plot_3d = generate_3d_plotly(results, ul)
        plot_heatmap = generate_3d_heatmap(results, ul)
        plot_ca = generate_ca_plotly(results, ul, safety_thresholds)
        plot_morph = generate_morph_animation(results, ul)
        summary_cards = []
        for cn in sorted(results.keys()):
            r = results[cn]
            Ca = r["Ca_cm"]; t_s, t_e = r["trim_start"], r["trim_end"]
            Ca_t = Ca[t_s:t_e]; sv = r["s_common"]; mx = float(Ca_t.max())
            summary_cards.append({
                "name": cn, "max_ca": round(mx,4), "mean_ca": round(float(Ca_t.mean()),4),
                "inhale_length": round(r["length_inhale_3d"],2), "exhale_length": round(r["length_exhale_3d"],2),
                "arc_at_max": round(float(sv[t_s+int(np.argmax(Ca_t))]),2),
                "total_arc": round(float(sv[-1]),2), "n_peaks": len(r["all_peaks"]),
                "status": get_safety_status(mx, safety_thresholds), "fatigue": r.get("fatigue")})
        calc_breakdowns = {}
        for cn in sorted(results.keys()):
            calc_breakdowns[cn] = generate_calculation_breakdown(cn, results[cn], ul, input_unit)
    except Exception as exc:
        log.append(f"*** ERROR: {exc}")
        import traceback; log.append(traceback.format_exc())
        table = pd.DataFrame()
        plot_3d = plot_heatmap = plot_ca = plot_morph = None
        summary_cards = []; calc_breakdowns = {}
    out = {"table": table, "plot_3d": plot_3d, "plot_heatmap": plot_heatmap,
           "plot_ca": plot_ca, "plot_morph": plot_morph, "raw_results": results, "log": log,
           "summary_cards": summary_cards, "unit_label": ul, "patient_notes": patient_notes,
           "safety_thresholds": safety_thresholds, "calc_breakdowns": calc_breakdowns}
    out["html_report"] = generate_html_report(out) if summary_cards else ""
    return out


def _match_files_in_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".csv",".xlsx",".xls"))]
    mapping = {"front_inhale": None, "side_inhale": None, "front_exhale": None, "side_exhale": None}
    for fname in files:
        fl = fname.lower()
        is_front = "front" in fl
        is_side = "side" in fl or "right" in fl or "lat" in fl
        is_inhale = "inhal" in fl or "inhl" in fl
        is_exhale = "exhal" in fl or "exhl" in fl
        if is_front and is_inhale: mapping["front_inhale"] = os.path.join(folder, fname)
        elif is_front and is_exhale: mapping["front_exhale"] = os.path.join(folder, fname)
        elif is_side and is_inhale: mapping["side_inhale"] = os.path.join(folder, fname)
        elif is_side and is_exhale: mapping["side_exhale"] = os.path.join(folder, fname)
    return mapping


def run_batch_from_zip(zip_bytes, input_unit="um", trim_start_pct=0.0, trim_end_pct=0.0,
                       safety_thresholds=None, wire_od_mm=2.0, material_name="MP35N"):
    all_cards, errors = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "upload.zip")
        with open(zpath, "wb") as zf:
            zf.write(zip_bytes)
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(tmpdir)
        folders = []
        for item in os.listdir(tmpdir):
            ipath = os.path.join(tmpdir, item)
            if os.path.isdir(ipath) and item != "__MACOSX":
                m = _match_files_in_folder(ipath)
                if all(m.values()): folders.append((item, m))
        if not folders:
            m = _match_files_in_folder(tmpdir)
            if all(m.values()): folders.append(("root", m))
        for fn, fd in sorted(folders):
            try:
                out = run_web_analysis(fd, input_unit, trim_start_pct, trim_end_pct, safety_thresholds, "", wire_od_mm, material_name)
                for card in out.get("summary_cards", []):
                    card["patient"] = fn; all_cards.append(card)
            except Exception as e:
                errors.append(f"{fn}: {e}")
    return {"master_table": pd.DataFrame(all_cards) if all_cards else pd.DataFrame(), "errors": errors}


def run_comparison(files_a, files_b, label_a="Lead A", label_b="Lead B",
                   input_unit="um", trim_start_pct=0.0, trim_end_pct=0.0,
                   safety_thresholds=None, wire_od_mm=2.0, material_name="MP35N"):
    out_a = run_web_analysis(files_a, input_unit, trim_start_pct, trim_end_pct, safety_thresholds, "", wire_od_mm, material_name)
    out_b = run_web_analysis(files_b, input_unit, trim_start_pct, trim_end_pct, safety_thresholds, "", wire_od_mm, material_name)
    rows = []
    for c in out_a.get("summary_cards",[]): c["lead_set"]=label_a; rows.append(c)
    for c in out_b.get("summary_cards",[]): c["lead_set"]=label_b; rows.append(c)
    return {"lead_a": out_a, "lead_b": out_b, "comparison_table": pd.DataFrame(rows), "label_a": label_a, "label_b": label_b}
