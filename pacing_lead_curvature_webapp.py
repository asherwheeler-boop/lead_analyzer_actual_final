"""
3D Pacing Lead Curvature Amplitude Analyzer - Web App Engine v2
Features: safety thresholds, trim, batch, fatigue estimation,
curvature heatmap, morph animation, calculation breakdown,
configurable thresholds, patient notes, comparison mode.
"""
from __future__ import annotations
import io, re, os, zipfile, tempfile, datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks

# ═══════════════ CONSTANTS ═══════════════
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
    "MP35N": {
        "description": "Cobalt-Nickel alloy (common pacing lead conductor)",
        "fatigue_coeff": 0.59, "fatigue_exp": -0.12,
        "youngs_modulus_gpa": 233, "yield_strain": 0.008,
    },
    "DFT": {
        "description": "Drawn Filled Tube (silver-core composite)",
        "fatigue_coeff": 0.45, "fatigue_exp": -0.11,
        "youngs_modulus_gpa": 186, "yield_strain": 0.006,
    },
    "Elgiloy": {
        "description": "Cobalt-Chromium-Nickel alloy",
        "fatigue_coeff": 0.52, "fatigue_exp": -0.115,
        "youngs_modulus_gpa": 221, "yield_strain": 0.007,
    },
    "Nitinol": {
        "description": "Nickel-Titanium shape memory alloy",
        "fatigue_coeff": 0.80, "fatigue_exp": -0.09,
        "youngs_modulus_gpa": 75, "yield_strain": 0.010,
    },
}

# ═══════════════ HELPERS ═══════════════

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
        log.append(f"      [clean] Dropped {nd} duplicate point(s).")
    return cleaned

def _read_file_to_dataframe(source, log):
    if isinstance(source, (str, Path)):
        p = Path(source)
        ext = p.suffix.lower()
        log.append(f"  Reading file: {p.name}")
        if ext in (".xlsx", ".xls"):
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            return pd.read_excel(p, engine=engine)
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
        log.append(f"      Coord cols (P1): {xc[0]}, {yc[0]}")
        return xc[0], yc[0]
    xp = re.compile(r"x[\s_-]?coord", re.I)
    yp = re.compile(r"y[\s_-]?coord", re.I)
    xc = [c for c in cands if xp.search(c)]
    yc = [c for c in cands if yp.search(c)]
    if xc and yc:
        log.append(f"      Coord cols (P2): {xc[0]}, {yc[0]}")
        return xc[0], yc[0]
    xc = [c for c in cands if c.strip().upper() == "X"]
    yc = [c for c in cands if c.strip().upper() == "Y"]
    if xc and yc:
        log.append(f"      Coord cols (P3): {xc[0]}, {yc[0]}")
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
        log.append(f"      Coord cols (P4): {num_cols[0]}, {num_cols[1]}")
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
    log.append(f"      Curve-name column: '{curve_col}'")
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
            log.append(f"      Curve '{name}': only {len(pts)} pts - skipped.")
            continue
        curves[name] = pts
        log.append(f"      Curve '{name}': {len(pts)} clean points")
    return curves

# ═══════════════ CORE MATH ═══════════════

def reconstruct_3d_wire(front_pts, side_pts, n=N_INTERP):
    sf = _arc_length_2d(front_pts); sf /= sf[-1]
    ss = _arc_length_2d(side_pts);  ss /= ss[-1]
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
        log.append(f"    Reconstructing {label} ...")
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

# ═══════════════ FATIGUE ESTIMATION ═══════════════

def estimate_fatigue_life(ca_cm, wire_od_mm, material_name="MP35N"):
    """Basquin strain-life: eps_a = eps_f' * (2Nf)^c => Nf = 0.5*(eps_a/eps_f')^(1/c)"""
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
            fatigue_status = "MODERATE (" + str(round(cycles / 1e6, 1)) + "M cycles)"
        else:
            fatigue_status = "LOW RISK (> 40M cycles)"
    return {
        "material": material_name, "wire_od_mm": wire_od_mm,
        "ca_cm_inv": ca_cm, "bending_strain": round(bending_strain, 6),
        "estimated_cycles": cycles, "fatigue_status": fatigue_status,
        "formula": "N = 0.5 * (eps_a / eps_f')^(1/c)",
        "eps_f": eps_f, "c_exp": c,
    }

# ═══════════════ CALCULATION BREAKDOWN ═══════════════

def generate_calculation_breakdown(cname, res, unit_label, input_unit):
    ca_u = "cm\u207b\u00b9"
    ul = unit_label
    scale = UNIT_FACTORS.get(input_unit, 1.0)
    Ca = res["Ca_cm"]
    ts = res.get("trim_start", 0)
    te = res.get("trim_end", len(Ca))
    Ca_t = Ca[ts:te]
    steps = []
    steps.append({
        "step": "1. Input Parsing",
        "desc": "Read 2D coordinate traces from front and side X-ray views for inhale and exhale.",
        "detail": "Inhale points: " + str(res["n_pts_inhale"]) + " | Exhale points: " + str(res["n_pts_exhale"]),
    })
    steps.append({
        "step": "2. Arc-length Parameterization",
        "desc": "Compute cumulative arc length s_i = \u03a3 \u221a(dx\u00b2 + dy\u00b2) for each 2D trace.",
        "formula": "s_i = \u03a3_{j=1}^{i} \u221a(\u0394x_j\u00b2 + \u0394y_j\u00b2)",
        "detail": "Normalized to [0, 1] for cubic spline fitting.",
    })
    steps.append({
        "step": "3. 3D Wire Reconstruction",
        "desc": "Combine front-view (X, Y) and side-view (Z) via cubic spline interpolation on shared parameter t.",
        "formula": "wire(t) = [ X_front(t),  Y_front(t),  X_side(t) ],   t \u2208 [0, 1]",
        "detail": "Inhale 3D length: " + str(round(res["length_inhale_3d"], 2)) + " " + ul
                  + " | Exhale 3D length: " + str(round(res["length_exhale_3d"], 2)) + " " + ul
                  + " | " + str(N_INTERP) + " interpolation points.",
    })
    steps.append({
        "step": "4. Savitzky-Golay Smoothing",
        "desc": "Reduce digitization noise before differentiation.",
        "formula": "wire_smooth = SavGol(wire,  window=" + str(SG_WINDOW) + ",  order=" + str(SG_POLYORDER) + ")",
        "detail": "Applied independently to each X, Y, Z coordinate.",
    })
    steps.append({
        "step": "5. 3D Curvature (Frenet-Serret)",
        "desc": "Local curvature at every point via the cross-product formula.",
        "formula": "\u03ba = |r\u2032 \u00d7 r\u2033| / |r\u2032|\u00b3",
        "detail": "r\u2032 and r\u2033 computed with numpy.gradient(). Units: " + ul + "\u207b\u00b9.",
    })
    steps.append({
        "step": "6. Curvature Amplitude (Ca)",
        "desc": "Half-range of curvature change between respiratory states.",
        "formula": "Ca(s) = |\u03ba_exhale(s) \u2212 \u03ba_inhale(s)| / 2",
        "detail": "Inhale and exhale curvature resampled onto a common " + str(N_INTERP) + "-point arc-length grid.",
    })
    steps.append({
        "step": "7. Unit Conversion to cm\u207b\u00b9",
        "desc": "Scale raw curvature from input coordinate units to cm\u207b\u00b9.",
        "formula": "Ca [cm\u207b\u00b9] = Ca [" + ul + "\u207b\u00b9] \u00d7 " + str(scale),
        "detail": "Scale factor for " + input_unit + " \u2192 cm: " + str(scale) + ".",
    })
    steps.append({
        "step": "8. Endpoint Trimming",
        "desc": "Exclude noisy electrode tips / header regions from statistics.",
        "detail": "Active index range: [" + str(ts) + " : " + str(te) + "] of " + str(len(Ca)) + " total points.",
    })
    max_ca_val = float(Ca_t.max()) if len(Ca_t) > 0 else 0.0
    mean_ca_val = float(Ca_t.mean()) if len(Ca_t) > 0 else 0.0
    steps.append({
        "step": "9. Peak Detection & Results",
        "desc": "scipy.signal.find_peaks with prominence-based filtering (10 % of range).",
        "detail": "Peaks found: " + str(len(res.get("all_peaks", [])))
                  + " | Max Ca = " + str(round(max_ca_val, 4)) + " " + ca_u
                  + " | Mean Ca = " + str(round(mean_ca_val, 4)) + " " + ca_u,
    })
    return steps

# ═══════════════ PLOTLY VISUALIZATION ═══════════════

def generate_3d_plotly(results, unit_label="\u00b5m"):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    fig = go.Figure()
    for i, (cname, res) in enumerate(sorted(results.items())):
        c_in, c_ex = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        w_in, w_ex = res["wire_inhale"], res["wire_exhale"]
        fig.add_trace(go.Scatter3d(x=w_in[:, 0], y=w_in[:, 1], z=w_in[:, 2],
            mode="lines", name=cname + " Inhale", line=dict(width=5, color=c_in)))
        fig.add_trace(go.Scatter3d(x=w_ex[:, 0], y=w_ex[:, 1], z=w_ex[:, 2],
            mode="lines", name=cname + " Exhale", line=dict(width=5, color=c_ex, dash="dash")))
        fig.add_trace(go.Scatter3d(x=[w_in[0, 0]], y=[w_in[0, 1]], z=[w_in[0, 2]],
            mode="markers", marker=dict(size=5, color=c_in), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[w_ex[0, 0]], y=[w_ex[0, 1]], z=[w_ex[0, 2]],
            mode="markers", marker=dict(size=5, color=c_ex, symbol="diamond"), showlegend=False))
        all_peaks = res.get("all_peaks", [])
        if all_peaks:
            px_in = [p["xyz_inhale"][0] for p in all_peaks]
            py_in = [p["xyz_inhale"][1] for p in all_peaks]
            pz_in = [p["xyz_inhale"][2] for p in all_peaks]
            px_ex = [p["xyz_exhale"][0] for p in all_peaks]
            py_ex = [p["xyz_exhale"][1] for p in all_peaks]
            pz_ex = [p["xyz_exhale"][2] for p in all_peaks]
            ht = ["Ca=" + str(round(p["ca"], 2)) + " cm\u207b\u00b9<br>Arc=" + str(round(p["s"], 2)) + " " + unit_label for p in all_peaks]
            fig.add_trace(go.Scatter3d(x=px_in, y=py_in, z=pz_in, mode="markers",
                marker=dict(size=6, color=PEAK_COLOR, line=dict(width=1, color="white")),
                hovertext=ht, hoverinfo="text", name=cname + " Peaks (In)"))
            fig.add_trace(go.Scatter3d(x=px_ex, y=py_ex, z=pz_ex, mode="markers",
                marker=dict(size=6, color=PEAK_COLOR, symbol="diamond", line=dict(width=1, color="white")),
                hovertext=ht, hoverinfo="text", name=cname + " Peaks (Ex)"))
    fig.update_layout(
        title=dict(text="3-D Pacing Lead Reconstruction", font=dict(size=18)),
        scene=dict(aspectmode="data",
            xaxis_title="X (" + unit_label + ")",
            yaxis_title="Y (" + unit_label + ")",
            zaxis_title="Z (" + unit_label + ")"),
        legend=dict(font=dict(size=11)), height=750, margin=dict(l=0, r=0, t=50, b=0))
    return fig

def generate_3d_heatmap(results, unit_label="\u00b5m"):
    """3D wire colored by local curvature magnitude."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    fig = go.Figure()
    for cname, res in sorted(results.items()):
        for tag, wkey, kkey in [("Inhale", "wire_inhale", "k_inhale"), ("Exhale", "wire_exhale", "k_exhale")]:
            w = res[wkey]
            k = res[kkey]
            fig.add_trace(go.Scatter3d(
                x=w[:, 0], y=w[:, 1], z=w[:, 2], mode="lines",
                line=dict(color=k, colorscale="Jet", width=6, cmin=0, cmax=float(np.percentile(k, 99)),
                          showscale=(tag == "Inhale")),
                name=cname + " " + tag + " (curvature)"))
    fig.update_layout(
        title=dict(text="Curvature Heatmap", font=dict(size=18)),
        scene=dict(aspectmode="data",
            xaxis_title="X (" + unit_label + ")",
            yaxis_title="Y (" + unit_label + ")",
            zaxis_title="Z (" + unit_label + ")"),
        height=750, margin=dict(l=0, r=0, t=50, b=0))
    return fig

def generate_morph_animation(results, unit_label="\u00b5m", n_frames=30):
    """Animated interpolation between inhale and exhale wire positions."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    figs = {}
    for cname, res in sorted(results.items()):
        w_in = res["wire_inhale"]
        w_ex = res["wire_exhale"]
        n = min(len(w_in), len(w_ex))
        w_in = w_in[:n]
        w_ex = w_ex[:n]
        frames = []
        alphas = np.linspace(0, 1, n_frames)
        alphas = np.concatenate([alphas, alphas[::-1]])
        for idx, a in enumerate(alphas):
            w = w_in * (1.0 - a) + w_ex * a
            label = "Inhale" if a < 0.5 else "Exhale"
            frames.append(go.Frame(
                data=[go.Scatter3d(x=w[:, 0], y=w[:, 1], z=w[:, 2],
                    mode="lines", line=dict(width=5, color="#0077BB"))],
                name=str(idx),
                layout=go.Layout(title_text=cname + " \u2013 " + label + " (t=" + str(round(a, 2)) + ")")))
        fig = go.Figure(
            data=[go.Scatter3d(x=w_in[:, 0], y=w_in[:, 1], z=w_in[:, 2],
                mode="lines", line=dict(width=5, color="#0077BB"), name="Wire")],
            frames=frames)
        fig.update_layout(
            scene=dict(aspectmode="data",
                xaxis_title="X (" + unit_label + ")",
                yaxis_title="Y (" + unit_label + ")",
                zaxis_title="Z (" + unit_label + ")"),
            height=700, margin=dict(l=0, r=0, t=50, b=0),
            updatemenus=[dict(type="buttons", showactive=False, y=0.05, x=0.05,
                buttons=[
                    dict(label="\u25b6 Play", method="animate",
                        args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="\u23f8 Pause", method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
        figs[cname] = fig
    return figs

def generate_ca_plotly(results, unit_label="\u00b5m", safety_thresholds=None):
    if safety_thresholds is None:
        safety_thresholds = DEFAULT_SAFETY
    ca_unit = "cm\u207b\u00b9"
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    figs = {}
    thresh_list = sorted(safety_thresholds.items(), key=lambda x: x[1])
    for i, (cname, res) in enumerate(sorted(results.items())):
        c_in, _ = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        s, Ca = res["s_common"], res["Ca_cm"]
        ts = res.get("trim_start", 0)
        te = res.get("trim_end", len(Ca))
        Ca_active = Ca[ts:te]
        mean_ca = float(Ca_active.mean()) if te > ts else 0.0
        max_ca = float(Ca_active.max()) if te > ts else 0.0
        idx_max = ts + int(np.argmax(Ca_active)) if te > ts else 0
        all_peaks = res.get("all_peaks", [])
        fig = go.Figure()
        if ts > 0:
            fig.add_trace(go.Scatter(x=s[:ts + 1], y=Ca[:ts + 1], mode="lines",
                line=dict(color="rgba(180,180,180,0.4)", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(200,200,200,0.08)",
                name="Trimmed", showlegend=True, hoverinfo="skip"))
        if te < len(Ca):
            fig.add_trace(go.Scatter(x=s[te - 1:], y=Ca[te - 1:], mode="lines",
                line=dict(color="rgba(180,180,180,0.4)", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(200,200,200,0.08)",
                showlegend=False, hoverinfo="skip"))
        hover_tpl = "<b>Arc:</b> %{x:.2f} " + unit_label + "<br><b>Ca:</b> %{y:.4f} " + ca_unit + "<extra></extra>"
        fig.add_trace(go.Scatter(x=s[ts:te], y=Ca[ts:te], mode="lines",
            line=dict(color=c_in, width=3), fill="tozeroy",
            fillcolor=_hex_to_rgba(c_in, 0.12), hovertemplate=hover_tpl,
            name="Ca (" + cname + ")"))
        if all_peaks:
            pk_s = [p["s"] for p in all_peaks]
            pk_ca = [p["ca"] for p in all_peaks]
            fig.add_trace(go.Scatter(x=pk_s, y=pk_ca, mode="markers",
                marker=dict(size=10, color=PEAK_COLOR, symbol="diamond",
                    line=dict(width=2, color="white")),
                name="Peaks"))
        colors_thresh = ["#FF0000", "#FF8C00", "#FFD700", "#999999"]
        for ti, (tname, tval) in enumerate(thresh_list):
            tc = colors_thresh[ti % len(colors_thresh)]
            fig.add_hline(y=tval, line_dash="dash", line_color=tc, line_width=1.5,
                annotation_text=tname + " (" + str(tval) + ")",
                annotation_position="top right",
                annotation_font=dict(size=10, color=tc))
        fig.add_hline(y=mean_ca, line_dash="dot", line_color="rgba(100,100,100,0.5)",
            line_width=1.5, annotation_text="Mean = " + str(round(mean_ca, 4)),
            annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(80,80,80,0.8)"))
        fig.add_annotation(x=float(s[idx_max]), y=max_ca,
            text="<b>Peak = " + str(round(max_ca, 4)) + "</b>",
            showarrow=True, arrowhead=2, arrowcolor=PEAK_COLOR, ax=40, ay=-35,
            font=dict(size=11, color=PEAK_COLOR), bgcolor="rgba(255,255,255,0.85)",
            bordercolor=PEAK_COLOR, borderwidth=1, borderpad=4)
        fig.update_layout(
            title=dict(text="Curvature Amplitude \u2013 " + cname,
                font=dict(size=18, color="#333"), x=0.5, xanchor="center"),
            xaxis=dict(title="Arc Length (" + unit_label + ")",
                gridcolor="rgba(220,220,220,0.5)", zeroline=False),
            yaxis=dict(title="Ca (" + ca_unit + ")",
                gridcolor="rgba(220,220,220,0.5)", zeroline=False),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=True, height=450,
            margin=dict(l=60, r=30, t=60, b=50), hovermode="x unified")
        figs[cname] = fig
    return figs

def generate_results_table(results, unit_label="\u00b5m", safety_thresholds=None):
    rows = []
    for cname, res in results.items():
        Ca = res["Ca_cm"]
        s = res["s_common"]
        ts = res.get("trim_start", 0)
        te = res.get("trim_end", len(Ca))
        Ca_t = Ca[ts:te]
        s_t = s[ts:te]
        idx = int(np.argmax(Ca_t))
        mx = float(Ca_t.max())
        rows.append({
            "Curve": cname,
            "Status": get_safety_status(mx, safety_thresholds),
            "Max Ca (cm\u207b\u00b9)": round(mx, 6),
            "Mean Ca (cm\u207b\u00b9)": round(float(Ca_t.mean()), 6),
            "Arc at Max (" + unit_label + ")": round(float(s_t[idx]), 2),
            "Total Arc (" + unit_label + ")": round(float(s[-1]), 2),
            "Inhale Len (" + unit_label + ")": round(res["length_inhale_3d"], 2),
            "Exhale Len (" + unit_label + ")": round(res["length_exhale_3d"], 2),
            "Peaks": len(res.get("all_peaks", [])),
        })
    return pd.DataFrame(rows)

# ═══════════════ HTML REPORT (with patient notes) ═══════════════

def generate_html_report(output):
    ul = output.get("unit_label", "\u00b5m")
    ca_u = "cm\u207b\u00b9"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = output.get("summary_cards", [])
    notes = output.get("patient_notes", "")
    safety_thresholds = output.get("safety_thresholds", DEFAULT_SAFETY)
    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>Pacing Lead Analysis Report</title>")
    parts.append("<style>")
    parts.append("body{font-family:Arial,sans-serif;max-width:960px;margin:40px auto;color:#333}")
    parts.append("h1{color:#0077BB;border-bottom:2px solid #0077BB;padding-bottom:8px}")
    parts.append("h2{color:#444;margin-top:30px}")
    parts.append("table{border-collapse:collapse;width:100%;margin:15px 0}")
    parts.append("th,td{border:1px solid #ddd;padding:8px 12px;text-align:left}")
    parts.append("th{background:#f0f0f0;font-weight:bold}")
    parts.append(".pass{color:#009988;font-weight:bold}")
    parts.append(".warning{color:#EE7733;font-weight:bold}")
    parts.append(".fail{color:#FF0000;font-weight:bold}")
    parts.append(".notes{background:#fafafa;border-left:4px solid #0077BB;padding:12px 16px;margin:20px 0;font-style:italic}")
    parts.append("</style></head><body>")
    parts.append("<h1>Pacing Lead Curvature Analysis Report</h1>")
    parts.append("<p>Generated: " + now + " | Units: " + ul + " | Ca units: " + ca_u + "</p>")
    thresh_str = " | ".join([k + " = " + str(v) + " " + ca_u for k, v in safety_thresholds.items()])
    parts.append("<p><b>Safety Thresholds:</b> " + thresh_str + "</p>")
    if notes:
        parts.append("<div class='notes'><b>Patient Notes:</b> " + notes + "</div>")
    parts.append("<hr>")
    for card in cards:
        st_css = card["status"].lower()
        parts.append("<h2>" + card["name"] + " &mdash; <span class='" + st_css + "'>" + card["status"] + "</span></h2>")
        parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
        parts.append("<tr><td>Peak Ca</td><td class='" + st_css + "'>" + str(round(card["max_ca"], 4)) + " " + ca_u + "</td></tr>")
        parts.append("<tr><td>Mean Ca</td><td>" + str(round(card["mean_ca"], 4)) + " " + ca_u + "</td></tr>")
        parts.append("<tr><td>Inhale 3D Length</td><td>" + str(round(card["inhale_length"], 2)) + " " + ul + "</td></tr>")
        parts.append("<tr><td>Exhale 3D Length</td><td>" + str(round(card["exhale_length"], 2)) + " " + ul + "</td></tr>")
        parts.append("<tr><td>Arc at Peak Ca</td><td>" + str(round(card["arc_at_max"], 2)) + " " + ul + "</td></tr>")
        parts.append("<tr><td>Total Arc Length</td><td>" + str(round(card["total_arc"], 2)) + " " + ul + "</td></tr>")
        parts.append("<tr><td>Peaks Found</td><td>" + str(card["n_peaks"]) + "</td></tr>")
        fat = card.get("fatigue")
        if fat:
            parts.append("<tr><td>Bending Strain</td><td>" + str(fat["bending_strain"]) + "</td></tr>")
            cyc = fat["estimated_cycles"]
            cyc_str = "\u221e" if cyc == float("inf") else format(int(cyc), ",")
            parts.append("<tr><td>Est. Fatigue Life</td><td>" + cyc_str + " cycles (" + fat["fatigue_status"] + ")</td></tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)

# ═══════════════ ORCHESTRATOR ═══════════════

def run_web_analysis(files_dict, input_unit="um", trim_start_pct=0.0, trim_end_pct=0.0,
                     safety_thresholds=None, patient_notes="",
                     wire_od_mm=2.0, material_name="MP35N"):
    if safety_thresholds is None:
        safety_thresholds = dict(DEFAULT_SAFETY)
    log, results = [], {}
    ul = UNIT_LABELS.get(input_unit.lower(), input_unit)
    try:
        log.append("=" * 60)
        log.append("STEP 1 - Parsing & cleaning input files")
        log.append("=" * 60)
        cfi = parse_input_file(files_dict["front_inhale"], log)
        csi = parse_input_file(files_dict["side_inhale"], log)
        cfe = parse_input_file(files_dict["front_exhale"], log)
        cse = parse_input_file(files_dict["side_exhale"], log)
        common = cfi.keys() & csi.keys() & cfe.keys() & cse.keys()
        if not common:
            raise ValueError("No matching curve names across the four files.")
        log.append("")
        log.append("  Matched curves: " + str(sorted(common)))
        for cname in sorted(common):
            log.append("")
            log.append("-" * 60)
            log.append("  Processing: " + cname)
            log.append("-" * 60)
            fi, si = cfi[cname], csi[cname]
            fe, se = cfe[cname], cse[cname]
            log.append("")
            log.append("  STEP 2 & 3 - 3-D reconstruction + curvature")
            wire_in, k_in, arc_in = analyse_wire(fi, si, "Inhale", log)
            wire_ex, k_ex, arc_ex = analyse_wire(fe, se, "Exhale", log)
            len_in = float(arc_in[-1])
            len_ex = float(arc_ex[-1])
            log.append("    Inhale 3D length: " + str(round(len_in, 2)) + " " + ul)
            log.append("    Exhale 3D length: " + str(round(len_ex, 2)) + " " + ul)
            log.append("")
            log.append("  STEP 4 - Curvature Amplitude")
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
            log.append("    Trim: [" + str(trim_s) + ":" + str(trim_e) + "] of " + str(n))
            log.append("    Max Ca: " + str(round(max_ca, 6)) + " cm\u207b\u00b9  [" + status + "]")
            log.append("    Mean Ca: " + str(round(mean_ca, 6)) + " cm\u207b\u00b9")
            peaks_raw = find_ca_peaks(Ca_trimmed, s_common[trim_s:trim_e])
            all_peaks = []
            for pidx, ps, pca in peaks_raw:
                pxyz_in = _interpolate_3d_point(wire_in, arc_in, ps)
                pxyz_ex = _interpolate_3d_point(wire_ex, arc_ex, ps)
                all_peaks.append({"idx": pidx + trim_s, "s": ps, "ca": pca,
                    "xyz_inhale": pxyz_in, "xyz_exhale": pxyz_ex})
            log.append("    Peaks found: " + str(len(all_peaks)))
            fat = estimate_fatigue_life(max_ca, wire_od_mm, material_name)
            if "estimated_cycles" in fat:
                cyc = fat["estimated_cycles"]
                cyc_str = "\u221e" if cyc == float("inf") else format(int(cyc), ",")
                log.append("    Fatigue (" + material_name + ", OD=" + str(wire_od_mm) + "mm): "
                           + cyc_str + " cycles [" + fat["fatigue_status"] + "]")
            results[cname] = {
                "wire_inhale": wire_in, "wire_exhale": wire_ex,
                "k_inhale": k_in, "k_exhale": k_ex,
                "s_common": s_common, "Ca_cm": Ca_cm,
                "length_inhale_3d": len_in, "length_exhale_3d": len_ex,
                "n_pts_inhale": len(fi), "n_pts_exhale": len(fe),
                "all_peaks": all_peaks, "peak_s": s_at_max, "peak_ca": max_ca,
                "trim_start": trim_s, "trim_end": trim_e, "fatigue": fat,
            }
        table = generate_results_table(results, ul, safety_thresholds)
        plot_3d = generate_3d_plotly(results, ul)
        plot_heatmap = generate_3d_heatmap(results, ul)
        plot_ca = generate_ca_plotly(results, ul, safety_thresholds)
        plot_morph = generate_morph_animation(results, ul)
        summary_cards = []
        for cn in sorted(results.keys()):
            r = results[cn]
            Ca = r["Ca_cm"]
            t_s, t_e = r["trim_start"], r["trim_end"]
            Ca_t = Ca[t_s:t_e]
            sv = r["s_common"]
            mx = float(Ca_t.max())
            summary_cards.append({
                "name": cn, "max_ca": round(mx, 4),
                "mean_ca": round(float(Ca_t.mean()), 4),
                "inhale_length": round(r["length_inhale_3d"], 2),
                "exhale_length": round(r["length_exhale_3d"], 2),
                "arc_at_max": round(float(sv[t_s + int(np.argmax(Ca_t))]), 2),
                "total_arc": round(float(sv[-1]), 2),
                "n_pts_in": r["n_pts_inhale"], "n_pts_ex": r["n_pts_exhale"],
                "n_peaks": len(r["all_peaks"]),
                "status": get_safety_status(mx, safety_thresholds),
                "fatigue": r.get("fatigue"),
            })
        calc_breakdowns = {}
        for cn in sorted(results.keys()):
            calc_breakdowns[cn] = generate_calculation_breakdown(cn, results[cn], ul, input_unit)
    except Exception as exc:
        log.append("*** ERROR: " + str(exc))
        import traceback
        log.append(traceback.format_exc())
        table = pd.DataFrame()
        plot_3d = plot_heatmap = plot_ca = plot_morph = None
        summary_cards = []
        calc_breakdowns = {}
    out = {"table": table, "plot_3d": plot_3d, "plot_heatmap": plot_heatmap,
           "plot_ca": plot_ca, "plot_morph": plot_morph,
           "raw_results": results, "log": log,
           "summary_cards": summary_cards, "unit_label": ul,
           "patient_notes": patient_notes, "safety_thresholds": safety_thresholds,
           "calc_breakdowns": calc_breakdowns}
    out["html_report"] = generate_html_report(out) if summary_cards else ""
    return out

# ═══════════════ BATCH ═══════════════

def _match_files_in_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    mapping = {"front_inhale": None, "side_inhale": None,
               "front_exhale": None, "side_exhale": None}
    for fname in files:
        fl = fname.lower()
        is_front = "front" in fl
        is_side = "side" in fl or "right" in fl or "lat" in fl
        is_inhale = "inhal" in fl or "inhl" in fl
        is_exhale = "exhal" in fl or "exhl" in fl
        if is_front and is_inhale:
            mapping["front_inhale"] = os.path.join(folder, fname)
        elif is_front and is_exhale:
            mapping["front_exhale"] = os.path.join(folder, fname)
        elif is_side and is_inhale:
            mapping["side_inhale"] = os.path.join(folder, fname)
        elif is_side and is_exhale:
            mapping["side_exhale"] = os.path.join(folder, fname)
    return mapping

def run_batch_from_zip(zip_bytes, input_unit="um", trim_start_pct=0.0, trim_end_pct=0.0,
                        safety_thresholds=None, wire_od_mm=2.0, material_name="MP35N"):
    all_cards = []
    errors = []
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
                mapping = _match_files_in_folder(ipath)
                if all(mapping.values()):
                    folders.append((item, mapping))
        if not folders:
            mapping = _match_files_in_folder(tmpdir)
            if all(mapping.values()):
                folders.append(("root", mapping))
        for folder_name, files_dict in sorted(folders):
            try:
                out = run_web_analysis(files_dict, input_unit, trim_start_pct, trim_end_pct,
                                       safety_thresholds, "", wire_od_mm, material_name)
                for card in out.get("summary_cards", []):
                    card["patient"] = folder_name
                    all_cards.append(card)
            except Exception as e:
                errors.append(folder_name + ": " + str(e))
    if all_cards:
        master_df = pd.DataFrame(all_cards)
    else:
        master_df = pd.DataFrame()
    return {"master_table": master_df, "errors": errors}

# ═══════════════ COMPARISON MODE ═══════════════

def run_comparison(files_a, files_b, label_a="Lead A", label_b="Lead B",
                   input_unit="um", trim_start_pct=0.0, trim_end_pct=0.0,
                   safety_thresholds=None, wire_od_mm=2.0, material_name="MP35N"):
    """Run analysis on two lead sets and return combined output for comparison."""
    out_a = run_web_analysis(files_a, input_unit, trim_start_pct, trim_end_pct,
                              safety_thresholds, "", wire_od_mm, material_name)
    out_b = run_web_analysis(files_b, input_unit, trim_start_pct, trim_end_pct,
                              safety_thresholds, "", wire_od_mm, material_name)
    rows = []
    for card in out_a.get("summary_cards", []):
        card["lead_set"] = label_a
        rows.append(card)
    for card in out_b.get("summary_cards", []):
        card["lead_set"] = label_b
        rows.append(card)
    comparison_table = pd.DataFrame(rows)
    return {
        "lead_a": out_a, "lead_b": out_b,
        "comparison_table": comparison_table,
        "label_a": label_a, "label_b": label_b,
    }
