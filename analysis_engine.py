
import io
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CANDIDATE_X = ['x', 'x_mm', 'x_cm', 'xcoord', 'x_coord', 'xr', 'x coordinate']
CANDIDATE_Y = ['y', 'y_mm', 'y_cm', 'ycoord', 'y_coord', 'yr', 'y coordinate']
CANDIDATE_Z = ['z', 'z_mm', 'z_cm', 'zcoord', 'z_coord', 'zr', 'z coordinate']


def _clean_name(name: str) -> str:
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum() or ch == ' ')


def infer_coordinate_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    cleaned = {_clean_name(c): c for c in cols}

    def pick(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            key = _clean_name(cand)
            if key in cleaned:
                return cleaned[key]
        return None

    x = pick(CANDIDATE_X)
    y = pick(CANDIDATE_Y)
    z = pick(CANDIDATE_Z)

    if not x or not y:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            x = x or numeric_cols[0]
            y = y or numeric_cols[1]
        if len(numeric_cols) >= 3:
            z = z or numeric_cols[2]

    coords = [c for c in [x, y, z] if c is not None]
    return coords


def normalize_dataframe(df: pd.DataFrame, source_name: str, coord_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if coord_cols is None:
        coord_cols = infer_coordinate_columns(df)
    out = pd.DataFrame()
    out['point_index'] = np.arange(len(df))
    if len(coord_cols) >= 1:
        out['x'] = pd.to_numeric(df[coord_cols[0]], errors='coerce')
    if len(coord_cols) >= 2:
        out['y'] = pd.to_numeric(df[coord_cols[1]], errors='coerce')
    if len(coord_cols) >= 3:
        out['z'] = pd.to_numeric(df[coord_cols[2]], errors='coerce')
    out = out.dropna(subset=['x', 'y']).reset_index(drop=True)
    out['source_file'] = source_name
    return out


def _coords(df: pd.DataFrame) -> np.ndarray:
    cols = ['x', 'y'] + (['z'] if 'z' in df.columns else [])
    return df[cols].to_numpy(dtype=float)


def _moving_average(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(a) == 0:
        return a
    window = int(window)
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode='same')


def curvature_profile(df: pd.DataFrame, smooth_window: int = 3) -> pd.DataFrame:
    pts = _coords(df)
    n = len(pts)
    if n == 0:
        return pd.DataFrame({'point_index': [], 'curvature': [], 'step_length': []})
    if n == 1:
        return pd.DataFrame({'point_index': [0], 'curvature': [0.0], 'step_length': [0.0]})

    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)

    if pts.shape[1] == 2:
        numerator = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    else:
        numerator = np.linalg.norm(np.cross(d1, d2), axis=1)

    denominator = np.linalg.norm(d1, axis=1) ** 3
    denominator = np.where(denominator == 0, np.nan, denominator)
    curv = np.nan_to_num(numerator / denominator, nan=0.0, posinf=0.0, neginf=0.0)
    curv = _moving_average(curv, smooth_window)

    diffs = np.diff(pts, axis=0)
    step = np.linalg.norm(diffs, axis=1)
    step = np.insert(step, 0, 0.0)
    step = _moving_average(step, smooth_window)

    return pd.DataFrame({'point_index': np.arange(n), 'curvature': curv, 'step_length': step})


def summarize_dataset(df: pd.DataFrame, smooth_window: int = 3) -> Dict[str, float]:
    pts = _coords(df)
    curv = curvature_profile(df, smooth_window=smooth_window)
    diffs = np.diff(pts, axis=0) if len(pts) >= 2 else np.empty((0, pts.shape[1]))
    step = np.linalg.norm(diffs, axis=1) if len(diffs) else np.array([])
    path_length = float(step.sum()) if len(step) else 0.0

    summary = {
        'file': str(df['source_file'].iloc[0]) if len(df) else 'unknown',
        'point_count': int(len(df)),
        'dimensions': int(pts.shape[1]) if len(df) else 0,
        'path_length': path_length,
        'x_min': float(df['x'].min()) if len(df) else np.nan,
        'x_max': float(df['x'].max()) if len(df) else np.nan,
        'y_min': float(df['y'].min()) if len(df) else np.nan,
        'y_max': float(df['y'].max()) if len(df) else np.nan,
        'mean_step_length': float(step.mean()) if len(step) else 0.0,
        'max_step_length': float(step.max()) if len(step) else 0.0,
        'mean_curvature': float(curv['curvature'].mean()) if len(curv) else 0.0,
        'max_curvature': float(curv['curvature'].max()) if len(curv) else 0.0,
        'std_curvature': float(curv['curvature'].std(ddof=0)) if len(curv) else 0.0,
    }
    if 'z' in df.columns:
        summary['z_min'] = float(df['z'].min())
        summary['z_max'] = float(df['z'].max())
    return summary


def _aligned_coordinates(df: pd.DataFrame, n: int) -> np.ndarray:
    pts = _coords(df)
    if len(pts) == n:
        return pts
    old_idx = np.linspace(0, 1, len(pts)) if len(pts) > 1 else np.array([0.0])
    new_idx = np.linspace(0, 1, n)
    out = []
    for dim in range(pts.shape[1]):
        out.append(np.interp(new_idx, old_idx, pts[:, dim]))
    return np.column_stack(out)


def pairwise_alignment_summary(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str):
    target_n = max(min(len(df_a), len(df_b)), 2)
    a = _aligned_coordinates(df_a, target_n)
    b = _aligned_coordinates(df_b, target_n)
    dims = min(a.shape[1], b.shape[1])
    a = a[:, :dims]
    b = b[:, :dims]
    delta = a - b
    dist = np.linalg.norm(delta, axis=1)

    result = {
        'file_a': name_a,
        'file_b': name_b,
        'aligned_points': int(target_n),
        'mean_distance': float(dist.mean()),
        'max_distance': float(dist.max()),
        'rmse_distance': float(np.sqrt(np.mean(dist ** 2))),
        'distance_profile': pd.DataFrame({'point_index': np.arange(target_n), 'distance': dist}),
    }
    return result


def export_report_workbook(normalized_tables: Dict[str, pd.DataFrame], summary_df: pd.DataFrame, pair_df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary_metrics', index=False)
        safe_pair_df = pair_df.copy()
        if 'distance_profile' in safe_pair_df.columns:
            safe_pair_df = safe_pair_df.drop(columns=['distance_profile'])
        safe_pair_df.to_excel(writer, sheet_name='pairwise_comparison', index=False)
        for i, (name, df) in enumerate(normalized_tables.items(), start=1):
            sheet = f'file_{i}'[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            curvature_profile(df).to_excel(writer, sheet_name=f'curv_{i}'[:31], index=False)
    buffer.seek(0)
    return buffer
