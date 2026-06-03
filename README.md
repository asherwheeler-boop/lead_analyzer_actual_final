# 4-File Excel Upload Graph & Diagram Builder

This Streamlit webpage lets a user upload **four Excel/CSV files** and automatically generates a dashboard with:

- Upload summary + inferred coordinate columns
- Dataset metric table
- 2D overlay trajectory diagram
- 3D overlay trajectory diagram (when Z data exists)
- Per-file coordinate profile graphs
- Per-file curvature and step-length graphs
- Pairwise XY overlay comparison plots
- Pairwise point-to-point distance graphs
- Downloadable consolidated Excel report

## Files included

- `app.py` – Streamlit UI
- `analysis_engine.py` – data normalization, curvature math, comparison logic, workbook export
- `requirements.txt` – Python dependencies

## Expected input

The app tries to infer coordinate columns automatically. It first looks for common names such as:

- X: `x`, `x_mm`, `x_cm`, `x_coord`, `xcoord`
- Y: `y`, `y_mm`, `y_cm`, `y_coord`, `ycoord`
- Z: `z`, `z_mm`, `z_cm`, `z_coord`, `zcoord`

If those are not present, it falls back to the first numeric columns.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app works with `.xlsx`, `.xls`, and `.csv` files.
- Pairwise comparisons can be automatic sequential pairs (1-2 and 3-4) or all pair combinations.
- A consolidated Excel report is created directly from the webpage for download.
