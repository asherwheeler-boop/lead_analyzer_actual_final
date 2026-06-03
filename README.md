# Pacing Lead Curvature Amplitude Analyzer v2

## Features
- Single / Batch / Comparison analysis modes
- 3D wire reconstruction from biplane X-ray traces
- Curvature amplitude with safety thresholds
- Fatigue life estimation (MP35N, DFT, Elgiloy, Nitinol)
- Curvature heatmaps and inhale/exhale animation
- Calculation transparency (step-by-step breakdown)
- CSV + HTML report export

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push files to GitHub (not as ZIP)
2. Go to https://share.streamlit.io
3. New App -> select repo -> main file: app.py -> Deploy
