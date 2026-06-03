"""Streamlit front-end for Pacing Lead Analyzer v2."""
import streamlit as st
import io
from pacing_lead_curvature_webapp import (
    run_web_analysis, run_batch_from_zip, run_comparison,
    MATERIAL_PROPERTIES, DEFAULT_SAFETY, UNIT_LABELS,
)

st.set_page_config(page_title="Pacing Lead Analyzer v2", page_icon="\U0001fac0", layout="wide")
STATUS_COLORS = {"PASS": "#009988", "WARNING": "#EE7733", "FAIL": "#FF0000"}
st.title("\U0001fac0 3D Pacing Lead Curvature Amplitude Analyzer v2")
st.caption("Upload biplane X-ray trace files, configure settings, then click Run.")
st.divider()

with st.sidebar:
    st.header("Settings")
    input_unit = st.selectbox("Input coordinate units", options=["um","mm","cm"], index=0,
        format_func=lambda u: {"um":"Micrometers","mm":"Millimeters","cm":"Centimeters"}[u])
    st.divider()
    st.subheader("Safety Thresholds")
    visone_val = st.number_input("VisONE Stimulation (cm^-1)", value=0.88, step=0.01, format="%.2f")
    resp_val = st.number_input("Respiration (cm^-1)", value=0.91, step=0.01, format="%.2f")
    safety_thresholds = {"VisONE Stimulation": visone_val, "Respiration": resp_val}
    st.divider()
    st.subheader("Data Trimming")
    trim_start = st.slider("Trim start (%)", 0, 50, 0, 1)
    trim_end = st.slider("Trim end (%)", 0, 50, 0, 1)
    st.divider()
    st.subheader("Fatigue Estimation")
    material = st.selectbox("Wire material", list(MATERIAL_PROPERTIES.keys()))
    wire_od = st.number_input("Wire OD (mm)", value=2.0, step=0.1, format="%.1f")
    st.divider()
    st.subheader("Patient Notes")
    patient_notes = st.text_area("Notes (included in report)", placeholder="e.g. Patient 42, RV lead")
    st.divider()
    mode = st.radio("Analysis Mode", ["Single", "Batch (ZIP)", "Comparison (A vs B)"])

if mode == "Single":
    st.subheader("Upload 4 Files")
    c1, c2 = st.columns(2)
    with c1:
        fi = st.file_uploader("Front Inhale", type=["csv","xlsx","xls"], key="fi")
        fe = st.file_uploader("Front Exhale", type=["csv","xlsx","xls"], key="fe")
    with c2:
        si = st.file_uploader("Side Inhale", type=["csv","xlsx","xls"], key="si")
        se = st.file_uploader("Side Exhale", type=["csv","xlsx","xls"], key="se")
    if all([fi, si, fe, se]):
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                output = run_web_analysis(
                    {"front_inhale": io.BytesIO(fi.getvalue()), "side_inhale": io.BytesIO(si.getvalue()),
                     "front_exhale": io.BytesIO(fe.getvalue()), "side_exhale": io.BytesIO(se.getvalue())},
                    input_unit, float(trim_start), float(trim_end),
                    safety_thresholds, patient_notes, wire_od, material)
            ul = output.get("unit_label", "um")
            if not output["table"].empty:
                st.success("Analysis complete!")
                st.divider()
                st.subheader("Results")
                for card in output.get("summary_cards", []):
                    status = card["status"]
                    color = STATUS_COLORS.get(status, "#333")
                    st.markdown(f"#### {card['name']} <span style='color:{color};font-weight:bold'>[{status}]</span>", unsafe_allow_html=True)
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Peak Ca (cm^-1)", f"{card['max_ca']:.4f}")
                    c2.metric("Mean Ca (cm^-1)", f"{card['mean_ca']:.4f}")
                    c3.metric(f"Inhale Len ({ul})", f"{card['inhale_length']:.1f}")
                    c4.metric(f"Exhale Len ({ul})", f"{card['exhale_length']:.1f}")
                    c5,c6,c7,c8 = st.columns(4)
                    c5.metric(f"Arc at Peak ({ul})", f"{card['arc_at_max']:.1f}")
                    c6.metric(f"Total Arc ({ul})", f"{card['total_arc']:.1f}")
                    c7.metric("Peaks", f"{card['n_peaks']}")
                    fat = card.get("fatigue")
                    if fat and "estimated_cycles" in fat:
                        cyc = fat["estimated_cycles"]
                        cyc_str = "inf" if cyc == float("inf") else f"{int(cyc):,}"
                        c8.metric("Fatigue Life", cyc_str + " cyc")
                        st.caption(f"{fat['fatigue_status']} | Strain: {fat['bending_strain']:.4f} | {material}")
                    st.markdown("---")
                dl1, dl2 = st.columns(2)
                dl1.download_button("Download CSV", data=output["table"].to_csv(index=False),
                    file_name="curvature_results.csv", mime="text/csv")
                if output.get("html_report"):
                    dl2.download_button("Download HTML Report", data=output["html_report"],
                        file_name="curvature_report.html", mime="text/html")
                st.divider()
                st.subheader("3-D Wire Reconstruction")
                if output["plot_3d"]:
                    st.plotly_chart(output["plot_3d"], use_container_width=True)
                st.subheader("Curvature Heatmap")
                if output.get("plot_heatmap"):
                    st.plotly_chart(output["plot_heatmap"], use_container_width=True)
                st.subheader("Inhale/Exhale Animation")
                if output.get("plot_morph") and isinstance(output["plot_morph"], dict):
                    for cn, fig in output["plot_morph"].items():
                        st.plotly_chart(fig, use_container_width=True)
                st.divider()
                st.subheader("Curvature Amplitude vs Arc Length")
                if output["plot_ca"] and isinstance(output["plot_ca"], dict):
                    for cn, fig in output["plot_ca"].items():
                        st.plotly_chart(fig, use_container_width=True)
                st.divider()
                st.subheader("Calculation Transparency")
                for cn, steps in output.get("calc_breakdowns", {}).items():
                    with st.expander(f"{cn} - Step-by-Step"):
                        for s in steps:
                            st.markdown(f"**{s['step']}**")
                            st.write(s.get("desc", ""))
                            if "formula" in s: st.code(s["formula"], language="text")
                            if "detail" in s: st.caption(s["detail"])
                            st.markdown("---")
                with st.expander("Processing Log"):
                    st.code("\n".join(output["log"]), language="text")
            else:
                st.error("Analysis failed.")
                with st.expander("Log", expanded=True):
                    st.code("\n".join(output["log"]), language="text")
    else:
        st.info("Upload all 4 files to begin.")

elif mode == "Batch (ZIP)":
    st.subheader("Batch Processing")
    st.caption("Upload a .zip with subfolders containing 4 files each (front/side x inhale/exhale in filenames).")
    zip_file = st.file_uploader("Upload ZIP", type=["zip"], key="batch_zip")
    if zip_file and st.button("Run Batch", type="primary", use_container_width=True):
        with st.spinner("Processing batch..."):
            result = run_batch_from_zip(zip_file.getvalue(), input_unit,
                float(trim_start), float(trim_end), safety_thresholds, wire_od, material)
        if not result["master_table"].empty:
            st.success(f"{len(result['master_table'])} results!")
            st.dataframe(result["master_table"], use_container_width=True)
            st.download_button("Master CSV", data=result["master_table"].to_csv(index=False),
                file_name="batch_results.csv", mime="text/csv")
        else:
            st.error("No results.")
        if result["errors"]:
            with st.expander("Errors"):
                for e in result["errors"]: st.warning(e)

elif mode == "Comparison (A vs B)":
    st.subheader("Side-by-Side Comparison")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Lead A")
        a_fi = st.file_uploader("A: Front Inhale", type=["csv","xlsx","xls"], key="a_fi")
        a_si = st.file_uploader("A: Side Inhale", type=["csv","xlsx","xls"], key="a_si")
        a_fe = st.file_uploader("A: Front Exhale", type=["csv","xlsx","xls"], key="a_fe")
        a_se = st.file_uploader("A: Side Exhale", type=["csv","xlsx","xls"], key="a_se")
    with cb:
        st.markdown("#### Lead B")
        b_fi = st.file_uploader("B: Front Inhale", type=["csv","xlsx","xls"], key="b_fi")
        b_si = st.file_uploader("B: Side Inhale", type=["csv","xlsx","xls"], key="b_si")
        b_fe = st.file_uploader("B: Front Exhale", type=["csv","xlsx","xls"], key="b_fe")
        b_se = st.file_uploader("B: Side Exhale", type=["csv","xlsx","xls"], key="b_se")
    if all([a_fi,a_si,a_fe,a_se]) and all([b_fi,b_si,b_fe,b_se]):
        if st.button("Compare", type="primary", use_container_width=True):
            with st.spinner("Comparing..."):
                fa = {"front_inhale":io.BytesIO(a_fi.getvalue()),"side_inhale":io.BytesIO(a_si.getvalue()),
                      "front_exhale":io.BytesIO(a_fe.getvalue()),"side_exhale":io.BytesIO(a_se.getvalue())}
                fb = {"front_inhale":io.BytesIO(b_fi.getvalue()),"side_inhale":io.BytesIO(b_si.getvalue()),
                      "front_exhale":io.BytesIO(b_fe.getvalue()),"side_exhale":io.BytesIO(b_se.getvalue())}
                comp = run_comparison(fa, fb, "Lead A", "Lead B", input_unit,
                    float(trim_start), float(trim_end), safety_thresholds, wire_od, material)
            if not comp["comparison_table"].empty:
                st.success("Comparison complete!")
                st.dataframe(comp["comparison_table"], use_container_width=True)
                st.download_button("Comparison CSV", data=comp["comparison_table"].to_csv(index=False),
                    file_name="comparison.csv", mime="text/csv")
                st.divider()
                la, lb = st.columns(2)
                with la:
                    st.markdown("### Lead A")
                    if comp["lead_a"].get("plot_3d"): st.plotly_chart(comp["lead_a"]["plot_3d"], use_container_width=True)
                with lb:
                    st.markdown("### Lead B")
                    if comp["lead_b"].get("plot_3d"): st.plotly_chart(comp["lead_b"]["plot_3d"], use_container_width=True)
            else:
                st.error("Comparison failed.")
    else:
        st.info("Upload all 8 files (4 per lead) to compare.")
