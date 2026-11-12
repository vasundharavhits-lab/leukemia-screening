import os

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import auth, db
from src.features import analyze_cell, decode_image

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_mobilenetv2_finetuned.keras")
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/model_config_mnv2.json")
POWERBI_URL = os.getenv("POWERBI_EMBED_URL", "").strip()

st.set_page_config(page_title="Leukemia Screening", page_icon="🩸", layout="wide")
db.init_db()


# ------------------------------------------------------------------ login
def login_gate():
    """Nothing below this runs until someone has signed in."""
    if st.session_state.get("user"):
        return
    st.title("🩸 Leukemia Screening Dashboard")
    st.subheader("Sign in")
    if auth.user_count() == 0:
        st.error("No users exist yet. An administrator must run this in the project folder:  "
                 "python manage_users.py add USERNAME")
        st.stop()
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        go = st.form_submit_button("Sign in", type="primary")
    if go:
        ok, msg = auth.login(u, p)
        if ok:
            st.session_state["user"] = u.strip().lower()
            st.rerun()
        else:
            st.error(msg)
    st.stop()


login_gate()
user = st.session_state["user"]

with st.sidebar:
    st.write(f"Signed in as **{user}**")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    from src.model import LeukemiaModel
    return LeukemiaModel(MODEL_PATH, CONFIG_PATH)


def fmt(v, digits=3):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}f}"


# ------------------------------------------------------------------ analytics panel (top right)
def render_analytics():
    st.subheader("Live analytics")
    if POWERBI_URL:
        components.iframe(POWERBI_URL, height=560, scrolling=False)
        st.caption("Power BI report. It refreshes on the schedule set in Power BI.")
        return

    # Built-in preview used until a Power BI report is connected. Updates after every scan.
    df = db.load_scans()
    if df.empty:
        st.info("No scans yet. Analyze an image and the statistics appear here.")
        return
    df["lab"] = df["lab"].replace("", "Unspecified").fillna("Unspecified").str.strip().str.title()
    flagged = df["prediction"].str.startswith("Leukemia")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scans", len(df))
    c2.metric("Flagged leukemia", f"{flagged.mean():.0%}")
    c3.metric("Pending review", int((df["review_status"] == "Pending").sum()))

    st.markdown("**Scans per day**")
    st.bar_chart(df.assign(day=pd.to_datetime(df["scanned_at"]).dt.date).groupby("day").size())
    st.markdown("**Predictions by lab**")
    st.bar_chart(df.groupby(["lab", "prediction"]).size().unstack(fill_value=0))
    st.markdown("**Latest scans**")
    st.dataframe(
        df.sort_values("id", ascending=False).head(8)[
            ["id", "scanned_at", "patient_id", "prediction", "prob_leukemia", "review_status",
             "reviewed_by"]],
        hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ page
st.title("🩸 Leukemia Screening Dashboard")
st.caption("Decision-support prototype. Results must be confirmed by a qualified hematologist "
           "and are not a diagnosis.")

left, right = st.columns([3, 2], gap="large")

with right:
    render_analytics()

with left:
    st.subheader("New scan")
    with st.form("scan_form"):
        c1, c2, c3 = st.columns(3)
        patient_id = c1.text_input("Patient ID (no names)")
        age = c2.number_input("Age", 0, 120, 30)
        gender = c3.selectbox("Gender", ["Male", "Female", "Other"])
        lab = st.text_input("Lab / branch")
        file = st.file_uploader("Blood cell image",
                                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not patient_id.strip():
            st.warning("Enter a Patient ID.")
        elif file is None:
            st.warning("Upload an image first.")
        else:
            try:
                model = get_model()
            except Exception as e:
                st.error(f"Could not load the model from '{MODEL_PATH}': {e}")
                st.stop()
            try:
                with st.spinner("Analyzing..."):
                    bgr = decode_image(file.getvalue())

                    p = model.predict(bgr)
                    is_single = model.last_route.startswith("EfficientNet")
                    if is_single:
                        feats = analyze_cell(bgr)
                    else:       # segmentation is built for single cells on black, not wide smears
                        feats = {"nucleus_area": None, "nucleus_circularity": None,
                                 "nucleus_solidity": None, "nc_ratio": None, "overlay": None}
                    try:
                        cam = model.gradcam_overlay(bgr)
                    except Exception:
                        cam = None                       # heatmap is optional
                    result = {
                        "prediction": model.label(p), "p": p, "threshold": model.threshold,
                        "uncertain": model.is_uncertain(p), "route": model.last_route,
                        "is_single": is_single,
                    }
                    scan_id = db.insert_scan(
                        patient_id=patient_id.strip(), age=int(age), gender=gender,
                        lab=lab.strip() or "Unspecified", image_name=file.name,
                        prob_leukemia=p, threshold=model.threshold,
                        prediction=result["prediction"], uncertain=result["uncertain"],
                        nucleus_area=feats["nucleus_area"],
                        nucleus_circularity=feats["nucleus_circularity"],
                        nucleus_solidity=feats["nucleus_solidity"], nc_ratio=feats["nc_ratio"],
                        scanned_by=user)
                st.session_state["last"] = {**result, "scan_id": scan_id, "bgr": bgr,
                                            "feats": feats, "cam": cam,
                                            "patient_id": patient_id.strip()}
                st.rerun()           # redraw so the analytics panel includes this scan
            except ValueError as e:
                st.error(str(e))

    last = st.session_state.get("last")
    if last:
        st.divider()
        st.subheader(f"Result for {last['patient_id']}  (scan #{last['scan_id']})")
        msg = (f"{last['prediction']}   |   P(leukemia) = {last['p']:.1%}   "
               f"(decision threshold {last['threshold']:.2f})")
        (st.error if last["prediction"].startswith("Leukemia") else st.success)(msg)
        st.caption(f"Model used: {last.get('route', 'n/a')}")
        if last["uncertain"]:
            st.warning("Score is close to the decision threshold. Review this case manually.")

        f = last["feats"]
        if last.get("is_single", True):
            i1, i2, i3 = st.columns(3)
            i1.image(last["bgr"], channels="BGR", caption="Original", use_container_width=True)
            i2.image(f["overlay"], channels="BGR",
                     caption="OpenCV: nucleus (green) / cell (blue)", use_container_width=True)
            if last["cam"] is not None:
                i3.image(last["cam"], caption="Grad-CAM: where the model looked",
                         use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nucleus area (px)", fmt(f["nucleus_area"], 0))
            m2.metric("Circularity", fmt(f["nucleus_circularity"]))
            m3.metric("Solidity", fmt(f["nucleus_solidity"]))
            m4.metric("N/C ratio (approx.)", fmt(f["nc_ratio"]))
        else:
            i1, i2 = st.columns(2)
            i1.image(last["bgr"], channels="BGR", caption="Original", use_container_width=True)
            if last["cam"] is not None:
                i2.image(last["cam"], caption="Grad-CAM: rough view of where the model looked",
                         use_container_width=True)
            st.info("Nucleus measurements are shown only for single-cell images. "
                    "On smear images like these, the segmentation would describe a fragment of the "
                    "image, not the cells, so it is switched off.")

        st.markdown("**Clinician review**")
        r1, r2 = st.columns([2, 1])
        status = r1.selectbox("Review decision",
                              ["Pending", "Confirmed", "Rejected", "Needs second opinion"],
                              key=f"review_{last['scan_id']}")
        if r2.button("Save review"):
            db.update_review(last["scan_id"], status, reviewer=user)
            st.rerun()
