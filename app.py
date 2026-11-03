# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Streamlit operator console: upload a video, tune the detector, review incidents.

    streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from tcd import ROOT, load_config
from tcd.alerts import send_all
from tcd.detect import run_video, scorer
from tcd.incidents import IncidentStore

st.set_page_config(page_title="Collision detection", layout="wide")
cfg = load_config()
store = IncidentStore()

st.title("Real-time traffic collision detection")
available = [p.stem for p in (ROOT / "models").glob("*.keras")]
with st.sidebar:
    st.header("Detector settings")
    if not available:
        st.warning("No trained model in models/. Run `python -m tcd.train --model vgg19` first.")
    model_name = st.selectbox("Model", available or ["vgg19"])
    cfg["detection"]["threshold"] = st.slider("Alert threshold", 0.5, 0.95, float(cfg["detection"]["threshold"]), 0.05)
    cfg["detection"]["confirm_frames"] = st.slider("Consecutive frames to confirm", 1, 10, int(cfg["detection"]["confirm_frames"]))
    cfg["detection"]["sample_every_n_frames"] = st.slider("Score every N frames", 1, 15, int(cfg["detection"]["sample_every_n_frames"]))
    cfg["alerts"]["dry_run"] = not st.toggle("Send real alerts (needs .env)", value=False)

video = st.file_uploader("CCTV clip", type=["mp4", "avi", "mov", "mkv"])
if video and available and st.button("Analyse"):
    import tensorflow as tf
    with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as f:
        f.write(video.read())
    model = tf.keras.models.load_model(ROOT / "models" / f"{model_name}.keras", safe_mode=False)
    found = []

    def handle(inc):
        rid = store.add(inc, source=video.name, model=model_name)
        found.append((inc, send_all(inc, cfg, incident_id=rid)))

    with st.spinner("Scoring frames..."):
        incidents, fps = run_video(f.name, scorer(model, cfg["models"][model_name]["input"]), cfg, handle)
    st.success(f"{len(incidents)} incident(s) found - scoring at {fps:.1f} frames/s")
    for inc, msg in found:
        c1, c2 = st.columns([1, 2])
        c1.image(inc.frame[:, :, ::-1], caption=f"frame {inc.frame_index} - {inc.confidence:.0%}")
        c2.code(msg["body"])

st.subheader("Incident log")
rows = store.all()
st.dataframe(rows, use_container_width=True) if rows else st.caption("No incidents yet.")
