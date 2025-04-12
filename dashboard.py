import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import tempfile
import zipfile
import glob
from detector import detect_damage

# === Page Setup ===
st.set_page_config(page_title="🧠 Fabric Defect Dashboard", layout="wide")
st.title("🧵 Fabric Defect Detection Dashboard")

# === Session Init ===
if "inputs" not in st.session_state:
    st.session_state.inputs = []
if "results" not in st.session_state:
    st.session_state.results = []

# === Dashboard Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["📂 Input", "🧠 Predictions", "📊 Analytics", "📥 Export"])

# ----------------------------------------------------------
# 📂 TAB 1 - INPUT
# ----------------------------------------------------------
with tab1:
    st.header("📂 Choose Input Source")

    input_mode = st.radio("Select Input Type", ["Image", "Video", "Webcam", "Image Folder"])

    if input_mode == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.inputs = [(image, "uploaded_image")]

    elif input_mode == "Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                stframe.image(frame[:, :, ::-1], channels="RGB")
                frames.append((frame.copy(), f"video_frame_{len(frames)}"))
            cap.release()
            st.session_state.inputs = frames

    elif input_mode == "Webcam":
        st.warning("📸 Webcam preview in Streamlit is experimental. Capture not supported here.")
        st.info("Please use image/video upload for now.")

    elif input_mode == "Image Folder":
        zip_file = st.file_uploader("Upload a ZIP folder of images", type=["zip"])
        if zip_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                paths = glob.glob(os.path.join(tmpdir, "**", "*.*"), recursive=True)
                images = []
                for p in paths:
                    if p.lower().endswith((".jpg", ".jpeg", ".png")):
                        img = cv2.imread(p)
                        if img is not None:
                            images.append((img, os.path.basename(p)))
                st.success(f"📸 Found {len(images)} valid images.")
                st.session_state.inputs = images

# ----------------------------------------------------------
# 🧠 TAB 2 - PREDICTIONS
# ----------------------------------------------------------
with tab2:
    st.header("🧠 Detection Results")

    if not st.session_state.inputs:
        st.warning("⚠️ No input images found. Upload from the Input tab.")
    else:
        all_results = []
        for image, name in st.session_state.inputs:
            st.subheader(f"🖼️ {name}")
            annotated, results = detect_damage(image)
            st.image(annotated[:, :, ::-1], caption="Detected Image", use_column_width=True)

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                for r in results:
                    r["source"] = name
                all_results.extend(results)
            else:
                st.info("✅ No defects detected in this input.")

        st.session_state.results = all_results

# ----------------------------------------------------------
# 📊 TAB 3 - ANALYTICS
# ----------------------------------------------------------
with tab3:
    st.header("📊 Defect Summary")

    if not st.session_state.results:
        st.info("Run predictions first to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.results)
        counts = df["label"].value_counts()
        st.bar_chart(counts)

        st.subheader("Defect Counts per Image")
        st.dataframe(df.groupby("source")["label"].value_counts().unstack(fill_value=0))

# ----------------------------------------------------------
# 📥 TAB 4 - EXPORT
# ----------------------------------------------------------
with tab4:
    st.header("📥 Export Results")

    if not st.session_state.results:
        st.warning("No results to export.")
    else:
        df = pd.DataFrame(st.session_state.results)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "defect_results.csv", "text/csv")

        st.success("✅ Results ready to export.")
