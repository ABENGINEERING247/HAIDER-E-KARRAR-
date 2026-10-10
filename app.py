import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="HAIDER E KARRAR - AI Vision & Training System",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stAppHeader { background-color: #161b22; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #2ea043; }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State Variables
if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = None
if "training_logs" not in st.session_state:
    st.session_state.training_logs = "No training run initiated."
if "is_training" not in st.session_state:
    st.session_state.is_training = False


# Cache Base Pretrained Models
@st.cache_resource
def load_base_model(model_name="yolov8n.pt"):
    return YOLO(model_name)


def load_custom_model(weights_path):
    return YOLO(weights_path)


# App Title Header
st.title("🦅 HAIDER E KARRAR")
st.caption(
    "AI-Based Real-Time Object Detection, Custom Model Training & Computer Vision System"
)

# Sidebar Configuration
st.sidebar.title("🎛️ System Controls")
mode = st.sidebar.radio(
    "Select System Mode",
    [
        "🔴 Live Webcam Feed",
        "📂 Media Processing (Image/Video)",
        "⚙️ Custom Model Training",
        "📊 Analytics & System Status",
    ],
)

# Sidebar Model Selection
st.sidebar.subheader("Model Weights")
model_type = st.sidebar.selectbox(
    "Choose Active Weights",
    [
        "yolov8n.pt (COCO Pretrained)",
        "yolov8s.pt (COCO Pretrained)",
        "yolov8m.pt (COCO Pretrained)",
        "Custom Trained Model",
    ],
)

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.35, 0.05)
iou_thresh = st.sidebar.slider("IoU (NMS) Threshold", 0.1, 1.0, 0.45, 0.05)

# Load Selected Model
active_model = None
if model_type == "Custom Trained Model":
    if (
        st.session_state.trained_model_path
        and os.path.exists(st.session_state.trained_model_path)
    ):
        active_model = load_custom_model(st.session_state.trained_model_path)
        st.sidebar.success(
            f"Active: `{st.session_state.trained_model_path}`"
        )
    else:
        st.sidebar.warning(
            "No custom model found! Falling back to yolov8n.pt."
        )
        active_model = load_base_model("yolov8n.pt")
elif "yolov8s" in model_type:
    active_model = load_base_model("yolov8s.pt")
elif "yolov8m" in model_type:
    active_model = load_base_model("yolov8m.pt")
else:
    active_model = load_base_model("yolov8n.pt")

# Target Class Filtering
all_classes = list(active_model.names.values())
selected_classes = st.sidebar.multiselect(
    "Filter Classes (Leave empty for ALL)", options=all_classes, default=[]
)
class_indices = (
    [k for k, v in active_model.names.items() if v in selected_classes]
    if selected_classes
    else None
)

# ==========================================
# MODE 1: LIVE WEBCAM FEED
# ==========================================
if mode == "🔴 Live Webcam Feed":
    st.header("Real-Time Webcam Object Detection")
    st.markdown(
        "Stream video directly from your camera for live object detection and class identification."
    )

    class YOLOVideoProcessor:

        def __init__(self):
            self.model = active_model
            self.conf = conf_thresh
            self.iou = iou_thresh
            self.classes = class_indices

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Execute YOLO Inference
            results = self.model(
                img,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                verbose=False,
            )
            annotated_frame = results[0].plot()

            return frame.from_ndarray(annotated_frame, format="bgr24")

    col1, col2 = st.columns([3, 1])

    with col1:
        webrtc_streamer(
            key="haider-e-karrar-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("Stream Telemetry")
        st.info("WebRTC live stream active.")
        st.write(f"**Model:** `{model_type}`")
        st.write(f"**Confidence:** `{conf_thresh}`")
        st.write(f"**IoU Threshold:** `{iou_thresh}`")
        st.write(
            f"**Filtered Classes:** `{len(selected_classes) if selected_classes else 'All'}`"
        )


# ==========================================
# MODE 2: MEDIA PROCESSING (IMAGE / VIDEO)
# ==========================================
elif mode == "📂 Media Processing (Image/Video)":
    st.header("Static & Recorded Media Inference")

    media_type = st.radio(
        "Select Source Type", ["Upload Image", "Upload Video (MP4)"], horizontal=True
    )

    if media_type == "Upload Image":
        img_file = st.file_uploader(
            "Choose an Image File", type=["jpg", "jpeg", "png", "webp"]
        )

        if img_file is not None:
            image = Image.open(img_file)
            img_np = np.array(image)

            # Perform Detection
            results = active_model(
                img_np,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=class_indices,
            )
            annotated_img = results[0].plot()

            # Display side-by-side
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            with c2:
                st.subheader("Detected Output")
                st.image(annotated_img, use_container_width=True)

            # Object Counts Table
            boxes = results[0].boxes
            if len(boxes) > 0:
                class_ids = boxes.cls.cpu().numpy().astype(int)
                counts = {}
                for cid in class_ids:
                    cname = active_model.names[cid]
                    counts[cname] = counts.get(cname, 0) + 1

                st.subheader("Detected Objects Summary")
                df_counts = pd.DataFrame(
                    list(counts.items()), columns=["Object Class", "Count"]
                )
                st.table(df_counts)

    elif media_type == "Upload Video (MP4)":
        vid_file = st.file_uploader(
            "Choose an MP4 Video File", type=["mp4", "avi", "mov"]
        )

        if vid_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())

            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()

            stop_btn = st.button("Stop Video Processing")

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame-by-frame inference
                results = active_model(
                    frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    classes=class_indices,
                    verbose=False,
                )
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for Streamlit rendering
                annotated_frame_rgb = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )
                st_frame.image(annotated_frame_rgb, use_container_width=True)

            cap.release()


# ==========================================
# MODE 3: CUSTOM MODEL TRAINING
# ==========================================
elif mode == "⚙️ Custom Model Training":
    st.header("HAIDER E KARRAR Custom Model Training Pipeline")
    st.markdown(
        "Upload a dataset in **YOLO format** (zipped containing `images/train`, `images/val`, `labels/train`, `labels/val`, and `data.yaml`) to train custom weights."
    )

    c_up, c_param = st.columns([1, 1])

    with c_up:
        st.subheader("1. Dataset Package")
        dataset_zip = st.file_uploader(
            "Upload YOLO Dataset (.zip)", type=["zip"]
        )
        base_pretrained = st.selectbox(
            "Pretrained Backbone Weights",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        )

    with c_param:
        st.subheader("2. Hyperparameter Settings")
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=15)
        batch = st.selectbox("Batch Size", [4, 8, 16, 32, -1], index=2)
        imgsz = st.selectbox("Image Resolution", [320, 416, 640, 1280], index=2)
        lr0 = st.number_input(
            "Learning Rate (lr0)", value=0.01, format="%.4f"
        )

    # Threaded Training Launcher
    def train_worker(
        yaml_path, base_weights, train_epochs, train_imgsz, train_batch, train_lr
    ):
        try:
            st.session_state.is_training = True
            model = YOLO(base_weights)
            results = model.train(
                data=yaml_path,
                epochs=train_epochs,
                imgsz=train_imgsz,
                batch=train_batch,
                lr0=train_lr,
                verbose=True,
            )
            best_weights = os.path.join(results.save_dir, "weights", "best.pt")
            st.session_state.trained_model_path = best_weights
            st.session_state.training_logs = (
                f"Training completed successfully! Saved weights to: {best_weights}"
            )
        except Exception as err:
            st.session_state.training_logs = f"Training failed: {str(err)}"
        finally:
            st.session_state.is_training = False

    if st.button("🚀 Launch Training Execution"):
        if dataset_zip is not None:
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "dataset.zip")

            with open(zip_path, "wb") as f:
                f.write(dataset_zip.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as zref:
                zref.extractall(temp_dir)

            # Locate data.yaml
            data_yaml = None
            for root, dirs, files in os.walk(temp_dir):
                if "data.yaml" in files:
                    data_yaml = os.path.join(root, "data.yaml")
                    break

            if data_yaml:
                st.success("Unpacked dataset! Found `data.yaml`.")
                t = threading.Thread(
                    target=train_worker,
                    args=(
                        data_yaml,
                        base
