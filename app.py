import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="YOLO AI Detection & Training Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {background-color: #0e1117;}
    .stTabs [data-baseweb="tab-list"] gap {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        color: #c9d1d9;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State Paths & Defaults
if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = None
if "training_logs" not in st.session_state:
    st.session_state.training_logs = "No training active."
if "is_training" not in st.session_state:
    st.session_state.is_training = False


# Helper: Load YOLO Model
@st.cache_resource
def load_base_model(model_name="yolov8n.pt"):
    return YOLO(model_name)


def load_custom_model(weights_path):
    return YOLO(weights_path)


# App Header
st.title("🚀 YOLO Real-Time AI Detection & Training Suite")
st.markdown(
    "A complete computer vision package built with **Ultralytics YOLO** and **Streamlit**. "
    "Perform real-time webcam inference on COCO classes or train custom models directly on your custom datasets."
)

# Navigation Tabs
tab_detect, tab_train, tab_docs = st.tabs(
    ["🔴 Real-Time Webcam Detection", "⚙️ Custom Model Training", "📖 System Overview"]
)

# ==========================================
# TAB 1: REAL-TIME WEBCAM DETECTION
# ==========================================
with tab_detect:
    st.header("Live Webcam Object Detection")
    st.markdown(
        "Stream video directly from your webcam, perform high-speed YOLO inference, apply confidence filtering, and inspect detections live."
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        model_choice = st.selectbox(
            "Select Model Weights",
            ["yolov8n.pt (COCO)", "yolov8s.pt (COCO)", "Custom Trained Model"],
        )
    with col2:
        conf_threshold = st.slider(
            "Confidence Threshold", 0.1, 1.0, 0.35, 0.05
        )

    # Handle Model Source Selection
    active_model = None
    if model_choice == "Custom Trained Model":
        if (
            st.session_state.trained_model_path
            and os.path.exists(st.session_state.trained_model_path)
        ):
            active_model = load_custom_model(st.session_state.trained_model_path)
            st.sidebar.success(
                f"Loaded custom weights: {st.session_state.trained_model_path}"
            )
        else:
            st.sidebar.warning(
                "No custom model found. Falling back to yolov8n.pt."
            )
            active_model = load_base_model("yolov8n.pt")
    elif "s.pt" in model_choice:
        active_model = load_base_model("yolov8s.pt")
    else:
        active_model = load_base_model("yolov8n.pt")

    # Optional Class Filtering for COCO
    selected_classes = st.sidebar.multiselect(
        "Filter COCO Classes (Leave empty for all)",
        options=list(active_model.names.values()),
        default=[],
    )
    class_indices = (
        [
            k
            for k, v in active_model.names.items()
            if v in selected_classes
        ]
        if selected_classes
        else None
    )

    # WebRTC Video Processor Callback
    class YOLOVideoProcessor:

        def __init__(self):
            self.model = active_model
            self.conf = conf_threshold
            self.classes = class_indices

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO Inference
            results = self.model(
                img,
                conf=self.conf,
                classes=self.classes,
                verbose=False,
            )
            annotated_frame = results[0].plot()

            return frame.from_ndarray(annotated_frame, format="bgr24")

    col_webcam, col_stats = st.columns([3, 1])

    with col_webcam:
        st.subheader("Live Feed")
        webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_stats:
        st.subheader("Detection Metrics")
        st.info(
            "The WebRTC stream processes frames locally via OpenCV and Ultralytics YOLO."
        )
        st.markdown(f"**Active Model:** `{model_choice}`")
        st.markdown(f"**Confidence:** `{conf_threshold}`")
        if selected_classes:
            st.markdown(
                f"**Active Classes:** `{len(selected_classes)} filtered`"
            )
        else:
            st.markdown("**Active Classes:** `All COCO Classes`")


# ==========================================
# TAB 2: CUSTOM MODEL TRAINING
# ==========================================
with tab_train:
    st.header("Custom Dataset Training Suite")
    st.markdown(
        "Upload a zipped dataset structured in **YOLO format** (`images/train`, `images/val`, `labels/train`, `labels/val`, and a `data.yaml`) to train your custom model."
    )

    col_up, col_param = st.columns([1, 1])

    with col_up:
        st.subheader("1. Dataset Upload")
        dataset_file = st.file_uploader(
            "Upload Dataset ZIP Archive", type=["zip"]
        )

        base_model_train = st.selectbox(
            "Base Pretrained Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
        )

    with col_param:
        st.subheader("2. Hyperparameters")
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=10)
        imgsz = st.selectbox(
            "Image Size", options=[320, 416, 640, 1280], index=2
        )
        batch_size = st.selectbox(
            "Batch Size", options=[4, 8, 16, 32, -1], index=2
        )
        lr0 = st.number_input(
            "Initial Learning Rate (lr0)", value=0.01, format="%.4f"
        )

    st.markdown("---")
    st.subheader("3. Execute Training Pipeline")

    # Training execution thread function
    def run_yolo_training(
        data_yaml_path, base_model, epochs, imgsz, batch, lr
    ):
        try:
            st.session_state.is_training = True
            model = YOLO(base_model)
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr,
                verbose=True,
            )
            saved_weights = os.path.join(
                results.save_dir, "weights", "best.pt"
            )
            st.session_state.trained_model_path = saved_weights
            st.session_state.training_logs = (
                f"Training completed successfully! Saved to: {saved_weights}"
            )
        except Exception as e:
            st.session_state.training_logs = f"Error during training: {str(e)}"
        finally:
            st.session_state.is_training = False

    if st.button("🚀 Start Training Run", type="primary"):
        if dataset_file is not None:
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "dataset.zip")

            with open(zip_path, "wb") as f:
                f.write(dataset_file.getbuffer())

            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            data_yaml_path = None
            for root, dirs, files in os.walk(temp_dir):
                if "data.yaml" in files:
                    data_yaml_path = os.path.join(root, "data.yaml")
                    break

            if data_yaml_path:
                st.success("Dataset unpacked! Found `data.yaml`.")
                training_thread = threading.Thread(
                    target=run_yolo_training,
                    args=(
                        data_yaml_path,
                        base_model_train,
                        epochs,
                        imgsz,
                        batch_size,
                        lr0,
                    ),
                )
                training_thread.start()
            else:
                st.error(
                    "Could not find `data.yaml` in the uploaded ZIP archive. Ensure your dataset conforms to the standard YOLO structure."
                )
        else:
            st.warning("Please upload a dataset ZIP archive before training.")

    st.markdown("### Training Logs & Status")
    log_container = st.empty()

    if st.session_state.is_training:
        log_container.info(
            "⏳ Model training is currently in progress. Please check the terminal console for detailed epoch progress."
        )
    else:
        log_container.text(st.session_state.training_logs)

    if st.session_state.trained_model_path:
        st.success(
            f"Active Custom Weights Ready: `{st.session_state.trained_model_path}`. You can now select 'Custom Trained Model' in the Webcam tab."
        )
        with open(st.session_state.trained_model_path, "rb") as f:
            st.download_button(
                label="📥 Download Best Weights (`best.pt`)",
                data=f,
                file_name="custom_yolo_best.pt",
                mime="application/octet-stream",
            )

# ==========================================
# TAB 3: SYSTEM DOCUMENTATION
# ==========================================
with tab_docs:
    st.header("System Overview & Instructions")
    st.markdown(
        """
        ### 📌 How to use this package:
        1. **Real-Time Webcam Detection**:
           - Navigate to the **Real-Time Webcam Detection** tab.
           - Click **START** on the WebRTC video stream component to give browser permission for your camera.
           - Select between pre-trained COCO models (`yolov8n`, `yolov8s`) or your custom-trained weights.
           - Adjust the confidence threshold or filter specific object classes dynamically.
           
        2. **Custom Model Training**:
           - Package your custom dataset into a ZIP file containing `images/` (images train/val), `labels/` (labels train/val), and a root `data.yaml` file pointing to them.
           - Upload the ZIP file in the **Custom Model Training** tab.
           - Configure hyperparameters (Epochs, Image Size, Batch Size, Learning Rate).
           - Click **Start Training Run**. Once complete, download your trained `best.pt` weights or test them instantly in the live detection tab.
        """
        )         mime="application/octet-stream",
            )

# ==========================================
# TAB 3: SYSTEM DOCUMENTATION
# ==========================================
with tab_docs:
    st.header("System Overview & Instructions")
    st.markdown(
        """
        ### 📌 How to use this package:
        1. **Real-Time Webcam Detection**:
           - Navigate to the **Real-Time Webcam Detection** tab.
           - Click **START** on the WebRTC video stream component to give browser permission for your camera.
           - Select between pre-trained COCO models (`yolov8n`, `yolov8s`) or your custom-trained weights.
           - Adjust the confidence threshold or filter specific object classes dynamically.
           
        2. **Custom Model Training**:
           - Package your custom dataset into a ZIP file containing `images/` (images train/val), `labels/` (labels train/val), and a root `data.yaml` file pointing to them.
           - Upload the ZIP file in the **Custom Model Training** tab.
           - Configure hyperparameters (Epochs, Image Size, Batch Size, Learning Rate).
           - Click **Start Training Run**. Once complete, download your trained `best.pt` weights or test them instantly in the live detection tab.
        """
    )n_imgsz,
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
