import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="YOLO AI Vision Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Dark Theme & UI Components
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        color: #c9d1d9;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #58a6ff;
        border-bottom: 3px solid #58a6ff;
    }
    .welcome-card {
        background: linear-gradient(135deg, #1f242d 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .status-badge {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE & WELCOME MODAL
# ============================================================
DEFAULTS = {
    "trained_model_path": None,
    "training_logs": "No training run has been started.",
    "training_workspace": None,
    "has_visited": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Interactive Welcome Dialog for First-Time Visitors
@st.dialog("👋 Welcome to YOLO AI Vision Suite!")
def show_welcome_modal():
    st.markdown(
        """
        ### Explore Real-Time Computer Vision & Training
        
        This application provides an all-in-one workspace for **YOLO Object Detection**:
        
        * 🎥 **Real-Time Webcam Detection:** Perform live object detection directly via your web browser.
        * ⚡ **Custom Model Training:** Upload your own dataset in YOLO format and train custom weights on the fly.
        * 📊 **Exportable Models:** Download your trained `.pt` weights anytime for local deployment.
        
        ---
        *Grant camera permissions in your browser when using the real-time tab for the best experience.*
        """
    )
    if st.button("🚀 Get Started", type="primary", use_container_width=True):
        st.session_state.has_visited = True
        st.rerun()

if not st.session_state.has_visited:
    show_welcome_modal()

# ============================================================
# MODEL HELPERS & UTILITIES
# ============================================================
@st.cache_resource(show_spinner="⚡ Loading YOLO model...")
def load_base_model(model_name: str):
    """Cache standard pretrained models between Streamlit reruns."""
    return YOLO(model_name)


def load_custom_model(weights_path: str):
    """Load custom weights without caching a potentially stale path."""
    return YOLO(weights_path)


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract ZIP files while preventing Path Traversal vulnerabilities."""
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"Unsafe path detected in ZIP archive: {member.filename}"
                )

            target = (destination / Path(*member_path.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(
                    f"Unsafe path detected in ZIP archive: {member.filename}"
                )

        archive.extractall(destination)


def find_data_yaml(root: Path):
    """Locate the dataset YAML file inside extracted directory."""
    for candidate in root.rglob("data.yaml"):
        if candidate.is_file():
            return candidate
    return None

# ============================================================
# APP HEADER
# ============================================================
st.title("🚀 YOLO Real-Time AI Detection & Training Suite")
st.markdown(
    "A computer vision platform built with **Ultralytics YOLO**, **OpenCV**, "
    "**Streamlit**, and **WebRTC**. Deploy live webcam inference or train custom models seamlessly."
)

tab_detect, tab_train, tab_docs = st.tabs(
    [
        "🔴 Real-Time Webcam Detection",
        "⚙️ Custom Model Training",
        "📖 System Overview",
    ]
)

# ============================================================
# TAB 1: REAL-TIME WEBCAM DETECTION
# ============================================================
with tab_detect:
    st.header("🎥 Live Webcam Object Detection")
    st.write(
        "Select pretrained weights or choose custom weights trained during this session. "
        "Allow camera access in your browser, then press **START**."
    )

    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        model_choice = st.selectbox(
            "📦 Select Model Weights",
            [
                "yolov8n.pt (COCO Nano - Fast)",
                "yolov8s.pt (COCO Small - Accurate)",
                "Custom Trained Model",
            ],
            key="detection_model_choice",
        )
    with settings_col2:
        conf_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.35,
            step=0.05,
            key="detection_confidence",
        )

    active_model = None
    model_label = model_choice

    try:
        if model_choice == "Custom Trained Model":
            custom_path = st.session_state.trained_model_path
            if custom_path and Path(custom_path).is_file():
                active_model = load_custom_model(custom_path)
                st.success("✅ Custom weights successfully loaded!")
            else:
                st.warning(
                    "⚠️ No custom weights found. Train a model in the Training tab first. "
                    "Falling back to `yolov8n.pt`."
                )
                active_model = load_base_model("yolov8n.pt")
                model_label = "yolov8n.pt (fallback)"
        elif model_choice.startswith("yolov8s"):
            active_model = load_base_model("yolov8s.pt")
        else:
            active_model = load_base_model("yolov8n.pt")
    except Exception as exc:
        st.error(f"❌ Could not load the selected model: {exc}")
        active_model = None

    selected_classes = []
    class_indices = None

    if active_model is not None:
        names = active_model.names
        if isinstance(names, dict):
            class_options = list(names.values())
            selected_classes = st.multiselect(
                "🔍 Filter Target Classes (leave empty to detect all)",
                options=class_options,
                default=[],
                key="detection_classes",
            )
            if selected_classes:
                class_indices = [
                    index for index, class_name in names.items()
                    if class_name in selected_classes
                ]

    if active_model is not None:
        class YOLOVideoProcessor:
            def __init__(self):
                self.model = active_model
                self.conf = conf_threshold
                self.classes = class_indices

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                results = self.model(
                    image,
                    conf=self.conf,
                    classes=self.classes,
                    verbose=False,
                )
                annotated = results[0].plot()
                return frame.from_ndarray(annotated, format="bgr24")

        webcam_col, metrics_col = st.columns([3, 1])
        with webcam_col:
            st.subheader("📹 Video Stream")
            webrtc_streamer(
                key="yolo-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=YOLOVideoProcessor,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with metrics_col:
            st.subheader("📊 Session Info")
            st.info(
                "YOLO inference runs server-side on frame data received via WebRTC."
            )
            st.markdown(f"**Model:** `{model_label}`")
            st.markdown(f"**Confidence:** `{conf_threshold:.2f}`")
            st.markdown(
                f"**Classes Active:** `{len(selected_classes)} selected`"
                if selected_classes
                else "**Classes Active:** `All Classes`"
            )
    else:
        st.warning("⚠️ The detection model could not be initialized.")

# ============================================================
# TAB 2: CUSTOM MODEL TRAINING
# ============================================================
with tab_train:
    st.header("⚙️ Custom Dataset Training Suite")
    st.write(
        "Upload a ZIP archive containing your YOLO-formatted dataset and a valid `data.yaml` configuration file."
    )

    upload_col, parameter_col = st.columns(2)

    with upload_col:
        st.subheader("1. Dataset & Base Model")
        dataset_file = st.file_uploader(
            "📂 Upload Dataset (.zip)",
            type=["zip"],
            key="training_dataset_zip",
        )
        base_model_train = st.selectbox(
            "🏗️ Base Pretrained Weights",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
        )

    with parameter_col:
        st.subheader("2. Hyperparameters")
        epochs = st.slider("🔄 Epochs", min_value=1, max_value=100, value=10)
        imgsz = st.selectbox(
            "🖼️ Image Resolution",
            options=[320, 416, 640, 1280],
            index=2,
        )
        batch_size = st.selectbox(
            "⚡ Batch Size",
            options=[4, 8, 16, 32, -1],
            index=2,
            help="Select -1 for automatic dynamic batch sizing.",
        )
        lr0 = st.number_input(
            "📈 Learning Rate (lr0)",
            min_value=0.00001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.5f",
        )

    st.markdown("---")
    st.subheader("3. Execute Training")

    if st.button("🚀 Start Training Run", type="primary", use_container_width=True):
        if dataset_file is None:
            st.warning("⚠️ Please upload a dataset ZIP archive first.")
        else:
            workspace = Path(tempfile.mkdtemp(prefix="yolo_training_"))
            zip_path = workspace / "dataset.zip"

            try:
                zip_path.write_bytes(dataset_file.getbuffer())
                with st.spinner("📦 Extracting and verifying dataset architecture..."):
                    safe_extract_zip(zip_path, workspace)

                data_yaml_path = find_data_yaml(workspace)
                if data_yaml_path is None:
                    st.error(
                        "❌ Could not locate `data.yaml` in the uploaded ZIP file. "
                        "Ensure your directory structure includes a dataset configuration file."
                    )
                else:
                    st.success(f"✅ Found dataset configuration at: `{data_yaml_path.name}`")
                    st.session_state.training_workspace = str(workspace)

                    with st.spinner("🏋️ Training YOLO model... This may take several minutes."):
                        model = YOLO(base_model_train)
                        results = model.train(
                            data=str(data_yaml_path),
                            epochs=epochs,
                            imgsz=imgsz,
                            batch=batch_size,
                            lr0=float(lr0),
                            project=str(workspace / "runs"),
                            name="custom_train",
                            exist_ok=True,
                            verbose=True,
                        )

                    save_dir = getattr(results, "save_dir", None)
                    if save_dir is None:
                        save_dir = workspace / "runs" / "custom_train"

                    best_weights = Path(save_dir) / "weights" / "best.pt"
                    if best_weights.is_file():
                        st.session_state.trained_model_path = str(best_weights)
                        st.session_state.training_logs = (
                            f"✅ Training finished successfully! Saved best weights at:\n{best_weights}"
                        )
                        st.success("🎉 Training finished successfully!")
                    else:
                        st.session_state.training_logs = (
                            "⚠️ Training execution completed, but `best.pt` was not produced."
                        )
                        st.warning(st.session_state.training_logs)

            except zipfile.BadZipFile:
                st.error("❌ The uploaded file is not a valid or readable ZIP archive.")
            except ValueError as exc:
                st.error(f"❌ ZIP Validation failed: {exc}")
            except Exception as exc:
                st.session_state.training_logs = f"❌ Training failed with error: {exc}"
                st.error(f"❌ Training failed: {exc}")

    st.markdown("### 📋 Training Output Logs")
    st.code(st.session_state.training_logs, language="text")

    weights_path = st.session_state.trained_model_path
    if weights_path and Path(weights_path).is_file():
        st.success("🎉 Custom model weights are ready for deployment!")
        with open(weights_path, "rb") as weights_file:
            st.download_button(
                label="📥 Download Best Weights (best.pt)",
                data=weights_file.read(),
                file_name="custom_yolo_best.pt",
                mime="application/octet-stream",
            )

# ============================================================
# TAB 3: SYSTEM OVERVIEW & DOCUMENTATION
# ============================================================
with tab_docs:
    st.header("📖 System Overview & Documentation")

    st.markdown(
        """
        ### 🎥 Real-Time Detection Guide
        1. Navigate to the **Real-Time Webcam Detection** tab.
        2. Choose a pre-trained COCO model or select your **Custom Trained Model**.
        3. Adjust thresholds or select target classes to isolate specific detections.
        4. Allow camera permissions when prompted by your browser and press **Start**.

        ### ⚙️ Custom Model Training Guide
        1. Organize your image and annotation files using standard YOLO format.
        2. Ensure a `data.yaml` file specifies paths and class names.
        3. Compress your dataset into a single `.zip` file.
        4. Upload the archive under **Custom Model Training**, select hyperparameters, and click **Start Training Run**.
        5. Download your exported `best.pt` file when complete.

        ### 📂 Expected Dataset Layout
        ```text
        dataset/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        ```

        ### 🛠️ Key Dependencies
        ```text
        streamlit
        streamlit-webrtc
        ultralytics
        opencv-python-headless
        Pillow
        numpy
        ```
        """
                    )eader("3. Execute Training")

    if st.button("ðŸš€ Start Training Run", type="primary"):
        if dataset_file is None:
            st.warning("Please upload a dataset ZIP archive first.")
        else:
            workspace = Path(tempfile.mkdtemp(prefix="yolo_training_"))
            zip_path = workspace / "dataset.zip"

            try:
                zip_path.write_bytes(dataset_file.getbuffer())
                with st.spinner("Validating and extracting dataset..."):
                    safe_extract_zip(zip_path, workspace)

                data_yaml_path = find_data_yaml(workspace)
                if data_yaml_path is None:
                    st.error(
                        "Could not find `data.yaml` in the uploaded ZIP. "
                        "Include a YOLO dataset configuration file."
                    )
                else:
                    st.success(f"Found dataset configuration: {data_yaml_path}")
                    st.session_state.training_workspace = str(workspace)

                    with st.spinner(
                        "Training model. This may take several minutes or longer..."
                    ):
                        model = YOLO(base_model_train)
                        results = model.train(
                            data=str(data_yaml_path),
                            epochs=epochs,
                            imgsz=imgsz,
                            batch=batch_size,
                            lr0=float(lr0),
                            project=str(workspace / "runs"),
                            name="custom_train",
                            exist_ok=True,
                            verbose=True,
                        )

                    # Ultralytics normally returns a trainer object with save_dir.
                    save_dir = getattr(results, "save_dir", None)
                    if save_dir is None:
                        save_dir = workspace / "runs" / "custom_train"

                    best_weights = Path(save_dir) / "weights" / "best.pt"
                    if best_weights.is_file():
                        st.session_state.trained_model_path = str(best_weights)
                        st.session_state.training_logs = (
                            f"Training completed successfully. Best weights: "
                            f"{best_weights}"
                        )
                        st.success("Training completed successfully.")
                    else:
                        st.session_state.training_logs = (
                            "Training finished, but best.pt was not found. "
                            "Check the training output and dataset labels."
                        )
                        st.warning(st.session_state.training_logs)

            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP archive.")
            except ValueError as exc:
                st.error(f"Dataset ZIP validation failed: {exc}")
            except Exception as exc:
                st.session_state.training_logs = f"Training failed: {exc}"
                st.error(f"Training failed: {exc}")

    st.markdown("### Training Status")
    st.text(st.session_state.training_logs)

    weights_path = st.session_state.trained_model_path
    if weights_path and Path(weights_path).is_file():
        st.success(
            "Custom weights are ready. Select 'Custom Trained Model' "
            "in the webcam detection tab to use them."
        )
        with open(weights_path, "rb") as weights_file:
            st.download_button(
                label="ðŸ“¥ Download best weights (best.pt)",
                data=weights_file.read(),
                file_name="custom_yolo_best.pt",
                mime="application/octet-stream",
            )
    elif weights_path:
        st.warning(
            "The previous weights path is no longer available. Please train "
            "again or upload weights through a future model-upload feature."
        )


# ============================================================
# TAB 3: SYSTEM DOCUMENTATION
# ============================================================
with tab_docs:
    st.header("System Overview & Instructions")

    st.markdown(
        """
        ### Real-time webcam detection
        1. Open the **Real-Time Webcam Detection** tab.
        2. Select a pretrained COCO model or custom weights.
        3. Adjust the confidence threshold and optional class filter.
        4. Press **START** in the WebRTC component and grant camera permission.
        5. Press **STOP** when finished.

        ### Custom model training
        1. Prepare a dataset in YOLO detection format.
        2. ZIP the dataset folders and include a `data.yaml` file.
        3. Upload the ZIP in **Custom Model Training**.
        4. Choose the base model, epochs, image size, batch size, and learning rate.
        5. Click **Start Training Run** and wait for training to finish.
        6. Download `best.pt`, or select the custom model in the webcam tab.

        ### Typical dataset layout
        ```text
        dataset/
        â”œâ”€â”€ data.yaml
        â”œâ”€â”€ images/
        â”‚   â”œâ”€â”€ train/
        â”‚   â””â”€â”€ val/
        â””â”€â”€ labels/
            â”œâ”€â”€ train/
            â””â”€â”€ val/
        ```

        The paths and class names in `data.yaml` must match the uploaded dataset.
        Each image generally needs a corresponding YOLO-format label file.
        Training speed depends on the available CPU/GPU, dataset size, and
        selected hyperparameters.

        ### Important deployment notes
        - Webcam access requires browser permission and working WebRTC networking.
        - A hosted app needs sufficient memory, storage, and compute for YOLO.
        - Training files are stored in temporary server storage and may be lost
          when the server restarts or the hosting platform recycles the app.
        - Do not upload private or sensitive datasets to a public deployment.
        """
    )

    st.markdown("### Installed packages")
    st.code(
        "streamlit\n"
        "streamlit-webrtc\n"
        "ultralytics\n"
        "opencv-python-headless\n"
        "Pillow\n"
        "numpy",
        language="text",
    )
