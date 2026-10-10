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
    page_title="YOLO AI Detection & Training Suite",
    page_icon="ðŸ¤–",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 6px 6px 0 0;
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

# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "trained_model_path": None,
    "training_logs": "No training run has been started.",
    "training_workspace": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL HELPERS
# ============================================================
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_base_model(model_name: str):
    """Cache standard pretrained models between Streamlit reruns."""
    return YOLO(model_name)


def load_custom_model(weights_path: str):
    """Load custom weights without caching a potentially stale path."""
    return YOLO(weights_path)


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract ZIP files while rejecting paths that escape destination."""
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            # ZIP paths use forward slashes, including on Windows.
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"Unsafe path found in ZIP archive: {member.filename}"
                )

            target = (destination / Path(*member_path.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(
                    f"Unsafe path found in ZIP archive: {member.filename}"
                )

        archive.extractall(destination)


def find_data_yaml(root: Path):
    """Find the first data.yaml file under the extracted dataset directory."""
    for candidate in root.rglob("data.yaml"):
        if candidate.is_file():
            return candidate
    return None


# ============================================================
# APP HEADER AND NAVIGATION
# ============================================================
st.title("ðŸš€ YOLO Real-Time AI Detection & Training Suite")
st.markdown(
    "A computer-vision application built with **Ultralytics YOLO**, "
    "**OpenCV**, **Streamlit**, and **WebRTC**. Run live webcam inference "
    "or train a model on a dataset in YOLO format."
)

tab_detect, tab_train, tab_docs = st.tabs(
    [
        "ðŸ”´ Real-Time Webcam Detection",
        "âš™ï¸ Custom Model Training",
        "ðŸ“– System Overview",
    ]
)

# ============================================================
# TAB 1: REAL-TIME WEBCAM DETECTION
# ============================================================
with tab_detect:
    st.header("Live Webcam Object Detection")
    st.write(
        "Choose a pretrained COCO model or a model trained in this session. "
        "Allow camera access in your browser, then press START."
    )

    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        model_choice = st.selectbox(
            "Select Model Weights",
            [
                "yolov8n.pt (COCO)",
                "yolov8s.pt (COCO)",
                "Custom Trained Model",
            ],
            key="detection_model_choice",
        )
    with settings_col2:
        conf_threshold = st.slider(
            "Confidence Threshold",
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
                st.success("Custom model loaded.")
            else:
                st.warning(
                    "No custom weights are available. Train a model in the "
                    "Custom Model Training tab first. Using yolov8n.pt for now."
                )
                active_model = load_base_model("yolov8n.pt")
                model_label = "yolov8n.pt (fallback)"
        elif model_choice.startswith("yolov8s"):
            active_model = load_base_model("yolov8s.pt")
        else:
            active_model = load_base_model("yolov8n.pt")
    except Exception as exc:
        st.error(f"Could not load the selected model: {exc}")
        active_model = None

    selected_classes = []
    class_indices = None

    if active_model is not None:
        names = active_model.names
        if isinstance(names, dict):
            class_options = list(names.values())
            selected_classes = st.multiselect(
                "Filter classes (leave empty to detect all classes)",
                options=class_options,
                default=[],
                key="detection_classes",
            )
            if selected_classes:
                class_indices = [
                    index for index, class_name in names.items()
                    if class_name in selected_classes
                ]
        else:
            st.info("Class filtering is unavailable for this model.")

    if active_model is not None:
        # Keep this processor class local to the selected model/configuration.
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
            st.subheader("Live Feed")
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
            st.subheader("Detection Settings")
            st.info(
                "YOLO inference runs on the server that hosts this Streamlit "
                "application. A stable network connection is required."
            )
            st.markdown(f"**Active model:** `{model_label}`")
            st.markdown(f"**Confidence:** `{conf_threshold:.2f}`")
            st.markdown(
                f"**Class filter:** `{len(selected_classes)} selected`"
                if selected_classes
                else "**Class filter:** `All model classes`"
            )
    else:
        st.warning("The detection model could not be initialized.")


# ============================================================
# TAB 2: CUSTOM MODEL TRAINING
# ============================================================
with tab_train:
    st.header("Custom Dataset Training Suite")
    st.write(
        "Upload a ZIP archive containing a YOLO-format dataset and a "
        "`data.yaml` file. Training runs synchronously in this Streamlit "
        "session, so the page may remain busy until training finishes."
    )

    upload_col, parameter_col = st.columns(2)

    with upload_col:
        st.subheader("1. Dataset Upload")
        dataset_file = st.file_uploader(
            "Upload dataset ZIP archive",
            type=["zip"],
            key="training_dataset_zip",
        )
        base_model_train = st.selectbox(
            "Base pretrained model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
        )

    with parameter_col:
        st.subheader("2. Hyperparameters")
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=10)
        imgsz = st.selectbox(
            "Image size",
            options=[320, 416, 640, 1280],
            index=2,
        )
        batch_size = st.selectbox(
            "Batch size",
            options=[4, 8, 16, 32, -1],
            index=2,
            help="Use -1 for Ultralytics automatic batch sizing where supported.",
        )
        lr0 = st.number_input(
            "Initial learning rate (lr0)",
            min_value=0.00001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.5f",
        )

    st.markdown("---")
    st.subheader("3. Execute Training")

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
