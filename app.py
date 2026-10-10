import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from functools import partial

import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)
from ultralytics import YOLO


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="YOLO AI Vision Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PREMIUM LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff, #edf5ff);
        color: #183153;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dbeafe;
    }

    [data-testid="stSidebar"] * {
        color: #183153;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, 0.9);
    }

    .main-title {
        text-align: center;
        font-size: clamp(28px, 4vw, 46px);
        font-weight: 900;
        color: #1261b5;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 17px;
        color: #526b8b;
        margin-bottom: 24px;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff, #eaf4ff);
        border: 1px solid #c8e1ff;
        border-radius: 22px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(36, 99, 170, 0.08);
        margin-bottom: 18px;
    }

    .robot-card {
        background: #ffffff;
        border: 1px solid #d5e7fb;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 5px 18px rgba(36, 99, 170, 0.07);
        margin-bottom: 14px;
    }

    .robot-avatar {
        font-size: 62px;
        line-height: 1.4;
        animation: floatRobot 3s ease-in-out infinite;
    }

    @keyframes floatRobot {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d5e7fb;
        border-radius: 15px;
        padding: 14px;
        box-shadow: 0 4px 12px rgba(36, 99, 170, 0.05);
    }

    div[data-testid="stMetricValue"] {
        color: #1261b5;
        font-weight: 800;
    }

    div.stButton > button {
        border-radius: 11px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #526b8b;
        padding: 24px 5px;
        font-size: 13px;
        line-height: 1.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULTS = {
    "has_visited": False,
    "custom_model_path": "",
    "training_workspace": "",
    "last_training_path": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 4. YOLO MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load and cache the requested YOLO model."""
    return YOLO(str(model_path))


# ============================================================
# 5. WEBRTC VIDEO PROCESSOR
# Must be defined at module level.
# ============================================================

class YOLOVideoProcessor(VideoProcessorBase):
    """Run YOLO detection on incoming camera frames."""

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.45,
        classes=None,
    ):
        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.classes = list(classes) if classes else None
        self.model = load_yolo_model(self.model_path)

    def recv(self, frame):
        import av

        image = frame.to_ndarray(format="bgr24")

        prediction_args = {
            "source": image,
            "conf": self.confidence,
            "verbose": False,
        }

        if self.classes:
            prediction_args["classes"] = self.classes

        try:
            results = self.model.predict(**prediction_args)

            if results:
                image = results[0].plot()

        except Exception:
            # Return the original frame if inference fails.
            pass

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# 6. SAFE ZIP EXTRACTION
# ============================================================

def safe_extract_zip(zip_path, destination):
    """Validate ZIP paths before extracting a dataset."""

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()

        if len(members) > 20000:
            raise ValueError(
                "Dataset contains too many files (maximum 20,000)."
            )

        total_size = sum(item.file_size for item in members)

        if total_size > 2 * 1024 * 1024 * 1024:
            raise ValueError(
                "Dataset exceeds the 2 GB uncompressed size limit."
            )

        for member in members:
            if not member.filename:
                continue

            member_path = PurePosixPath(member.filename)

            if member_path.is_absolute():
                raise ValueError(
                    "ZIP contains an absolute path."
                )

            if ".." in member_path.parts:
                raise ValueError(
                    "ZIP contains an unsafe parent-directory path."
                )

            if "\\" in member.filename:
                raise ValueError(
                    "ZIP contains an unsafe path."
                )

            unix_mode = (member.external_attr >> 16) & 0o170000

            if unix_mode == 0o120000:
                raise ValueError(
                    "ZIP contains a symbolic link."
                )

            target = (
                destination / Path(*member_path.parts)
            ).resolve()

            if target != destination and destination not in target.parents:
                raise ValueError(
                    "ZIP contains an unsafe file path."
                )

        archive.extractall(destination)

    return destination


# ============================================================
# 7. DATASET YAML HELPERS
# ============================================================

def find_data_yaml(folder):
    """Find data.yaml or data.yml in the extracted dataset."""

    folder = Path(folder)

    for filename in ("data.yaml", "data.yml"):
        matches = sorted(folder.rglob(filename))

        for path in matches:
            if path.is_file():
                return path

    return None


def validate_dataset_yaml(yaml_path):
    """Check the basic structure of a YOLO dataset YAML."""

    import yaml

    yaml_path = Path(yaml_path).resolve()

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The dataset YAML must contain a valid mapping."
        )

    if "train" not in config or "val" not in config:
        raise ValueError(
            "The YAML must define both train and val."
        )

    if "names" not in config and "nc" not in config:
        raise ValueError(
            "The YAML must define names or nc."
        )

    # Normalize the dataset root to the YAML directory.
    config["path"] = str(yaml_path.parent)

    normalized_yaml = (
        yaml_path.parent / "validated_data.yaml"
    )

    with normalized_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return normalized_yaml


# ============================================================
# 8. WELCOME POPUP
# ============================================================

@st.dialog("🤗 Welcome to YOLO AI Vision Suite!")
def welcome_popup():
    st.markdown(
        """
        <div class="hero-card">
            <div class="robot-avatar">🤖</div>
            <h2 style="color:#1261b5;">Hello, Vision Explorer!</h2>
            <p style="color:#526b8b;">
                Welcome to your intelligent computer vision lab.
                Detect objects, explore real-time AI, and train
                your own YOLO model.
            </p>
            <h3>👁️ See · 🎯 Detect · 🧠 Learn</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "✨ Let's Get Started!",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.has_visited = True
        st.rerun()


if not st.session_state.has_visited:
    welcome_popup()


# ============================================================
# 9. HEADER AND METRICS
# ============================================================

st.markdown(
    '<div class="main-title">🤖 YOLO AI VISION SUITE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        🚀 Intelligent Computer Vision · Real-Time Detection · AI Training
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("🧠 AI Engine", "YOLO")

with metric2:
    st.metric("👁️ Vision Mode", "Real-Time")

with metric3:
    st.metric("🎯 AI Capability", "Object Detection")

st.divider()


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🤖 AI Control Center")

    st.markdown(
        """
        <div class="robot-card">
            <div class="robot-avatar">🦾</div>
            <h3 style="color:#1261b5;">VISION AI ASSISTANT</h3>
            <p>See · Detect · Learn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🛠️ Available Features")
    st.write("📹 Live camera detection")
    st.write("🎯 Object and class filtering")
    st.write("🧠 Custom model training")
    st.write("📦 Dataset ZIP upload")
    st.write("📘 User guide")

    st.divider()

    st.caption(
        "🔐 Protect private images, datasets, and model weights."
    )


# ============================================================
# 11. MAIN TABS
# ============================================================

live_tab, training_tab, guide_tab = st.tabs(
    [
        "📹 Live Detection",
        "🧠 Custom Model Training",
        "📘 User Guide",
    ]
)


# ============================================================
# TAB 1 — LIVE DETECTION
# ============================================================

with live_tab:
    st.markdown("## 📹 Real-Time AI Vision")

    st.write(
        "Select a model, adjust the confidence threshold, and "
        "allow browser camera access. Video frames are processed "
        "by the application server."
    )

    model_options = {
        "YOLOv8 Nano — Fast": "yolov8n.pt",
        "YOLOv8 Small — Balanced": "yolov8s.pt",
        "YOLOv8 Medium — More Detail": "yolov8m.pt",
    }

    model_choice = st.selectbox(
        "🧠 Select AI Model",
        options=list(model_options.keys()),
    )

    use_custom_model = st.checkbox(
        "🧩 Use my trained custom model",
        value=bool(st.session_state.custom_model_path),
    )

    if use_custom_model:
        custom_path = st.session_state.custom_model_path

        if custom_path and Path(custom_path).is_file():
            selected_model_path = custom_path
            model_label = Path(custom_path).name
        else:
            selected_model_path = None
            model_label = "Custom model unavailable"

            st.warning(
                "Train a custom model first. Its weights must "
                "still exist on the application server."
            )
    else:
        selected_model_path = model_options[model_choice]
        model_label = model_choice

    conf_threshold = st.slider(
        "🎯 Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
    )

    selected_classes = []

    if selected_model_path:
        try:
            with st.spinner("🧠 Loading YOLO model..."):
                preview_model = load_yolo_model(
                    selected_model_path
                )

            class_names = preview_model.names

            if isinstance(class_names, dict):
                class_items = list(class_names.items())
            else:
                class_items = list(enumerate(class_names))

            class_labels = [
                f"{class_id}: {class_name}"
                for class_id, class_name in class_items
            ]

            selected_labels = st.multiselect(
                "🔎 Optional: Select classes to detect",
                options=class_labels,
                help="Leave empty to detect all available classes.",
            )

            selected_classes = [
                int(label.split(":", 1)[0])
                for label in selected_labels
            ]

        except Exception as exc:
            st.error(f"Unable to load the model: {exc}")

    st.info(
        "📌 Standard YOLO weights may download automatically "
        "during first use."
    )

    video_col, info_col = st.columns([3, 1])

    with video_col:
        st.markdown("### 📷 Live Camera")

        if selected_model_path:
            try:
                processor_factory = partial(
                    YOLOVideoProcessor,
                    model_path=selected_model_path,
                    confidence=conf_threshold,
                    classes=selected_classes,
                )

                webrtc_streamer(
                    key="yolo-ai-vision-live",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=processor_factory,
                    rtc_configuration={
                        "iceServers": [
                            {
                                "urls": [
                                    "stun:stun.l.google.com:19302"
                                ]
                            }
                        ]
                    },
                    media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
                    async_processing=True,
                )

            except Exception as exc:
                st.error(
                    "Unable to initialize live video. "
                    f"Details: {exc}"
                )

        else:
            st.warning(
                "⚠️ Select a valid model before starting detection."
            )

    with info_col:
        st.markdown("### 📊 Session Details")
        st.markdown(f"**Model:** `{model_label}`")
        st.markdown(f"**Confidence:** `{conf_threshold:.2f}`")

        class_count = (
            str(len(selected_classes))
            if selected_classes
            else "All"
        )

        st.markdown(f"**Classes:** `{class_count}`")
        st.markdown("**Mode:** `Live Camera`")

        st.info(
            "💡 Good lighting and a stable camera often "
            "improve detection."
        )


# ============================================================
# TAB 2 — CUSTOM MODEL TRAINING
# ============================================================

with training_tab:
    st.markdown("## 🧠 Train Your Own YOLO Model")

    st.write(
        "Upload a ZIP containing a YOLO dataset with images, "
        "labels, and a data.yaml or data.yml configuration."
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="robot-avatar">🦾</div>
            <h3 style="color:#1261b5;">Custom AI Training Lab</h3>
            <p>📦 Upload → ⚙️ Configure → 🧠 Train → 🎯 Detect</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_dataset = st.file_uploader(
        "📦 Upload YOLO Dataset ZIP",
        type=["zip"],
        help=(
            "Include a YAML configuration, images, and labels. "
            "Ensure all dataset paths are correct."
        ),
        key="dataset_zip",
    )

    epochs = st.slider(
        "🔁 Training Epochs",
        min_value=1,
        max_value=100,
        value=10,
    )

    image_size = st.select_slider(
        "🖼️ Training Image Size",
        options=[320, 416, 512, 640, 768, 1024],
        value=640,
    )

    batch_size = st.select_slider(
        "📚 Batch Size",
        options=[1, 2, 4, 8, 16],
        value=4,
    )

    device_choice = st.selectbox(
        "⚙️ Training Device",
        options=["Auto", "CPU", "GPU (0)"],
        help="GPU training requires compatible CUDA hardware.",
    )

    st.caption(
        "⚠️ Training speed depends on server memory, GPU "
        "availability, disk space, and dataset size."
    )

    if st.button(
        "🚀 Start YOLO Training",
        type="primary",
        use_container_width=True,
    ):
        if uploaded_dataset is None:
            st.error("Please upload a dataset ZIP file first.")

        else:
            try:
                workspace = tempfile.mkdtemp(
                    prefix="yolo_vision_"
                )

                st.session_state.training_workspace = workspace

                zip_path = Path(workspace) / "dataset.zip"

                with zip_path.open("wb") as file:
                    file.write(uploaded_dataset.getvalue())

                extract_folder = Path(workspace) / "dataset"

                with st.spinner("📦 Preparing dataset..."):
                    safe_extract_zip(
                        zip_path,
                        extract_folder,
                    )

                    data_yaml = find_data_yaml(extract_folder)

                    if data_yaml is None:
                        raise ValueError(
                            "Could not find data.yaml or data.yml "
                            "inside the uploaded dataset."
                        )

                    validated_yaml = validate_dataset_yaml(
                        data_yaml
                    )

                st.success("✅ Dataset configuration validated.")

                device_map = {
                    "Auto": None,
                    "CPU": "cpu",
                    "GPU (0)": "0",
                }

                device = device_map[device_choice]

                with st.spinner(
                    "🧠 Training is running. Please wait..."
                ):
                    base_model = YOLO("yolov8n.pt")

                    train_args = {
                        "data": str(validated_yaml),
                        "epochs": epochs,
                        "imgsz": image_size,
                        "batch": batch_size,
                        "project": str(
                            Path(workspace) / "runs"
                        ),
                        "name": "custom_training",
                        "exist_ok": True,
                        "verbose": True,
                    }

                    if device is not None:
                        train_args["device"] = device

                    base_model.train(**train_args)

                run_dir = (
                    Path(workspace)
                    / "runs"
                    / "custom_training"
                )

                best_weights = (
                    run_dir / "weights" / "best.pt"
                )

                if best_weights.is_file():
                    st.session_state.custom_model_path = str(
                        best_weights
                    )

                    st.session_state.last_training_path = str(
                        best_weights
                    )

                    st.success(
                        "🎉 Training completed! Your custom model is ready."
                    )

                    st.code(str(best_weights))

                    st.download_button(
                        label="📥 Download best.pt",
                        data=best_weights.read_bytes(),
                        file_name="best.pt",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )

                    st.info(
                        "Open Live Detection and enable "
                        "'Use my trained custom model'."
                    )

                else:
                    st.warning(
                        "Training finished, but best.pt was not found. "
                        "Check the training output and dataset."
                    )

            except zipfile.BadZipFile:
                st.error(
                    "❌ The uploaded file is not a valid ZIP archive."
                )

            except Exception as exc:
                st.error(f"❌ Training failed: {exc}")

                st.info(
                    "Check the YAML paths, dataset structure, "
                    "installed dependencies, and server memory."
                )

    st.divider()

    st.markdown("### 📂 Existing Custom Model")

    current_model = st.session_state.custom_model_path

    if current_model and Path(current_model).is_file():
        st.success(
            f"Loaded custom model: `{Path(current_model).name}`"
        )

        if st.button("🗑️ Clear Custom Model Selection"):
            st.session_state.custom_model_path = ""
            st.session_state.last_training_path = ""
            st.rerun()

    else:
        st.caption("No custom model is currently selected.")


# ============================================================
# TAB 3 — USER GUIDE
# ============================================================

with guide_tab:
    st.markdown("## 📘 YOLO AI Vision Suite — User Guide")

    with st.expander(
        "📹 1. Live Object Detection",
        expanded=True,
    ):
        st.markdown(
            """
            1. Open Live Detection.
            2. Select a YOLO model.
            3. Adjust the confidence threshold.
            4. Optionally select object classes.
            5. Allow browser camera access.
            6. Start the camera in the video panel.
            """
        )

    with st.expander("🧠 2. Custom Model Training"):
        st.markdown(
            """
            1. Prepare a YOLO-format dataset.
            2. Include data.yaml or data.yml.
            3. Upload the dataset as a ZIP file.
            4. Select epochs, image size, and batch size.
            5. Click Start YOLO Training.
            6. Download best.pt after training.
            7. Enable the custom model in Live Detection.

            Download and back up your weights before a restart
            or redeployment.
            """
        )

    with st.expander("📦 3. Dataset Structure"):
        st.code(
            """
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
            """,
            language="text",
        )

        st.markdown(
            """
            The YAML must define valid train and val paths,
            together with class names or the number of classes.
            The paths must match your extracted dataset structure.
            """
        )

    with st.expander("🛠️ 4. Troubleshooting"):
        st.markdown(
            """
            - **Camera not starting:** Check HTTPS, browser
              permissions, and WebRTC network connectivity.
            - **Model download failure:** Check internet access.
            - **Training error:** Verify YAML paths and labels.
            - **Out of memory:** Reduce image size or batch size.
            - **GPU unavailable:** Select CPU or Auto.
            - **Missing package:** Check requirements.txt.
            """
        )

    with st.expander("🔐 5. Privacy and Security"):
        st.markdown(
            """
            - Upload only datasets you are authorized to use.
            - Avoid uploading sensitive or confidential images.
            - Webcam frames are transmitted to the app server.
            - Use HTTPS for deployment.
            - Never execute scripts from uploaded datasets.
            - Protect downloaded weights and private files.
            """
        )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🤖 <b>YOLO AI Vision Suite</b><br>
        🚀 Computer Vision · Object Detection · AI Model Training<br>
        ✨ Designed with ❤️ using Streamlit and Ultralytics YOLO
    </div>
    """,
    unsafe_allow_html=True,
)
