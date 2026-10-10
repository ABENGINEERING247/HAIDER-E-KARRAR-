
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)
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


# ============================================================
# PREMIUM LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fbff, #edf5ff);
        color: #183153;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, 0.92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #edf5ff);
        border-right: 1px solid #dbeafe;
    }

    [data-testid="stSidebar"] * {
        color: #183153;
    }

    .main-title {
        text-align: center;
        font-size: clamp(28px, 4vw, 46px);
        font-weight: 900;
        color: #1261b5;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }

    .sub-title {
        text-align: center;
        color: #526b8b;
        font-size: 17px;
        margin: 5px 0 22px 0;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff, #e9f4ff);
        border: 1px solid #c8e1ff;
        border-radius: 22px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(36, 99, 170, 0.08);
        margin-bottom: 18px;
    }

    .robot-card {
        background: #ffffff;
        border: 1px solid #d5e7fb;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 18px rgba(36, 99, 170, 0.07);
        margin-bottom: 14px;
    }

    .robot-avatar {
        font-size: 62px;
        line-height: 1.3;
        animation: robotFloat 3s ease-in-out infinite;
    }

    @keyframes robotFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    .section-title {
        color: #1261b5;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #526b8b;
        padding: 24px 5px;
        font-size: 13px;
        line-height: 1.9;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d5e7fb;
        padding: 14px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(36, 99, 170, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #526b8b;
    }

    div[data-testid="stMetricValue"] {
        color: #1261b5;
        font-weight: 800;
    }

    div.stButton > button {
        border-radius: 11px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #1684e8;
        transform: translateY(-1px);
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    div[data-baseweb="tab"] {
        border-radius: 10px;
    }

    .stAlert {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
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
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load a YOLO model and cache it."""
    return YOLO(str(model_path))


# ============================================================
# VIDEO PROCESSOR
# Defined at module level for WebRTC compatibility.
# ============================================================

class YOLOVideoProcessor(VideoProcessorBase):
    """Run YOLO detection on incoming webcam frames."""

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
        image = frame.to_ndarray(format="bgr24")

        try:
            prediction_args = {
                "source": image,
                "conf": self.confidence,
                "verbose": False,
            }

            if self.classes:
                prediction_args["classes"] = self.classes

            results = self.model.predict(**prediction_args)

            if results:
                image = results[0].plot()

        except Exception:
            # Keep the camera running if an individual frame fails.
            pass

        import av

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def safe_extract_zip(zip_file, destination):
    """Extract ZIP files while rejecting unsafe paths."""

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file) as archive:
        for member in archive.infolist():
            if not member.filename:
                continue

            member_path = PurePosixPath(member.filename)

            if member_path.is_absolute():
                raise ValueError("ZIP contains an absolute path.")

            if ".." in member_path.parts:
                raise ValueError("ZIP contains an unsafe path.")

            if "\\" in member.filename:
                raise ValueError("ZIP contains an unsafe path.")

            # Reject symbolic links in uploaded archives.
            unix_mode = (member.external_attr >> 16) & 0o170000

            if unix_mode == 0o120000:
                raise ValueError("ZIP contains a symbolic link.")

            target = (
                destination / Path(*member_path.parts)
            ).resolve()

            if target != destination and destination not in target.parents:
                raise ValueError("ZIP contains an unsafe file path.")

        # Extraction occurs only after all paths are validated.
        archive.extractall(destination)

    return destination


# ============================================================
# FIND DATA YAML
# ============================================================

def find_data_yaml(folder):
    """Find a dataset configuration file."""

    folder = Path(folder)

    for name in ("data.yaml", "data.yml"):
        matches = sorted(folder.rglob(name))

        for path in matches:
            if path.is_file():
                return path

    return None


# ============================================================
# DATASET CONFIGURATION VALIDATION
# ============================================================

def validate_dataset_yaml(yaml_path):
    """Check that the YAML is readable and contains class names."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is missing. Add pyyaml to requirements.txt."
        ) from exc

    yaml_path = Path(yaml_path).resolve()

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The dataset YAML must contain a mapping.")

    if "train" not in config or "val" not in config:
        raise ValueError(
            "The dataset YAML must define both 'train' and 'val'."
        )

    if "names" not in config and "nc" not in config:
        raise ValueError(
            "The dataset YAML must define 'names' or 'nc'."
        )

    # Make relative dataset paths resolve from the YAML folder.
    # Preserve absolute paths when explicitly supplied.
    config["path"] = str(yaml_path.parent)

    normalized_yaml = yaml_path.parent / "validated_data.yaml"

    with normalized_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return normalized_yaml


# ============================================================
# WELCOME POPUP
# ============================================================

@st.dialog("🤗 Welcome to YOLO AI Vision Suite!")
def welcome_popup():
    st.markdown(
        """
        <div class="hero-card">
            <div class="robot-avatar">🤖</div>
            <h2 style="color:#1261b5;">Hello, Vision Explorer!</h2>
            <p style="color:#526b8b;">
                Welcome to your intelligent computer vision laboratory.
                Detect objects, explore real-time AI, and train your
                own YOLO model.
            </p>
            <h3>👁️ See • 🎯 Detect • 🧠 Learn</h3>
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
# HEADER
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
# SIDEBAR
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
    st.write("🧠 Custom YOLO training")
    st.write("📦 Dataset ZIP upload")
    st.write("📘 Learning guide")

    st.divider()

    st.caption(
        "🔐 Upload only trusted datasets. "
        "Protect private images and trained model files."
    )


# ============================================================
# MAIN TABS
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
        "Choose a model, configure detection, and allow browser "
        "camera access. Video frames are sent to the application "
        "server for processing."
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
                "Train a custom model first. The model file must "
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
                preview_model = load_yolo_model(selected_model_path)

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
                help="Leave empty to detect every available class.",
            )

            selected_classes = [
                int(label.split(":", 1)[0])
                for label in selected_labels
            ]

        except Exception as exc:
            st.error(
                f"Unable to load the selected model: {exc}"
            )

    st.info(
        "📌 Standard YOLO weights may need to be downloaded "
        "the first time you use a model."
    )

    video_col, info_col = st.columns([3, 1])

    with video_col:
        st.markdown("### 📷 Live Camera")

        if selected_model_path:
            try:
                webrtc_streamer(
                    key="yolo-ai-vision-live",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=YOLOVideoProcessor,
                    video_processor_factory_kwargs={
                        "model_path": selected_model_path,
                        "confidence": conf_threshold,
                        "classes": selected_classes,
                    },
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

        st.markdown(
            f"**Classes:** `{len(selected_classes) if selected_classes else 'All'}`"
        )

        st.markdown("**Mode:** `Live Camera`")

        st.info(
            "💡 Good lighting and a stable camera can improve "
            "detection quality."
        )


# ============================================================
# TAB 2 — CUSTOM MODEL TRAINING
# ============================================================

with training_tab:
    st.markdown("## 🧠 Train Your Own YOLO Model")

    st.write(
        "Upload a YOLO-format dataset ZIP containing images, labels, "
        "and a valid data.yaml or data.yml configuration."
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="robot-avatar">🦾</div>
            <h3 style="color:#1261b5;">Custom AI Training Lab</h3>
            <p>📦 Upload Dataset → ⚙️ Configure → 🧠 Train → 🎯 Detect</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_dataset = st.file_uploader(
        "📦 Upload YOLO Dataset ZIP",
        type=["zip"],
        help=(
            "Include data.yaml or data.yml, images, and labels. "
            "The YAML paths must match your dataset."
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
        "⚠️ Training can consume substantial RAM, disk space, "
        "and GPU resources. Start with a small dataset."
    )

    if st.button(
        "🚀 Start YOLO Training",
        type="primary",
        use_container_width=True,
    ):
        if uploaded_dataset is None:
            st.error("Please upload a dataset ZIP file first.")

        else:
            workspace = None

            try:
                with st.spinner("📦 Preparing dataset..."):
                    workspace = tempfile.mkdtemp(
                        prefix="yolo_vision_"
                    )

                    st.session_state.training_workspace = workspace

                    zip_path = Path(workspace) / "dataset.zip"

                    with zip_path.open("wb") as file:
                        file.write(uploaded_dataset.getvalue())

                    extract_folder = Path(workspace) / "dataset"

                    safe_extract_zip(
                        zip_path,
                        extract_folder,
                    )

                    data_yaml = find_data_yaml(extract_folder)

                    if data_yaml is None:
                        raise ValueError(
                            "Could not find data.yaml or data.yml "
                            "inside the ZIP archive."
                        )

                    validated_yaml = validate_dataset_yaml(
                        data_yaml
                    )

                st.success(
                    "✅ Dataset configuration found and validated."
                )

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
                        "name": "cust
