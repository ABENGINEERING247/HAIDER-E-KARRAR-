
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
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


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f, #101d33, #07111f);
        color: #f2f7ff;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 900;
        color: #48e5ff;
        margin-bottom: 0;
    }

    .sub-title {
        text-align: center;
        color: #d5e7ff;
        font-size: 17px;
        margin-top: 5px;
    }

    .robot-card {
        background: rgba(25, 48, 77, 0.8);
        border: 1px solid #36cfff;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
    }

    .robot-avatar {
        font-size: 62px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    .section-title {
        color: #48e5ff;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #b8cce4;
        padding: 20px 5px;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: rgba(25, 48, 77, 0.65);
        border: 1px solid #274c70;
        padding: 12px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "has_visited": False,
    "custom_model_path": "",
    "training_logs": [],
    "training_workspace": "",
    "last_training_path": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# WELCOME POPUP
# ============================================================

if not st.session_state.has_visited:

    @st.dialog("🤗 Welcome to YOLO AI Vision Suite!")
    def welcome_popup():
        st.markdown(
            """
            <div class="robot-card">
                <div class="robot-avatar">🤖</div>
                <h2>Welcome, Vision Explorer!</h2>
                <p>
                    Discover real-time object detection,
                    computer vision, and custom YOLO training.
                </p>
                <p>🚀 Let's build something intelligent!</p>
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

    welcome_popup()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load and cache a YOLO model."""
    return YOLO(model_path)


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def safe_extract_zip(zip_file, destination):
    """
    Extract ZIP contents while preventing path traversal.
    Returns the destination directory.
    """
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

            target = (destination / Path(*member_path.parts)).resolve()

            if target != destination and destination not in target.parents:
                raise ValueError("ZIP contains an unsafe file path.")

        archive.extractall(destination)

    return destination


# ============================================================
# FIND YOLO DATA CONFIGURATION
# ============================================================

def find_data_yaml(folder):
    """Find data.yaml or data.yml inside a dataset folder."""
    folder = Path(folder)

    matches = list(folder.rglob("data.yaml"))
    matches += list(folder.rglob("data.yml"))

    for path in matches:
        if path.is_file():
            return path

    return None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 YOLO AI VISION SUITE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    '🚀 Intelligent Computer Vision • Real-Time Detection • AI Training'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

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
            <b>VISION AI ASSISTANT</b>
            <p>See • Detect • Learn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🛠️ Available Features")
    st.write("📹 Live camera detection")
    st.write("🎯 Object and class filtering")
    st.write("🧠 Custom YOLO model training")
    st.write("📦 Dataset ZIP upload")
    st.write("📘 Learning guide")

    st.divider()

    st.caption("🔐 Never publish API keys, passwords, or private data.")


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
# TAB 1: LIVE DETECTION
# ============================================================

with live_tab:

    st.markdown("## 📹 Real-Time AI Vision")
    st.write(
        "Allow camera access to detect objects using a YOLO model. "
        "Camera frames are processed by the app's Python server."
    )

    model_options = {
        "YOLOv8 Nano — Fast": "yolov8n.pt",
        "YOLOv8 Small — Balanced": "yolov8s.pt",
        "YOLOv8 Medium — More Detail": "yolov8m.pt",
    }

    model_choice = st.selectbox(
        "🧠 Select AI Model",
        list(model_options.keys()),
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
            model_label = "Custom model not available"
            st.warning(
                "Please train or upload a custom model in the "
                "Custom Model Training tab."
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
                help="Leave empty to detect all available classes.",
            )

            selected_classes = [
                int(label.split(":", 1)[0])
                for label in selected_labels
            ]

        except Exception as exc:
            st.error(f"Unable to load the selected model: {exc}")

    st.info(
        "📌 First-time use of a standard YOLO model may download "
        "model weights. An internet connection may be required."
    )

    video_col, info_col = st.columns([3, 1])

    with video_col:
        st.markdown("### 📷 Live Camera")

        if selected_model_path:

            class YOLOVideoProcessor:
                """Process incoming webcam frames with YOLO."""

                def __init__(
                    self,
                    model_path,
                    confidence,
                    classes,
                ):
                    self.model_path = model_path
                    self.confidence = confidence
                    self.classes = classes

                    self.model = load_yolo_model(self.model_path)

                def recv(self, frame):
                    image = frame.to_ndarray(format="bgr24")

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

                    return type(frame).from_ndarray(
                        image,
                        format="bgr24",
                    )

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
                "⚠️ Select a valid model before starting live detection."
            )

    with info_col:
        st.markdown("### 📊 Session Details")

        st.info(
            "🧠 YOLO analyzes the video frames received by the app."
        )

        st.markdown(f"**Model:** `{model_label}`")
        st.markdown(f"**Confidence:** `{conf_threshold:.2f}`")

        if selected_classes:
            st.markdown(
                f"**Classes:** `{len(selected_classes)} selected`"
            )
        else:
            st.markdown("**Classes:** `All classes`")

        st.markdown("**Mode:** `Live Camera`")


# ============================================================
# TAB 2: CUSTOM MODEL TRAINING
# ============================================================

with training_tab:

    st.markdown("## 🧠 Train Your Own YOLO Model")

    st.write(
        "Upload a ZIP file containing a YOLO dataset. "
        "The dataset should include a data.yaml or data.yml file."
    )

    st.markdown(
        """
        <div class="robot-card">
            <div class="robot-avatar">🦾</div>
            <h3>Custom AI Training Lab</h3>
            <p>Upload dataset → Configure training → Train → Use model</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_dataset = st.file_uploader(
        "📦 Upload YOLO Dataset ZIP",
        type=["zip"],
        help=(
            "Expected contents usually include images, labels, "
            "and a data.yaml or data.yml file."
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
        help="GPU training requires a compatible CUDA setup.",
    )

    st.caption(
        "Training speed depends on your server hardware, "
        "dataset size, and available memory."
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
                with st.spinner("📦 Preparing dataset..."):

                    workspace = tempfile.mkdtemp(
                        prefix="yolo_vision_"
                    )

                    st.session_state.training_workspace = workspace

                    zip_path = Path(workspace) / "dataset.zip"

                    with open(zip_path, "wb") as file:
                        file.write(uploaded_dataset.getvalue())

                    extract_folder = Path(workspace) / "dataset"

                    safe_extract_zip(
                        zip_path,
                        extract_folder,
                    )

                    data_yaml = find_data_yaml(extract_folder)

                if data_yaml is None:
                    st.error(
                        "❌ Could not find data.yaml or data.yml "
                        "inside the uploaded dataset."
                    )

                else:
                    st.success(
                        f"✅ Dataset configuration found: {data_yaml.name}"
                    )

                    device = {
                        "Auto": None,
                        "CPU": "cpu",
                        "GPU (0)": "0",
                    }[device_choice]

                    with st.spinner(
                        "🧠 Training is running. Please wait..."
                    ):
                        base_model = YOLO("yolov8n.pt")

                        train_args = {
                            "data": str(data_yaml),
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

                    best_weights = run_dir / "weights" / "best.pt"

                    if best_weights.is_file():
                        st.session_state.custom_model_path = str(
                            best_weights
                        )
                        st.session_state.last_training_path = str(
                            best_weights
                        )

                        st.success(
                            "🎉 Training completed! Your custom model "
                            "is ready to use."
                        )

                        st.code(str(best_weights))

                        with open(best_weights, "rb") as model_file:
                            st.download_button(
                                label="📥 Download best.pt",
                                data=model_file.read(),
                                file_name="best.pt",
                                mime="application/octet-stream",
                                use_container_width=True,
                            )

                        st.info(
                            "Go to the Live Detection tab and enable "
                            "'Use my trained custom model'."
                        )

                    else:
                        st.warning(
                            "Training finished, but best.pt was not "
                            "found. Check the training output and dataset."
                        )

            except zipfile.BadZipFile:
                st.error(
                    "The uploaded file is not a valid ZIP archive."
                )

            except Exception as exc:
                st.error(f"❌ Training failed: {exc}")

                st.info(
                    "Check that the dataset YAML paths are correct, "
                    "all required files are included, and the server "
                    "has sufficient memory and disk space."
                )

    st.divider()

    st.markdown("### 📂 Existing Custom Model")

    current_model = st.session_state.custom_model_path

    if current_model and Path(current_model).is_file():
        st.success(f"Loaded custom model: `{Path(current_model).name}`")

        if st.button("🗑️ Clear Custom Model Selection"):
            st.session_state.custom_model_path = ""
            st.rerun()
    else:
        st.caption("No custom model is currently selected.")


# ============================================================
# TAB 3: USER GUIDE
# ============================================================

with guide_tab:

    st.markdown("## 📘 YOLO AI Vision Suite — User Guide")

    st.markdown(
        """
        ### 🤖 1. Live Object Detection

        1. Open the **Live Detection** tab.
        2. Choose a YOLO model.
        3. Adjust the confidence threshold.
        4. Optionally select specific object classes.
        5. Allow browser camera access when prompted.
        6. Start the camera in the video panel.

        ### 🧠 2. Custom Model Training

        1. Prepare a YOLO-format dataset.
        2. Include `data.yaml` or `data.yml`.
        3. Upload the dataset as a ZIP file.
        4. Select epochs, image size, and batch size.
        5. Click **Start YOLO Training**.
        6. Download `best.pt` when training completes.
        7. Enable your custom model in Live Detection.

        ### 📦 3. Typical YOLO Dataset Structure

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

        Your `data.yaml` must correctly describe the dataset paths
        and class names. Follow the dataset format expected by
        Ultralytics YOLO.

        ### ⚠️ 4. Troubleshooting

        - **Camera not starting:** Check browser permissions and HTTPS.
        - **Model download error:** Check internet access and server permissions.
        - **Training error:** Verify the YAML file and image/label paths.
        - **Out of memory:** Reduce batch size or image size.
        - **GPU unavailable:** Choose CPU or Auto, depending on your setup.

        ### 🔐 5. Privacy and Security

        - Only upload datasets you are authorized to use.
        - Avoid uploading confidential or sensitive images.
        - Do not expose private files or model weights publicly.
        - Use HTTPS when deploying the app.
        - Treat uploaded ZIP files and model files as untrusted inputs.
        - Remember that server-side webcam processing sends camera frames
          from the browser to the application server.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🤖 <b>YOLO AI Vision Suite</b><br>
        🚀 Computer Vision • Object Detection • AI Model Training<br><br>
        Designed with ❤️ using Streamlit and Ultralytics YOLO
    </div>
    """,
    unsafe_allow_html=True,
)
