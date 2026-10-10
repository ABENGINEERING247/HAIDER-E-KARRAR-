""" YOLO AI Vision Suite Run with: streamlit run app.py Features: - Animated AI avatar and first-visit welcome popup - YOLO pretrained/custom model webcam detection over WebRTC - Upload a YOLO-format ZIP dataset and train custom weights - Download trained best.pt weights """

import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="YOLO AI Vision Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING + AVATAR
# ============================================================
st.markdown(
    """ <style> .stApp { background: radial-gradient(circle at 10% 5%, rgba(40, 90, 160, .18), transparent 28%), linear-gradient(135deg, #0b1020 0%, #101827 50%, #090d16 100%); color: #edf4ff; } .stTabs [data-baseweb="tab-list"] { gap: 10px; } .stTabs [data-baseweb="tab"] { min-height: 48px; white-space: pre-wrap; border-radius: 10px 10px 0 0; background: rgba(255,255,255,.05); font-weight: 700; padding: 10px 16px; } .stTabs [aria-selected="true"] { background: rgba(70,140,255,.17); border-bottom: 3px solid #58a6ff; } .hero { border: 1px solid rgba(120,170,255,.35); background: linear-gradient(120deg, rgba(33,58,100,.8), rgba(18,26,43,.8)); border-radius: 20px; padding: 22px; margin: 8px 0 22px 0; box-shadow: 0 12px 36px rgba(0,0,0,.2); } .avatar-row { display:flex; align-items:center; gap:18px; } .avatar { width:86px; height:86px; border-radius:50%; flex:0 0 86px; display:flex; align-items:center; justify-content:center; font-size:48px; background: radial-gradient(circle at 35% 25%, #d7f7ff, #64a7ff 45%, #5b39b7 80%); box-shadow: 0 0 0 5px rgba(111,170,255,.12), 0 0 28px rgba(80,140,255,.5); animation: floaty 3s ease-in-out infinite; } .avatar-small { display:inline-flex; width:38px; height:38px; align-items:center; justify-content:center; border-radius:50%; background:linear-gradient(135deg,#78e8ff,#8c7bff); font-size:22px; vertical-align:middle; margin-right:8px; animation: floaty 2.5s ease-in-out infinite; } @keyframes floaty { 0%,100% { transform:translateY(0px) rotate(-2deg); } 50% { transform:translateY(-7px) rotate(2deg); } } .pill { display:inline-block; border:1px solid rgba(120,180,255,.45); border-radius:999px; padding:5px 10px; margin:4px 5px 0 0; color:#cde4ff; font-size:12px; background:rgba(60,110,190,.13); } .welcome-text { color:#dce8ff; line-height:1.65; } </style> """,
    unsafe_allow_html=True,
)

# ============================================================
# STATE + WELCOME POPUP
# ============================================================
if "has_visited" not in st.session_state:
    st.session_state.has_visited = False
if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = None
if "training_logs" not in st.session_state:
    st.session_state.training_logs = "✨ No training run has been started yet."
if "training_workspace" not in st.session_state:
    st.session_state.training_workspace = None

@st.dialog("🤗 Welcome to YOLO AI Vision Suite!", width="large")
def welcome_dialog():
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            """ <div style="text-align:center;padding:12px"> <div style="font-size:86px;animation:floaty 2.5s ease-in-out infinite">🤖</div> <h3 style="margin-top:0">Vision Buddy</h3> <p>👁️ I help you explore computer vision!</p> </div> """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### 👋 Hello, Vision Explorer!")
        st.markdown(
            """ <div class="welcome-text"> Welcome to your AI-powered computer vision lab. 🚀<br> Detect objects with a webcam, train your own YOLO model, and export your trained weights — all from one workspace. <br><br> <span class="pill">🎥 Live Detection</span> <span class="pill">🧠 Custom Training</span> <span class="pill">📦 Model Export</span> <span class="pill">🛡️ Safer Dataset Upload</span> </div> """,
            unsafe_allow_html=True,
        )
    st.info("💡 Tip: Allow camera access when using live detection. Training speed depends on your CPU/GPU.")
    if st.button("🚀 Let's Get Started!", type="primary", use_container_width=True):
        st.session_state.has_visited = True
        st.rerun()

if not st.session_state.has_visited:
    welcome_dialog()

# ============================================================
# UTILITIES
# ============================================================
@st.cache_resource(show_spinner="⚡ Loading YOLO model...")
def load_base_model(model_name: str):
    return YOLO(model_name)

def load_custom_model(weights_path: str):
    return YOLO(weights_path)

def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a dataset ZIP after checking every member for path traversal."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        archive.extractall(destination)

def find_data_yaml(root: Path):
    """Find data.yaml or data.yml in the uploaded dataset."""
    for pattern in ("data.yaml", "data.yml"):
        for candidate in root.rglob(pattern):
            if candidate.is_file():
                return candidate
    return None

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """ <div class="hero"> <div class="avatar-row"> <div class="avatar">🤖</div> <div> <h1 style="margin:0;color:#f4f8ff">🚀 YOLO AI Vision Suite</h1> <div style="font-size:17px;color:#c5d9ff;margin-top:6px"> Real-Time Detection • Custom Training • Intelligent Vision </div> <div style="margin-top:8px"> <span class="pill">⚡ Ultralytics YOLO</span> <span class="pill">🎥 WebRTC Webcam</span> <span class="pill">🧠 Custom Models</span> <span class="pill">📥 Export Weights</span> </div> </div> </div> </div> """,
    unsafe_allow_html=True,
)

st.caption("👋 Welcome, visitors! Explore computer vision, experiment responsibly, and have fun learning. ✨")

tab_detect, tab_train, tab_docs = st.tabs(
    ["🔴 Live Detection", "⚙️ Custom Model Training", "📚 Guide & System Info"]
)

# ============================================================
# TAB 1: WEBCAM DETECTION
# ============================================================
with tab_detect:
    st.header("🎥 Real-Time Webcam Object Detection")
    st.write("Choose model weights, set a confidence threshold, then press **START** and grant camera access.")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "📦 Model Weights",
            ["yolov8n.pt — Nano (fast)", "yolov8s.pt — Small (more capable)", "Custom Trained Model"],
            key="detection_model_choice",
        )
    with col2:
        conf_threshold = st.slider(
            "🎯 Confidence Threshold", min_value=0.10, max_value=1.00,
            value=0.35, step=0.05, key="detection_confidence",
        )

    active_model = None
    model_label = model_choice
    try:
        if model_choice == "Custom Trained Model":
            custom_path = st.session_state.trained_model_path
            if custom_path and Path(custom_path).is_file():
                active_model = load_custom_model(custom_path)
                st.success("✅ Your custom weights are loaded.")
            else:
                st.warning("⚠️ No custom weights found yet. Using `yolov8n.pt` as a fallback.")
                active_model = load_base_model("yolov8n.pt")
                model_label = "yolov8n.pt (fallback)"
        elif model_choice.startswith("yolov8s"):
            active_model = load_base_model("yolov8s.pt")
        else:
            active_model = load_base_model("yolov8n.pt")
    except Exception as exc:
        st.error(f"❌ Could not load the model: {exc}")

    selected_classes = []
    class_indices = None
    if active_model is not None:
        names = active_model.names
        if isinstance(names, dict):
            class_options = list(names.values())
            selected_classes = st.multiselect(
                "🔎 Filter classes (leave blank to detect all)",
                options=class_options,
                default=[],
                key="detection_classes",
            )
            if selected_classes:
                class_indices = [idx for idx, name in names.items() if name in selected_classes]

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

        video_col, info_col = st.columns([3, 1])
        with video_col:
            st.subheader("📹 Live Video")
            webrtc_streamer(
                key="yolo-ai-vision-live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=YOLOVideoProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        with info_col:
            st.subheader("📊 Session Details")
            st.info("🧠 Inference runs on the server using frames received over WebRTC.")
            st.markdown(f"**Model:** `{model_label}`")
            st.markdown(f"**Confidence:** `{conf_threshold:.2f}`")
            st.markdown(
                f"**Classes:** `{len(selected_classes)} selected`"
                if selected_classes else "**Classes:** `All classes`"
            )
    else:
        st.warning("⚠️ The selected model could not be initialized. Check your deployment logs and dependencies.")

# ============================================================
# TAB 2: TRAINING
# ============================================================
with tab_train:
    st.header("⚙️ Train Your Own YOLO Model")
    st.write("Upload a ZIP archive containing YOLO-format images, labels, and `data.yaml` or `data.yml`.")

    upload_col, params_col = st.columns(2)
    with upload_col:
        st.subheader("📂 Dataset & Base Model")
        dataset_file = st.file_uploader(
            "Upload dataset ZIP", type=["zip"], key="training_dataset_zip",
            help="Use a YOLO detection dataset with images/, labels/, and data.yaml.",
        )
        base_model_train = st.selectbox(
            "🏗️ Base pretrained weights", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0
        )
    with params_col:
        st.subheader("🎛️ Training Parameters")
        epochs = st.slider("🔄 Epochs", min_value=1, max_value=100, value=10)
        imgsz = st.selectbox("🖼️ Image size", [320, 416, 640, 1280], index=2)
        batch_size = st.selectbox("⚡ Batch size", [4, 8, 16, 32, -1], index=1,
                                  help="Choose -1 for automatic batch sizing.")
        lr0 = st.number_input("📈 Learning rate", min_value=0.00001, max_value=1.0,
                              value=0.01, step=0.001, format="%.5f")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if dataset_file is None:
            st.warning("📌 Please upload a dataset ZIP first.")
        else:
            workspace = Path(tempfile.mkdtemp(prefix="yolo_training_"))
            zip_path = workspace / "dataset.zip"
            try:
                zip_path.write_bytes(dataset_file.getbuffer())
                with st.spinner("📦 Validating and extracting dataset..."):
                    safe_extract_zip(zip_path, workspace)

                data_yaml_path = find_data_yaml(workspace)
                if data_yaml_path is None:
                    st.error("❌ Could not find `data.yaml` or `data.yml` in the ZIP.")
                else:
                    st.success(f"✅ Dataset configuration found: `{data_yaml_path.name}`")
                    st.session_state.training_workspace = str(workspace)
                    with st.spinner("🏋️ Training in progress. This may take several minutes..."):
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
                    save_dir = Path(getattr(results, "save_dir", workspace / "runs" / "custom_train"))
                    best_weights = save_dir / "weights" / "best.pt"
                    if best_weights.is_file():
                        st.session_state.trained_model_path = str(best_weights)
                        st.session_state.training_logs = f"✅ Training completed!\nBest weights: {best_weights}"
                        st.success("🎉 Training completed successfully. Your model is ready!")
                    else:
                        st.session_state.training_logs = (
                            "⚠️ Training finished, but best.pt was not found. Check dataset labels and training logs."
                        )
                        st.warning(st.session_state.training_logs)
            except zipfile.BadZipFile:
                st.error("❌ This is not a valid ZIP archive.")
                st.session_state.training_logs = "Training failed: invalid ZIP archive."
            except ValueError as exc:
                st.error(f"🛡️ Dataset archive validation failed: {exc}")
                st.session_state.training_logs = f"Training failed: {exc}"
            except Exception as exc:
                st.error(f"❌ Training failed: {exc}")
                st.session_state.training_logs = f"Training failed: {exc}"

    st.markdown("### 🧾 Training Logs")
    st.code(st.session_state.training_logs, language="text")

    weights_path = st.session_state.trained_model_path
    if weights_path and Path(weights_path).is_file():
        st.success("📦 Custom model weights are ready to download.")
        try:
            with open(weights_path, "rb") as weights_file:
                st.download_button(
                    "📥 Download best.pt",
                    data=weights_file.read(),
                    file_name="custom_yolo_best.pt",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        except OSError as exc:
            st.warning(f"Could not read weights file: {exc}")

# ============================================================
# TAB 3: DOCUMENTATION
# ============================================================
with tab_docs:
    st.header("📚 Quick Start Guide")
    st.markdown(
        """ ### 🎥 Live detection 1. Open **Live Detection**. 2. Select a pretrained model or your custom trained model. 3. Set the confidence threshold and optionally choose target classes. 4. Press **START** in the video widget and allow camera access. 5. Press **STOP** when you finish. ### ⚙️ Custom training 1. Prepare a YOLO-format dataset with images and labels. 2. Include a `data.yaml` or `data.yml` file. 3. ZIP the dataset folder and upload it in **Custom Model Training**. 4. Choose the base model and training settings. 5. Click **Start Training** and wait for completion. 6. Download `best.pt` and use it for deployment. ### 📁 Typical dataset layout ```text dataset/ ├── data.yaml ├── images/ │ ├── train/ │ └── val/ └── labels/ ├── train/ └── val/ ``` ### 🧰 Dependencies - `streamlit` - `streamlit-webrtc` - `ultralytics` - `opencv-python-headless` - `numpy` - `av` ### 🛡️ Deployment notes - Browser webcam access requires HTTPS in hosted deployments and permission from the visitor. - Training may be too resource-intensive for some free hosting tiers. - Uploaded datasets and trained weights use temporary server storage and may disappear after a restart. - Do not upload sensitive or confidential datasets to a public app. """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9db4d7;padding:10px'>"
    "🤖 YOLO AI Vision Suite • Built for curious minds • Detect responsibly ✨"
    "</div>",
    unsafe_allow_html=True,
                                                                )             video_processor_factory=YOLOVideoProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        with info_col:
            st.subheader("📊 Session Details")
            st.info("🧠 Inference runs on the server using frames received over WebRTC.")
            st.markdown(f"**Model:** `{model_label}`")
            st.markdown(f"**Confidence:** `{conf_threshold:.2f}`")
            st.markdown(
                f"**Classes:** `{len(selected_classes)} selected`"
                if selected_classes else "**Classes:** `All classes`"
            )
    else:
        st.warning("⚠️ The selected model could not be initialized. Check your deployment logs and dependencies.")

# ============================================================
# TAB 2: TRAINING
# ============================================================
with tab_train:
    st.header("⚙️ Train Your Own YOLO Model")
    st.write("Upload a ZIP archive containing YOLO-format images, labels, and `data.yaml` or `data.yml`.")

    upload_col, params_col = st.columns(2)
    with upload_col:
        st.subheader("📂 Dataset & Base Model")
        dataset_file = st.file_uploader(
            "Upload dataset ZIP", type=["zip"], key="training_dataset_zip",
            help="Use a YOLO detection dataset with images/, labels/, and data.yaml.",
        )
        base_model_train = st.selectbox(
            "🏗️ Base pretrained weights", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0
        )
    with params_col:
        st.subheader("🎛️ Training Parameters")
        epochs = st.slider("🔄 Epochs", min_value=1, max_value=100, value=10)
        imgsz = st.selectbox("🖼️ Image size", [320, 416, 640, 1280], index=2)
        batch_size = st.selectbox("⚡ Batch size", [4, 8, 16, 32, -1], index=1,
                                  help="Choose -1 for automatic batch sizing.")
        lr0 = st.number_input("📈 Learning rate", min_value=0.00001, max_value=1.0,
                              value=0.01, step=0.001, format="%.5f")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if dataset_file is None:
            st.warning("📌 Please upload a dataset ZIP first.")
        else:
            workspace = Path(tempfile.mkdtemp(prefix="yolo_training_"))
            zip_path = workspace / "dataset.zip"
            try:
                zip_path.write_bytes(dataset_file.getbuffer())
                with st.spinner("📦 Validating and extracting dataset..."):
                    safe_extract_zip(zip_path, workspace)

                data_yaml_path = find_data_yaml(workspace)
                if data_yaml_path is None:
                    st.error("❌ Could not find `data.yaml` or `data.yml` in the ZIP.")
                else:
                    st.success(f"✅ Dataset configuration found: `{data_yaml_path.name}`")
                    st.session_state.training_workspace = str(workspace)
                    with st.spinner("🏋️ Training in progress. This may take several minutes..."):
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
                    save_dir = Path(getattr(results, "save_dir", workspace / "runs" / "custom_train"))
                    best_weights = save_dir / "weights" / "best.pt"
                    if best_weights.is_file():
                        st.session_state.trained_model_path = str(best_weights)
                        st.session_state.training_logs = f"✅ Training completed!\nBest weights: {best_weights}"
                        st.success("🎉 Training completed successfully. Your model is ready!")
                    else:
                        st.session_state.training_logs = (
                            "⚠️ Training finished, but best.pt was not found. Check dataset labels and training logs."
                        )
                        st.warning(st.session_state.training_logs)
            except zipfile.BadZipFile:
                st.error("❌ This is not a valid ZIP archive.")
                st.session_state.training_logs = "Training failed: invalid ZIP archive."
            except ValueError as exc:
                st.error(f"🛡️ Dataset archive validation failed: {exc}")
                st.session_state.training_logs = f"Training failed: {exc}"
            except Exception as exc:
                st.error(f"❌ Training failed: {exc}")
                st.session_state.training_logs = f"Training failed: {exc}"

    st.markdown("### 🧾 Training Logs")
    st.code(st.session_state.training_logs, language="text")

    weights_path = st.session_state.trained_model_path
    if weights_path and Path(weights_path).is_file():
        st.success("📦 Custom model weights are ready to download.")
        try:
            with open(weights_path, "rb") as weights_file:
                st.download_button(
                    "📥 Download best.pt",
                    data=weights_file.read(),
                    file_name="custom_yolo_best.pt",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        except OSError as exc:
            st.warning(f"Could not read weights file: {exc}")

# ============================================================
# TAB 3: DOCUMENTATION
# ============================================================
with tab_docs:
    st.header("📚 Quick Start Guide")
    st.markdown(
        """ ### 🎥 Live detection 1. Open **Live Detection**. 2. Select a pretrained model or your custom trained model. 3. Set the confidence threshold and optionally choose target classes. 4. Press **START** in the video widget and allow camera access. 5. Press **STOP** when you finish. ### ⚙️ Custom training 1. Prepare a YOLO-format dataset with images and labels. 2. Include a `data.yaml` or `data.yml` file. 3. ZIP the dataset folder and upload it in **Custom Model Training**. 4. Choose the base model and training settings. 5. Click **Start Training** and wait for completion. 6. Download `best.pt` and use it for deployment. ### 📁 Typical dataset layout ```text dataset/ ├── data.yaml ├── images/ │ ├── train/ │ └── val/ └── labels/ ├── train/ └── val/ ``` ### 🧰 Dependencies - `streamlit` - `streamlit-webrtc` - `ultralytics` - `opencv-python-headless` - `numpy` - `av` ### 🛡️ Deployment notes - Browser webcam access requires HTTPS in hosted deployments and permission from the visitor. - Training may be too resource-intensive for some free hosting tiers. - Uploaded datasets and trained weights use temporary server storage and may disappear after a restart. - Do not upload sensitive or confidential datasets to a public app. """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9db4d7;padding:10px'>"
    "🤖 YOLO AI Vision Suite • Built for curious minds • Detect responsibly ✨"
    "</div>",
    unsafe_allow_html=True,
    ))

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
