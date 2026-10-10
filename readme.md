# 🚀 YOLO Real-Time AI Detection & Training Suite

A complete, production-ready **Streamlit** web application for real-time computer vision object detection and custom model training powered by **Ultralytics YOLO** and **OpenCV**.

---

## ✨ Features

1. **🔴 Real-Time Webcam Detection**
   - Live video streaming directly in the browser via `streamlit-webrtc`.
   - Real-time bounding boxes, class labels, and confidence score overlays.
   - Dynamic threshold slider to filter weak detections instantly.
   - Support for standard COCO pre-trained models (`yolov8n`, `yolov8s`) as well as custom-trained model weights.
   - COCO class filtering multi-select to isolate specific objects of interest.

2. **⚙️ Custom Model Training Suite**
   - Interactive dataset upload (ZIP archives structured in standard YOLO format: `images/`, `labels/`, and `data.yaml`).
   - Hyperparameter configuration panel (Epochs, Image Size, Batch Size, Initial Learning Rate).
   - Asynchronous background execution so the Streamlit interface remains responsive during training.
   - One-click download for your resulting custom `best.pt` model weights.

3. **📖 System Documentation & Overview**
   - Built-in guidance for structuring custom training datasets and operating the WebRTC stream.

---

## 🛠️ Prerequisites & Installation

Ensure you have **Python 3.8+** installed on your system.

1. Clone or download this repository containing `app.py` and `requirements.txt`.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

Start the Streamlit development server by running:

```bash
streamlit run app.py
```

Open the local address provided in your terminal (typically `http://localhost:8501`) in your web browser.

---

## 📂 Custom Dataset Structure

To train a custom model via the **Custom Model Training** tab, ensure your uploaded ZIP file contains a valid YOLO dataset directory layout:

```text
my_dataset/
│
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── image3.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        └── image3.txt
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
