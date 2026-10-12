"""Shared model, face-encoding, and optional email-alert helpers for GateKeeper AI."""

from __future__ import annotations

import os
import shutil
import smtplib
import tempfile
import threading
import time
import urllib.request
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / ".model_cache"

# Use the media.githubusercontent.com endpoint because OpenCV Zoo stores these
# ONNX files with Git LFS. The ordinary GitHub raw URL may return only an LFS pointer.
FACE_MODELS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "minimum_bytes": 100_000,
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "minimum_bytes": 10_000_000,
    },
}


def _ensure_model_file(model_key: str) -> Path:
    """Download a required OpenCV Zoo model on demand and cache it locally."""
    info = FACE_MODELS[model_key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / info["filename"]

    if target.exists() and target.stat().st_size >= info["minimum_bytes"]:
        return target

    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(info["url"], timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".download", dir=MODEL_DIR, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                shutil.copyfileobj(response, temp_file)

        if temp_path.stat().st_size < info["minimum_bytes"]:
            raise RuntimeError(
                f"Downloaded {info['filename']} is unexpectedly small. "
                "The model download may have failed."
            )
        os.replace(temp_path, target)
        return target
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the face model {info['filename']}. "
            "Check the app's network access and try again."
        ) from exc


@st.cache_resource(show_spinner=False)
def load_yolo_model() -> YOLO:
    """Load the YOLOv8n model used by uploaded-video analysis."""
    return YOLO("yolov8n.pt")


def load_live_yolo_model() -> YOLO:
    """Create a session-local YOLO instance for phone-camera tracking.

    This intentionally is not cached globally: tracker identities must not be
    shared between different users' live camera sessions.
    """
    return YOLO("yolov8n.pt")


def _build_face_models():
    """Create a detector/recognizer pair using cached-on-disk OpenCV Zoo weights."""
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError(
            "This OpenCV build does not expose FaceDetectorYN/FaceRecognizerSF. "
            "Install opencv-contrib-python-headless from requirements.txt."
        )
    detector_path = _ensure_model_file("yunet")
    recognizer_path = _ensure_model_file("sface")
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), 0.6, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    return detector, recognizer


@st.cache_resource(show_spinner=False)
def load_face_model():
    """Load face models dedicated to uploaded-video analysis."""
    return _build_face_models()


def load_live_face_model():
    """Create a session-local face-model pair for the WebRTC callback."""
    return _build_face_models()


def encode_face(uploaded_file, detector, recognizer):
    """Return a face feature vector from an uploaded image, or None if no face is found."""
    uploaded_file.seek(0)
    image_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None

    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None

    # Prefer the largest detected face when an image contains multiple people.
    face = max(faces, key=lambda item: float(item[2] * item[3]))
    aligned_face = recognizer.alignCrop(image, face)
    return recognizer.feature(aligned_face)


_EMAIL_LOCK = threading.Lock()


def _send_email_worker(
    sender_email: str,
    app_password: str,
    receiver_email: str,
    image_bytes: bytes | None = None,
    event_details: str = "",
) -> None:
    """Send one zone-entry alert with an optional cropped intruder JPEG attachment."""
    message = EmailMessage()
    message["Subject"] = "GateKeeper AI: Restricted-zone entry detected"
    message["From"] = sender_email
    message["To"] = receiver_email
    details = event_details.strip() or "A tracked, unrecognized person entered the restricted zone."
    message.set_content(
        "GateKeeper AI detected an unrecognized person entering the configured restricted zone.\n\n"
        f"Event details: {details}\n\n"
        "A cropped image of the detected person is attached when a usable crop was captured. "
        "Detection may be wrong; review the image manually."
    )
    if image_bytes:
        message.add_attachment(
            image_bytes,
            maintype="image",
            subtype="jpeg",
            filename=f"gatekeeper_intruder_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
        )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=25) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(message)
        print("[GateKeeper AI] Restricted-zone entry alert email sent.")
    except Exception as exc:  # Do not interrupt video processing if SMTP fails.
        print(f"[GateKeeper AI] Could not send alert email: {exc}")


def trigger_intruder_alert(
    sender: str,
    app_password: str,
    receiver: str,
    image_bytes: bytes | None = None,
    event_details: str = "",
) -> bool:
    """Queue one alert for a caller-detected entry event.

    Re-entry de-duplication belongs to the object tracker: it emits an alert only
    when a tracked person changes from outside to inside. There is intentionally no
    global time cooldown, so a valid leave-and-re-enter event can alert immediately.
    """
    if not (sender and app_password and receiver):
        return False
    with _EMAIL_LOCK:
        email_thread = threading.Thread(
            target=_send_email_worker,
            args=(sender, app_password, receiver, image_bytes, event_details),
            daemon=True,
        )
        email_thread.start()
    return True
