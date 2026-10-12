"""GateKeeper AI — upload a video, draw a restricted zone, and analyze it."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import av
from streamlit_webrtc import webrtc_streamer

from vision_utils import (
    encode_face,
    load_face_model,
    load_live_face_model,
    load_live_yolo_model,
    load_yolo_model,
    trigger_intruder_alert,
)

st.set_page_config(
    page_title="GateKeeper AI | Video Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual styling ----------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%); }
      .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1450px; }
      .hero { padding: 1.35rem 1.5rem; border-radius: 18px; color: #f8fafc;
              background: linear-gradient(115deg, #101e36 0%, #1d4168 58%, #187c89 100%);
              margin-bottom: 1.2rem; box-shadow: 0 10px 28px rgba(15, 35, 60, .12); }
      .hero h1 { margin: 0; font-size: 2rem; color: #fff; }
      .hero p { margin: .45rem 0 0 0; color: #dbeafe; font-size: 1rem; }
      div[data-testid="stMetric"] { background: #fff; border: 1px solid #e2e8f0;
           padding: .8rem 1rem; border-radius: 14px; box-shadow: 0 3px 12px rgba(15, 35, 60, .04); }
      div[data-testid="stTabs"] button { font-weight: 650; }
      .small-note { color: #64748b; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛡️ GateKeeper AI</h1>
      <p>Video-based restricted-zone monitoring · person detection · optional face matching</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Prototype for reviewing uploaded recordings. Detection and face matching can be wrong; "
    "do not use this app as the sole basis for security or employment decisions."
)

# ---------- Session state ----------
_STATE_DEFAULTS = {
    "known_encodings": [],
    "known_names": [],
    "restricted_zone_norm": None,
    "zone_video_signature": None,
    "zone_canvas_reset": 0,
    "source_video_signature": None,
    "source_video_path": None,
    "source_video_metadata": None,
    "analysis_output_path": None,
    "analysis_summary": None,
    # Email alert inputs are session-only; they are not written to disk.
    "email_sender": "",
    "email_app_password": "",
    "email_receiver": "",
    "email_alerts_enabled": False,
    "email_settings_saved": False,
}
for _key, _default in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ---------- Utility functions ----------
def _read_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(
                "OpenCV could not open this video. Try an MP4 file encoded with H.264 or MPEG-4."
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "fps": fps,
            "frame_count": max(frame_count, 0),
            "width": width,
            "height": height,
            "duration": frame_count / fps if frame_count > 0 else 0.0,
        }
    finally:
        cap.release()


def _fit_preview(frame: np.ndarray, max_width: int = 900, max_height: int = 560) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return frame.copy()
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _persist_uploaded_video(uploaded_video) -> str:
    """Write a Streamlit upload to a temporary file for OpenCV to read."""
    signature = "{}|{}|{}".format(
        uploaded_video.name,
        uploaded_video.size,
        getattr(uploaded_video, "file_id", ""),
    )
    existing_signature = st.session_state.source_video_signature
    if (
        signature == existing_signature
        and st.session_state.source_video_path
        and Path(st.session_state.source_video_path).exists()
    ):
        return st.session_state.source_video_path

    old_path = st.session_state.source_video_path
    if old_path and Path(old_path).exists():
        try:
            Path(old_path).unlink()
        except OSError:
            pass

    extension = Path(uploaded_video.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temporary_file:
        temporary_file.write(uploaded_video.getbuffer())
        new_path = temporary_file.name

    # A new source video invalidates its old zone and any old result.
    old_output = st.session_state.analysis_output_path
    if old_output and Path(old_output).exists():
        try:
            Path(old_output).unlink()
        except OSError:
            pass

    st.session_state.source_video_signature = signature
    st.session_state.source_video_path = new_path
    st.session_state.source_video_metadata = None
    st.session_state.restricted_zone_norm = None
    st.session_state.zone_video_signature = None
    st.session_state.analysis_output_path = None
    st.session_state.analysis_summary = None

    return new_path


def _looks_like_email(address: str) -> bool:
    """Basic format check; SMTP remains the final validation."""
    address = address.strip()
    if " " in address or address.count("@") != 1:
        return False
    local, domain = address.rsplit("@", 1)
    return bool(local and "." in domain and not domain.startswith(".") and not domain.endswith("."))


def _get_email_settings():
    """Return the current user's explicitly saved, session-only email settings."""
    if not st.session_state.get("email_alerts_enabled", False):
        return None
    if not st.session_state.get("email_settings_saved", False):
        return None

    sender = str(st.session_state.get("email_sender", "")).strip()
    # Google displays app passwords in groups; remove spaces before SMTP login.
    app_password = "".join(str(st.session_state.get("email_app_password", "")).split())
    receiver = str(st.session_state.get("email_receiver", "")).strip()
    if sender and app_password and receiver:
        return sender, app_password, receiver
    return None


def _match_face(
    face_crop: np.ndarray,
    detector,
    recognizer,
    known_names=None,
    known_encodings=None,
):
    """Compare the largest detected face with enrolled features.

    Explicit lists make this helper safe to use from a WebRTC worker thread.
    """
    if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        return None, 0.0
    crop_h, crop_w = face_crop.shape[:2]
    detector.setInputSize((crop_w, crop_h))
    _, faces = detector.detect(face_crop)
    if faces is None or len(faces) == 0:
        return None, 0.0
    face = max(faces, key=lambda item: float(item[2] * item[3]))
    aligned_face = recognizer.alignCrop(face_crop, face)
    feature = recognizer.feature(aligned_face)
    if feature is None:
        return None, 0.0
    names = known_names if known_names is not None else st.session_state.known_names
    encodings = known_encodings if known_encodings is not None else st.session_state.known_encodings
    best_name, best_score = None, 0.0
    for name, known_feature in zip(names, encodings):
        score = float(recognizer.match(known_feature, feature, cv2.FaceRecognizerSF_FR_COSINE))
        if score >= 0.363 and score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


class PhoneLiveState:
    """Mutable state shared safely between Streamlit UI and WebRTC worker thread."""
    def __init__(self):
        self.lock = threading.RLock()
        self.latest_frame = None
        self.zone_reference_frame = None  # frozen frame used while drawing the live zone
        self.zone_norm = None
        self.canvas_json = None
        self.model = None
        self.known_names = []
        self.known_encodings = []
        self.face_detector = None
        self.face_recognizer = None
        self.email_settings = None
        self.confidence = 0.45
        self.image_size = 320
        self.inside_by_track = {}
        self.alerts_queued = 0
        self.frame_count = 0
        self.last_error = None


def _jpeg_bytes(frame: np.ndarray) -> bytes | None:
    """Encode a BGR frame as a JPEG attachment, or return None on failure."""
    try:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return encoded.tobytes() if ok else None
    except Exception:
        return None


def _intruder_snapshot_bytes(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding_fraction: float = 0.16,
) -> bytes | None:
    """Encode a padded crop of a detected person for an email attachment."""
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = int(box_width * padding_fraction)
    pad_y = int(box_height * padding_fraction)
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0:
        return None
    cv2.rectangle(
        crop,
        (1, 1),
        (max(1, crop.shape[1] - 2), max(1, crop.shape[0] - 2)),
        (0, 0, 230),
        3,
        cv2.LINE_AA,
    )
    return _jpeg_bytes(crop)


def _phone_video_frame_callback(video_frame: av.VideoFrame, live_state: PhoneLiveState) -> av.VideoFrame:
    """Process phone-camera frames in the WebRTC worker (no Streamlit UI calls here)."""
    frame = video_frame.to_ndarray(format="bgr24")
    frame = _fit_preview(frame, max_width=960, max_height=720)
    # Keep a clean frame for person-specific email attachments before overlays.
    snapshot_source = frame.copy()

    with live_state.lock:
        # Keep an unannotated reference frame for drawing the zone in the UI.
        live_state.latest_frame = frame.copy()
        model = live_state.model
        zone_norm = None if live_state.zone_norm is None else live_state.zone_norm.copy()
        names = list(live_state.known_names)
        encodings = list(live_state.known_encodings)
        face_detector = live_state.face_detector
        face_recognizer = live_state.face_recognizer
        email_settings = live_state.email_settings
        confidence = live_state.confidence
        image_size = live_state.image_size

    if model is None:
        cv2.putText(frame, "Enable the live monitor to load the detector", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

    polygon = None
    if zone_norm is not None:
        h, w = frame.shape[:2]
        polygon = np.asarray([
            [int(np.clip(p[0] * w, 0, w - 1)), int(np.clip(p[1] * h, 0, h - 1))]
            for p in zone_norm
        ], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (0, 155, 230))
        frame = cv2.addWeighted(overlay, 0.13, frame, 0.87, 0)
        cv2.polylines(frame, [polygon], True, (0, 220, 255), 3, cv2.LINE_AA)
        _draw_label(frame, "RESTRICTED ZONE", int(polygon[:, 0].min()),
                    max(28, int(polygon[:, 1].min())), (15, 113, 160))
    else:
        cv2.putText(frame, "Draw and save a restricted zone below", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 170, 255), 2, cv2.LINE_AA)

    new_entry_candidates = []
    try:
        predictions = model.track(
            source=frame,
            imgsz=image_size,
            conf=confidence,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = predictions[0] if predictions else None
        if result is not None and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                track_id = None
                try:
                    if box.id is not None:
                        track_id = int(box.id[0])
                except Exception:
                    track_id = None

                foot = (int((x1 + x2) / 2), y2)
                is_inside = bool(polygon is not None and cv2.pointPolygonTest(polygon, foot, False) >= 0)
                label = f"Person #{track_id}" if track_id is not None else "Person"
                color = (220, 145, 35)
                just_entered = False

                if polygon is not None and track_id is not None:
                    with live_state.lock:
                        was_inside = live_state.inside_by_track.get(track_id, False)
                        just_entered = is_inside and not was_inside
                        # Update state on every observed frame. Exiting the polygon resets the entry.
                        live_state.inside_by_track[track_id] = is_inside

                if is_inside:
                    matched_name = None
                    if face_detector is not None and face_recognizer is not None:
                        head_bottom = min(y2, y1 + max(1, int((y2 - y1) * 0.48)))
                        face_crop = frame[y1:head_bottom, x1:x2]
                        matched_name, _ = _match_face(
                            face_crop, face_detector, face_recognizer, names, encodings
                        )
                    if matched_name:
                        label = f"AUTHORIZED: {matched_name[:22]}"
                        color = (44, 155, 82)
                    else:
                        label = "UNRECOGNIZED IN ZONE"
                        color = (45, 45, 225)
                        if just_entered and email_settings:
                            person_image = _intruder_snapshot_bytes(
                                snapshot_source, (x1, y1, x2, y2)
                            )
                            new_entry_candidates.append((track_id, person_image))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.circle(frame, foot, 5, color, -1, cv2.LINE_AA)
                _draw_label(frame, label, x1, max(25, y1 - 3), color)

        # Each entry receives a separate email with that person's cropped snapshot.
        # Per-track inside state prevents repeated alerts while they remain in the zone.
        if new_entry_candidates and email_settings:
            for track_id, person_image in new_entry_candidates:
                queued = trigger_intruder_alert(
                    *email_settings,
                    image_bytes=person_image,
                    event_details=f"Live phone camera event; tracker ID {track_id}; "
                                  f"server time {time.strftime('%Y-%m-%d %H:%M:%S')}",
                )
                if queued:
                    with live_state.lock:
                        live_state.alerts_queued += 1
    except Exception as exc:
        with live_state.lock:
            live_state.last_error = str(exc)
        cv2.putText(frame, "Live detector error - check app logs", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    with live_state.lock:
        live_state.frame_count += 1
    return av.VideoFrame.from_ndarray(frame, format="bgr24")


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - text_w - 12))
    y = max(text_h + 12, y)
    cv2.rectangle(
        frame,
        (x, y - text_h - 10),
        (x + text_w + 10, y + baseline - 4),
        color,
        -1,
    )
    cv2.putText(
        frame, text, (x + 5, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def _enhance_frame(frame: np.ndarray, clahe) -> np.ndarray:
    """Improve local luminance contrast while preserving the frame's color channels."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def _analyze_video(
    video_path: str,
    zone_norm: np.ndarray,
    confidence: float,
    image_size: int,
    frame_stride: int,
    max_seconds: int,
    enhance_low_light: bool,
    progress_bar,
    status_box,
) -> tuple[str, dict]:
    """Analyze sampled video frames and write a visually annotated MP4 result."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open the uploaded video. Try a different MP4 file.")

    writer = None
    output_path = None
    try:
        metadata = _read_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["frame_count"]
        if metadata["width"] <= 0 or metadata["height"] <= 0:
            raise ValueError("The uploaded file does not appear to contain valid video frames.")

        frames_limit = int(max_seconds * fps)
        if total_frames > 0:
            frames_limit = min(frames_limit, total_frames)
        if frames_limit <= 0:
            raise ValueError("No readable video frames were found.")
        with tempfile.NamedTemporaryFile(delete=False, suffix="_gatekeeper_annotated.mp4") as out:
            output_path = out.name

        model = load_yolo_model()
        detector = recognizer = None
        face_model_error = None
        if st.session_state.known_encodings:
            try:
                detector, recognizer = load_face_model()
            except Exception as exc:
                face_model_error = str(exc)

        output_fps = max(1.0, fps / frame_stride)
        max_side = 960
        # Keep the source aspect ratio while limiting output resolution for Cloud CPU/RAM.
        resize_scale = min(max_side / metadata["width"], max_side / metadata["height"], 1.0)
        out_width = max(1, int(round(metadata["width"] * resize_scale)))
        out_height = max(1, int(round(metadata["height"] * resize_scale)))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (out_width, out_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not initialize the MP4 writer on this server.")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance_low_light else None
        email_settings = _get_email_settings()
        email_attempted = False
        email_triggered = False
        email_event_count = 0
        # Per-tracker occupancy state: one alert on entry, reset only after a
        # tracked person is observed outside the polygon.
        inside_by_track: dict[int, bool] = {}
        processed_frames = 0
        people_detected = 0
        flagged_frames = 0
        flagged_people = 0
        frame_index = 0
        last_preview_frame = None
        preview_placeholder = st.empty()

        while frame_index < frames_limit:
            ok, raw_frame = cap.read()
            if not ok:
                break

            current_index = frame_index
            frame_index += 1
            if current_index % frame_stride != 0:
                continue

            height, width = raw_frame.shape[:2]
            frame = raw_frame
            if resize_scale != 1.0:
                frame = cv2.resize(
                    frame, (out_width, out_height), interpolation=cv2.INTER_AREA
                )
            if clahe is not None:
                frame = _enhance_frame(frame, clahe)

            h, w = frame.shape[:2]
            polygon = np.asarray(
                [
                    [int(np.clip(point[0] * w, 0, w - 1)),
                     int(np.clip(point[1] * h, 0, h - 1))]
                    for point in zone_norm
                ],
                dtype=np.int32,
            )
            # Keep the clean processed frame for per-person email attachments.
            snapshot_source = frame.copy()

            # Light translucent fill keeps the perimeter visible without hiding people.
            zone_overlay = frame.copy()
            cv2.fillPoly(zone_overlay, [polygon], (0, 155, 230))
            frame = cv2.addWeighted(zone_overlay, 0.13, frame, 0.87, 0)
            cv2.polylines(frame, [polygon], True, (0, 220, 255), 3, cv2.LINE_AA)
            _draw_label(frame, "RESTRICTED ZONE", int(polygon[:, 0].min()), max(28, int(polygon[:, 1].min())), (15, 113, 160))

            predictions = model.track(
                source=frame,
                imgsz=image_size,
                conf=confidence,
                classes=[0],
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            result = predictions[0] if predictions else None
            frame_people = 0
            frame_flagged = 0

            entry_candidates = []
            if result is not None and result.boxes is not None:
                for person_box in result.boxes:
                    x1, y1, x2, y2 = map(int, person_box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    people_detected += 1
                    foot_point = (int((x1 + x2) / 2), y2)
                    is_inside = cv2.pointPolygonTest(polygon, foot_point, False) >= 0
                    track_id = None
                    try:
                        if person_box.id is not None:
                            track_id = int(person_box.id[0])
                    except Exception:
                        track_id = None
                    was_inside = inside_by_track.get(track_id, False) if track_id is not None else False
                    just_entered = is_inside and not was_inside
                    if track_id is not None:
                        # An observed outside frame resets the state, allowing a new
                        # email if this same tracked person enters the zone again.
                        inside_by_track[track_id] = bool(is_inside)

                    label = f"Person #{track_id}" if track_id is not None else "Person"
                    box_color = (220, 145, 35)  # BGR blue/orange outside the zone.

                    if is_inside:
                        frame_people += 1
                        matched_name = None
                        if detector is not None and recognizer is not None:
                            head_bottom = min(y2, y1 + max(1, int((y2 - y1) * 0.48)))
                            face_crop = frame[y1:head_bottom, x1:x2]
                            matched_name, _ = _match_face(face_crop, detector, recognizer)

                        if matched_name:
                            label = f"AUTHORIZED · {matched_name[:28]}"
                            box_color = (44, 155, 82)
                        else:
                            label = "UNRECOGNIZED IN ZONE"
                            box_color = (45, 45, 225)
                            frame_flagged += 1
                            flagged_people += 1
                            if just_entered and track_id is not None and email_settings:
                                email_attempted = True
                                person_image = _intruder_snapshot_bytes(
                                    snapshot_source, (x1, y1, x2, y2)
                                )
                                entry_candidates.append((track_id, person_image))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, foot_point, 5, box_color, -1, cv2.LINE_AA)
                    _draw_label(frame, label, x1, max(25, y1 - 3), box_color)

            if entry_candidates and email_settings:
                for track_id, person_image in entry_candidates:
                    queued = trigger_intruder_alert(
                        *email_settings,
                        image_bytes=person_image,
                        event_details=(
                            f"Uploaded video frame {current_index + 1}; tracker ID {track_id}; "
                            f"server time {time.strftime('%Y-%m-%d %H:%M:%S')}"
                        ),
                    )
                    if queued:
                        email_triggered = True
                        email_event_count += 1

            if frame_flagged:
                flagged_frames += 1

            # A compact information strip improves readability in the result.
            cv2.rectangle(frame, (0, 0), (w, 40), (18, 30, 48), -1)
            cv2.putText(
                frame,
                f"GATEKEEPER AI  |  FRAME {current_index + 1}  |  PEOPLE IN ZONE {frame_people}  |  FLAGS {frame_flagged}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 248, 252), 1, cv2.LINE_AA,
            )

            writer.write(frame)
            processed_frames += 1

            if processed_frames == 1 or processed_frames % max(5, int(fps / frame_stride)) == 0:
                last_preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(
                    last_preview_frame,
                    caption="Analysis preview (updates while processing)",
                    use_container_width=True,
                )
                fraction = min(frame_index / frames_limit, 1.0)
                progress_bar.progress(fraction, text=f"Analyzed {processed_frames:,} frames…")
                status_box.caption(
                    f"Source frame {frame_index:,} of up to {frames_limit:,} · "
                    f"Detected {people_detected:,} person instances so far"
                )

        if processed_frames == 0:
            raise ValueError("No frames could be processed from this video.")

        progress_bar.progress(1.0, text="Video analysis complete")
        summary = {
            "source_fps": fps,
            "processed_frames": processed_frames,
            "people_detected": people_detected,
            "flagged_frames": flagged_frames,
            "flagged_people": flagged_people,
            "processed_seconds": min(max_seconds, processed_frames * frame_stride / fps),
            "frame_stride": frame_stride,
            "face_model_error": face_model_error,
            "email_configured": bool(email_settings),
            "email_attempted": email_attempted,
            "email_triggered": email_triggered,
            "email_event_count": email_event_count,
            "enhanced": enhance_low_light,
        }
        return output_path, summary
    except Exception:
        if output_path and Path(output_path).exists():
            try:
                Path(output_path).unlink()
            except OSError:
                pass
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


# ---------- Optional email alerts ----------
with st.sidebar:
    st.header("✉️ Email alerts")
    st.caption(
        "Uses Gmail SMTP. Enter a Gmail sender address and a Google App Password — "
        "not your normal Google account password. These values stay in this app session."
    )
    with st.form("email_alert_settings_form", clear_on_submit=False):
        st.text_input(
            "Sender Gmail address",
            key="email_sender",
            placeholder="you@gmail.com",
        )
        st.text_input(
            "Google App Password",
            key="email_app_password",
            type="password",
            placeholder="Enter your 16-character app password",
            help="For Google accounts with 2-Step Verification enabled.",
        )
        st.text_input(
            "Recipient email address",
            key="email_receiver",
            placeholder="recipient@example.com",
        )
        st.checkbox(
            "Enable email alerts during monitoring",
            key="email_alerts_enabled",
            help="One email with a cropped person image is queued when a tracked, unrecognized person enters the saved zone. Another email is sent only after the tracker observes that ID outside and then back inside.",
        )
        save_email_settings = st.form_submit_button(
            "Save email settings", use_container_width=True
        )

    if save_email_settings:
        sender = str(st.session_state.get("email_sender", "")).strip()
        password = "".join(str(st.session_state.get("email_app_password", "")).split())
        receiver = str(st.session_state.get("email_receiver", "")).strip()
        if not st.session_state.get("email_alerts_enabled", False):
            st.session_state.email_settings_saved = False
            st.info("Email alerts are disabled.")
        elif not sender or not password or not receiver:
            st.session_state.email_settings_saved = False
            st.warning("Fill in all three email fields before saving.")
        elif not _looks_like_email(sender) or not _looks_like_email(receiver):
            st.session_state.email_settings_saved = False
            st.warning("Enter valid sender and recipient email addresses.")
        else:
            st.session_state.email_settings_saved = True
            st.success("Email settings saved for this session.")

    if st.session_state.get("email_settings_saved", False) and st.session_state.get("email_alerts_enabled", False):
        st.caption("Email alerts are enabled for this session.")
    else:
        st.caption("Email alerts are currently off.")

# ---------- Source video upload ----------
with st.container(border=True):
    st.subheader("1 · Upload a recording")
    uploaded_video = st.file_uploader(
        "Choose a security video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="MP4 is recommended. Streamlit Community Cloud defaults to a 200 MB upload limit.",
        key="source_video_uploader",
    )
    st.markdown(
        '<p class="small-note">The video is processed by the app server. It is not a live webcam feed.</p>',
        unsafe_allow_html=True,
    )

video_path = None
video_metadata = None
if uploaded_video is not None:
    if uploaded_video.size > 200 * 1024 * 1024:
        st.error("This upload exceeds the default 200 MB limit. Use a shorter/compressed video.")
    else:
        try:
            video_path = _persist_uploaded_video(uploaded_video)
            if st.session_state.source_video_metadata is None:
                st.session_state.source_video_metadata = _read_video_metadata(video_path)
            video_metadata = st.session_state.source_video_metadata
            st.success(f"Loaded **{uploaded_video.name}** successfully.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Resolution", f"{video_metadata['width']} × {video_metadata['height']}" if video_metadata["width"] else "Unknown")
            m2.metric("Frame rate", f"{video_metadata['fps']:.1f} FPS")
            m3.metric("Duration", f"{video_metadata['duration']:.1f} s" if video_metadata["duration"] else "Unknown")
            m4.metric("File size", f"{uploaded_video.size / (1024 * 1024):.1f} MB")
        except Exception as exc:
            st.error(f"Unable to read this video: {exc}")
            video_path = None
else:
    st.info("Upload a video above to enable zone setup and analysis.")

# ---------- Main workflow ----------
tab_enroll, tab_zone, tab_analyze, tab_live = st.tabs(
    ["👤 Face enrollment", "📐 Restricted zone", "🎞️ Analyze video", "📱 Phone live"]
)

with tab_enroll:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Register authorized personnel")
        st.write("Upload a clear image containing one person's face. The largest detected face is enrolled.")
        employee_name = st.text_input("Person's name", placeholder="e.g. Alex", key="employee_name")
        face_image = st.file_uploader(
            "Face photo", type=["jpg", "jpeg", "png"], key="face_enrollment_image"
        )
        if face_image is not None:
            try:
                face_image.seek(0)
                st.image(Image.open(face_image), caption="Selected face photo", width=240)
                face_image.seek(0)
            except Exception:
                st.error("Could not preview this image. Try a JPG or PNG file.")

        if st.button("＋ Register face", type="primary", use_container_width=True):
            if not employee_name.strip():
                st.warning("Enter a name before registering a face.")
            elif face_image is None:
                st.warning("Upload a face image first.")
            else:
                try:
                    with st.spinner("Loading face models and encoding the image…"):
                        face_detector, face_recognizer = load_face_model()
                        feature = encode_face(face_image, face_detector, face_recognizer)
                    if feature is not None:
                        st.session_state.known_encodings.append(feature)
                        st.session_state.known_names.append(employee_name.strip())
                        st.success(f"Registered **{employee_name.strip()}**.")
                    else:
                        st.error("No face was detected. Try a clearer, front-facing photo.")
                except Exception as exc:
                    st.error(f"Face enrollment failed: {exc}")

    with right:
        st.subheader("Authorized list")
        if st.session_state.known_names:
            for idx, name in enumerate(st.session_state.known_names, start=1):
                row1, row2 = st.columns([0.8, 0.2])
                row1.write(f"**{idx}.** {name}")
                if row2.button("Remove", key=f"remove_face_{idx}"):
                    st.session_state.known_names.pop(idx - 1)
                    st.session_state.known_encodings.pop(idx - 1)
                    st.rerun()
            st.caption("This list is held in the current app session; save or re-enroll it after a reset.")
        else:
            st.info("No faces registered. People entering the zone will be flagged as unrecognized.")

with tab_zone:
    st.subheader("Draw the restricted area")
    st.write("Choose a frame from the uploaded video, draw one rectangle around the restricted area, then save it.")

    if video_path is None or video_metadata is None:
        st.warning("Upload a readable video before setting a zone.")
    elif video_metadata["frame_count"] <= 0:
        st.warning("The video does not report a frame count, so zone frame selection is unavailable.")
    else:
        preview_cap = cv2.VideoCapture(video_path)
        try:
            max_frame = max(0, video_metadata["frame_count"] - 1)
            zone_frame_index = st.slider(
                "Frame used for drawing",
                min_value=0,
                max_value=max_frame,
                value=min(max_frame, max(0, int(video_metadata["fps"] * 1))),
                help="Pick a frame where the scene and restricted area are easy to see.",
                key=f"zone_frame_{st.session_state.source_video_signature}",
            )
            preview_cap.set(cv2.CAP_PROP_POS_FRAMES, zone_frame_index)
            ok, preview_frame = preview_cap.read()
        finally:
            preview_cap.release()

        if not ok:
            st.error("Could not extract this frame. Try another frame or another video.")
        else:
            preview_frame = _fit_preview(preview_frame)
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            canvas_bg = Image.fromarray(preview_rgb)
            st.markdown("**Draw a rectangle on the frame below**")
            canvas_key = (
                f"zone_canvas_{st.session_state.source_video_signature}_"
                f"{zone_frame_index}_{st.session_state.zone_canvas_reset}"
            )
            canvas_state_key = f"_{canvas_key}_json"
            canvas_result = st_canvas(
                fill_color="rgba(255, 80, 80, 0.20)",
                stroke_width=3,
                stroke_color="#ff4b4b",
                background_image=canvas_bg,
                background_color="#ffffff",
                drawing_mode="rect",
                height=preview_frame.shape[0],
                width=preview_frame.shape[1],
                key=canvas_key,
                # Use our own reset control so the most recent rectangle is retained
                # reliably when a regular Streamlit button causes a script rerun.
                update_streamlit=True,
            )

            # st_canvas may briefly return json_data=None during a Streamlit rerun.
            # Keep the last payload containing a rectangle so clicking Save does not
            # discard the drawing just made by the user.
            incoming_json = canvas_result.json_data
            if incoming_json is not None:
                incoming_objects = incoming_json.get("objects", []) or []
                incoming_rectangles = [
-                    obj for obj in incoming_objects if obj.get("type") == "rect"
+                    obj
+                    for obj in incoming_objects
+                    if isinstance(obj, dict)
+                    and str(obj.get("type", "")).strip().lower() in {"rect", "rectangle"}
                 ]
                if incoming_rectangles:
                    st.session_state[canvas_state_key] = incoming_json

            cached_json = st.session_state.get(canvas_state_key, {})
            objects = [obj for obj in (cached_objects or []) if obj.get("type") == "rect"]
+           objects = [
+                      obj
+                      for obj in (cached_objects or [])
+                      if isinstance(obj, dict)
+                      and str(obj.get("type", "")).strip().lower() in {"rect", "rectangle"}
+                     ]

            if objects:
                st.caption(
                    f"Rectangle detected on the canvas ({len(objects)} found). "
                    "Click **Save restricted zone** to use the latest rectangle."
                )
            else:
                st.info("No rectangle detected yet. Drag on the video frame to draw one.")

            action_col1, action_col2, action_col3 = st.columns([0.46, 0.27, 0.27])
            with action_col1:
                save_zone_clicked = st.button(
                    "Save restricted zone", type="primary", use_container_width=True
                )
            with action_col2:
                clear_drawing_clicked = st.button("Clear drawing", use_container_width=True)
            with action_col3:
                clear_zone_clicked = st.button("Clear saved zone", use_container_width=True)

            if clear_drawing_clicked:
                # Change the component key to create a fresh, empty canvas on rerun.
                st.session_state.zone_canvas_reset += 1
                st.session_state.pop(canvas_state_key, None)
                st.rerun()

            if save_zone_clicked:
                if not objects:
                    st.warning(
                        "No rectangle was received from the drawing canvas. Drag a rectangle "
                        "over the frame, wait for the canvas to update, then click Save again."
                    )
                else:
                    rect = objects[-1]
                    x = float(rect.get("left", 0))
                    y = float(rect.get("top", 0))
                    rect_width = float(rect.get("width", 0)) * float(rect.get("scaleX", 1))
                    rect_height = float(rect.get("height", 0)) * float(rect.get("scaleY", 1))
                    # Fabric stores a rectangle's left/top and dimensions in canvas pixels.
                    # Clamp the corners in canvas space before converting to normalized coords.
                    canvas_w, canvas_h = preview_frame.shape[1], preview_frame.shape[0]
                    left = float(np.clip(min(x, x + rect_width), 0, canvas_w - 1))
                    top = float(np.clip(min(y, y + rect_height), 0, canvas_h - 1))
                    right = float(np.clip(max(x, x + rect_width), 0, canvas_w - 1))
                    bottom = float(np.clip(max(y, y + rect_height), 0, canvas_h - 1))
                    x1, y1 = left / canvas_w, top / canvas_h
                    x2, y2 = right / canvas_w, bottom / canvas_h
                    if (x2 - x1) < 0.01 or (y2 - y1) < 0.01:
                        st.error("The rectangle is too small. Draw a larger zone.")
                    else:
                        st.session_state.restricted_zone_norm = np.asarray(
                            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                        )
                        st.session_state.zone_video_signature = st.session_state.source_video_signature
                        st.success("✅ Restricted zone saved for this video. You can now open Analyze video.")

            if clear_zone_clicked:
                st.session_state.restricted_zone_norm = None
                st.session_state.zone_video_signature = None
                st.success("Saved zone cleared.")

            if (
                st.session_state.restricted_zone_norm is not None
                and st.session_state.zone_video_signature == st.session_state.source_video_signature
            ):
                st.caption("✅ A restricted zone is saved for the currently uploaded video.")
                # Show exactly what will be checked during video analysis.
                saved_preview = preview_frame.copy()
                saved_polygon = np.asarray(
                    [
                        [int(point[0] * saved_preview.shape[1]), int(point[1] * saved_preview.shape[0])]
                        for point in st.session_state.restricted_zone_norm
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(saved_preview, [saved_polygon], True, (0, 200, 255), 3, cv2.LINE_AA)
                overlay = saved_preview.copy()
                cv2.fillPoly(overlay, [saved_polygon], (0, 155, 230))
                saved_preview = cv2.addWeighted(overlay, 0.16, saved_preview, 0.84, 0)
                st.image(
                    cv2.cvtColor(saved_preview, cv2.COLOR_BGR2RGB),
                    caption="Saved restricted zone preview",
                    use_container_width=True,
                )

with tab_analyze:
    st.subheader("Analyze and annotate the recording")
    if video_path is None:
        st.warning("Upload a readable video above first.")
    elif (
        st.session_state.restricted_zone_norm is None
        or st.session_state.zone_video_signature != st.session_state.source_video_signature
    ):
        st.warning("Set and save a restricted zone in the **Restricted zone** tab first.")
    else:
        with st.expander("Analysis settings", expanded=True):
            settings_left, settings_right = st.columns(2)
            with settings_left:
                confidence = st.slider("Person detection confidence", 0.20, 0.85, 0.45, 0.05)
                image_size = st.select_slider("Detection image size", options=[320, 480, 640], value=480)
            with settings_right:
                frame_stride = st.select_slider(
                    "Analyze every Nth frame", options=[1, 2, 3, 5], value=2,
                    help="Higher values can speed up analysis but make the output less smooth.",
                )
                max_seconds = st.slider(
                    "Maximum video duration to analyze (seconds)", 15, 300, 60, 15,
                    help="Only the beginning of longer videos is processed. Raise this for longer clips.",
                )
            enhance_low_light = st.checkbox(
                "Enhance contrast for dim/low-contrast scenes", value=False,
                help="Applies CLAHE to luminance before detection. It can help some scenes but may reduce accuracy in others.",
            )

        st.caption(
            "The annotated export is resized to at most 960 pixels on its longest side and does not preserve source audio."
        )
        start_analysis = st.button("▶ Analyze video", type="primary", use_container_width=True)

        if start_analysis:
            progress = st.progress(0, text="Starting video analysis…")
            status = st.empty()
            try:
                with st.spinner("Loading models and processing video on the server…"):
                    output_path, summary = _analyze_video(
                        video_path=video_path,
                        zone_norm=st.session_state.restricted_zone_norm,
                        confidence=confidence,
                        image_size=image_size,
                        frame_stride=frame_stride,
                        max_seconds=max_seconds,
                        enhance_low_light=enhance_low_light,
                        progress_bar=progress,
                        status_box=status,
                    )
                st.session_state.analysis_output_path = output_path
                st.session_state.analysis_summary = summary
                st.success("Analysis completed. Review the result below.")
            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")
                st.info(
                    "For a first deployment, try a short MP4 clip. The first run also needs to download the YOLO model; "
                    "face models are downloaded only when registered faces are used."
                )

        if st.session_state.analysis_output_path and Path(st.session_state.analysis_output_path).exists():
            summary = st.session_state.analysis_summary or {}
            st.divider()
            st.subheader("Analysis results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Frames analyzed", f"{summary.get('processed_frames', 0):,}")
            r2.metric("Person detections", f"{summary.get('people_detected', 0):,}")
            r3.metric("Frames flagged", f"{summary.get('flagged_frames', 0):,}")
            r4.metric("Unrecognized instances", f"{summary.get('flagged_people', 0):,}")

            if summary.get("face_model_error"):
                st.warning(
                    "Face models could not be loaded, so people inside the zone were checked as unrecognized. "
                    f"Details: {summary['face_model_error']}"
                )
            if summary.get("email_configured"):
                if summary.get("email_triggered"):
                    st.info(
                        f"{summary.get('email_event_count', 0)} entry-alert email(s) queued. "
                        "Each tracked, unrecognized entry includes a JPEG snapshot. Check the recipient inbox and spam folder."
                    )
                elif summary.get("email_attempted"):
                    st.warning("An unrecognized entry was detected, but no email could be queued. Check your sender settings and app logs.")
                else:
                    st.caption("Email alerts are enabled, but no unrecognized person was flagged in the processed frames.")
            else:
                st.caption("Email alerts are off. Configure sender, app password, and recipient in the sidebar to enable them.")

            output_path = st.session_state.analysis_output_path
            try:
                with open(output_path, "rb") as video_file:
                    output_bytes = video_file.read()
                st.video(output_bytes, format="video/mp4")
                st.download_button(
                    "⬇ Download annotated video",
                    data=output_bytes,
                    file_name="gatekeeper_annotated.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Could not preview or prepare the annotated video: {exc}")

with tab_live:
    st.subheader("Live monitoring from a phone or computer camera")
    st.write(
        "Open this Streamlit app on your phone, allow camera access, and press **START**. "
        "This sends live frames from the phone browser to the app for analysis; it does not use the server's webcam."
    )
    st.warning(
        "Live WebRTC depends on network conditions. Streamlit Cloud serves HTTPS, but some networks require a TURN server. "
        "If the camera stays disconnected, try another network or configure TURN credentials in Streamlit Secrets."
    )
    enable_phone_live = st.checkbox(
        "Enable phone live monitor", value=False, key="enable_phone_live_monitor",
        help="Loads YOLO and enables the secure browser camera stream when checked."
    )
    if enable_phone_live:
        if "phone_live_state" not in st.session_state:
            st.session_state.phone_live_state = PhoneLiveState()
        phone_state: PhoneLiveState = st.session_state.phone_live_state
        try:
            if phone_state.model is None:
                with st.spinner("Loading the person detector for live monitoring…"):
                    live_model = load_live_yolo_model()
                with phone_state.lock:
                    phone_state.model = live_model
            # Snapshot current settings into the shared live state. The worker does not
            # read st.session_state directly.
            with phone_state.lock:
                phone_state.known_names = list(st.session_state.known_names)
                phone_state.known_encodings = list(st.session_state.known_encodings)
                phone_state.email_settings = _get_email_settings()
                phone_state.confidence = 0.45
                phone_state.image_size = 320
            if phone_state.known_encodings and not getattr(phone_state, "face_model_attempted", False):
                phone_state.face_model_attempted = True
                try:
                    with st.spinner("Loading optional face-recognition models…"):
                        phone_state.face_detector, phone_state.face_recognizer = load_live_face_model()
                except Exception as exc:
                    st.warning(f"Face recognition is unavailable in live mode; all zone entries will be treated as unrecognized. Details: {exc}")

            rtc_config = {
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
            # Optional TURN server values can be added in Streamlit Cloud → Settings → Secrets:
            # TURN_SERVER_URL = "turn:your-turn-host:3478"
            # TURN_SERVER_USERNAME = "your-username"
            # TURN_SERVER_CREDENTIAL = "your-credential"
            try:
                turn_url = str(st.secrets.get("TURN_SERVER_URL", "")).strip()
                turn_username = str(st.secrets.get("TURN_SERVER_USERNAME", "")).strip()
                turn_credential = str(st.secrets.get("TURN_SERVER_CREDENTIAL", "")).strip()
                if turn_url and turn_username and turn_credential:
                    rtc_config["iceServers"].append({
                        "urls": [turn_url],
                        "username": turn_username,
                        "credential": turn_credential,
                    })
            except Exception:
                pass

            webrtc_streamer(
                key="gatekeeper-phone-live-camera",
                video_frame_callback=lambda video_frame: _phone_video_frame_callback(video_frame, phone_state),
                media_stream_constraints={
                    "video": {
                        "facingMode": {"ideal": "environment"},
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                    },
                    "audio": False,
                },
                rtc_configuration=rtc_config,
                async_processing=True,
            )
            st.caption(
                f"Email alerts queued in this live session: {phone_state.alerts_queued}. "
                "One alert is sent when a tracked, unrecognized person enters; the state resets after they are detected outside the zone."
            )
            if phone_state.last_error:
                st.error(f"Live-frame processing reported an error: {phone_state.last_error}")

            refresh_zone_frame = st.button(
                "Capture/freeze a frame for restricted-zone drawing", use_container_width=True,
                key="refresh_phone_zone_frame",
            )
            with phone_state.lock:
                latest = None if phone_state.latest_frame is None else phone_state.latest_frame.copy()
                if refresh_zone_frame and latest is not None:
                    phone_state.zone_reference_frame = latest.copy()
                zone_reference = (
                    None if phone_state.zone_reference_frame is None
                    else phone_state.zone_reference_frame.copy()
                )
            if zone_reference is None:
                st.info("Start the camera above, then click **Capture/freeze a frame** to set up the zone.")
            else:
                preview_frame = _fit_preview(zone_reference)
                preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                canvas_bg = Image.fromarray(preview_rgb)
                st.write("**Draw the restricted rectangle on the latest phone-camera frame**")
                live_canvas = st_canvas(
                    fill_color="rgba(255, 80, 80, 0.20)",
                    stroke_width=3,
                    stroke_color="#ff4b4b",
                    background_image=canvas_bg,
                    background_color="#ffffff",
                    drawing_mode="rect",
                    height=preview_frame.shape[0],
                    width=preview_frame.shape[1],
                    key="gatekeeper_phone_zone_canvas",
                    update_streamlit=True,
                )
                incoming = live_canvas.json_data
                if incoming is not None:
                    incoming_objects = incoming.get("objects", []) or []
                    incoming_rectangles = [
                        obj for obj in incoming_objects
                        if isinstance(obj, dict)
                        and str(obj.get("type", "")).strip().lower() in {"rect", "rectangle"}
                    ]
                    if incoming_rectangles:
                        phone_state.canvas_json = incoming
                canvas_json = phone_state.canvas_json or {}
                objects = [
                    obj for obj in (canvas_json.get("objects", []) or [])
                    if isinstance(obj, dict)
                    and str(obj.get("type", "")).strip().lower() in {"rect", "rectangle"}
                ]
                if objects:
                    st.caption("Rectangle detected. Save it to use the zone for live alerts.")
                else:
                    st.info("No rectangle detected yet. Drag a rectangle on the frame above.")
                save_live_zone, clear_live_zone = st.columns(2)
                with save_live_zone:
                    save_live_zone_clicked = st.button(
                        "Save live restricted zone", type="primary", use_container_width=True,
                        key="save_phone_live_zone",
                    )
                with clear_live_zone:
                    clear_live_zone_clicked = st.button(
                        "Clear live zone", use_container_width=True, key="clear_phone_live_zone",
                    )
                if save_live_zone_clicked:
                    if not objects:
                        st.warning("No rectangle data was received. Draw a rectangle, wait for the canvas to update, and save again.")
                    else:
                        rect = objects[-1]
                        x = float(rect.get("left", 0))
                        y = float(rect.get("top", 0))
                        rw = float(rect.get("width", 0)) * float(rect.get("scaleX", 1))
                        rh = float(rect.get("height", 0)) * float(rect.get("scaleY", 1))
                        canvas_w, canvas_h = preview_frame.shape[1], preview_frame.shape[0]
                        left = float(np.clip(min(x, x + rw), 0, canvas_w - 1))
                        top = float(np.clip(min(y, y + rh), 0, canvas_h - 1))
                        right = float(np.clip(max(x, x + rw), 0, canvas_w - 1))
                        bottom = float(np.clip(max(y, y + rh), 0, canvas_h - 1))
                        if right - left < 5 or bottom - top < 5:
                            st.error("The rectangle is too small. Draw a larger zone.")
                        else:
                            with phone_state.lock:
                                phone_state.zone_norm = np.asarray(
                                    [[left / canvas_w, top / canvas_h],
                                     [right / canvas_w, top / canvas_h],
                                     [right / canvas_w, bottom / canvas_h],
                                     [left / canvas_w, bottom / canvas_h]], dtype=np.float32
                                )
                                phone_state.inside_by_track.clear()
                            st.success("Live restricted zone saved. Watch the live stream for entry detections.")
                if clear_live_zone_clicked:
                    with phone_state.lock:
                        phone_state.zone_norm = None
                        phone_state.inside_by_track.clear()
                    st.success("Live restricted zone cleared.")
                with phone_state.lock:
                    saved_live_zone = None if phone_state.zone_norm is None else phone_state.zone_norm.copy()
                if saved_live_zone is not None:
                    saved_preview = preview_frame.copy()
                    saved_poly = np.asarray([
                        [int(point[0] * saved_preview.shape[1]), int(point[1] * saved_preview.shape[0])]
                        for point in saved_live_zone
                    ], dtype=np.int32)
                    overlay = saved_preview.copy()
                    cv2.fillPoly(overlay, [saved_poly], (0, 155, 230))
                    saved_preview = cv2.addWeighted(overlay, 0.16, saved_preview, 0.84, 0)
                    cv2.polylines(saved_preview, [saved_poly], True, (0, 200, 255), 3, cv2.LINE_AA)
                    st.image(cv2.cvtColor(saved_preview, cv2.COLOR_BGR2RGB), caption="Saved live restricted zone", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not start phone live mode: {exc}")
    else:
        st.info(
            "For a live feed, open the deployed HTTPS app on the phone and enable the monitor above. "
            "You can still use the Upload a recording workflow when live WebRTC is not available on your network."
        )


st.divider()
st.markdown(
    '<p class="small-note">GateKeeper AI is a prototype. Face matching can produce false matches or false alarms. '
    'Keep enrolled images and videos authorized, verify flagged events manually, and only enter email credentials '
    'on a deployment you control.</p>',
    unsafe_allow_html=True,
)
