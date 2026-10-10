# GateKeeper AI — uploaded-video version

This version replaces webcam access (`cv2.VideoCapture(0)`) with a workflow designed for Streamlit Community Cloud.

## Features

- Upload a security video (MP4 recommended; other common formats are accepted if the server's OpenCV build can decode them).
- Choose a frame and draw a restricted-zone rectangle.
- Detect people with YOLOv8n and annotate detections and restricted-zone entries.
- Optionally register face photos and compare faces against enrolled session data.
- Optionally apply CLAHE contrast enhancement to dim or low-contrast video.
- Show progress, detection summary, video preview, and an annotated MP4 download.
- Configure a Gmail sender, Google App Password, and recipient directly in the sidebar. The password field is masked; values are used for the current Streamlit session and are not hard-coded into the source.

## Files to place together

- `demo_app.py`
- `vision_utils.py`
- `requirements.txt`

Streamlit Community Cloud also supports a `.streamlit/config.toml` project configuration file. This bundle includes one with the default 200 MB upload limit explicitly set.

## Deploy on Streamlit Community Cloud

1. Commit these files to a GitHub repository.
2. On Streamlit Community Cloud, create an app from that repository and select `demo_app.py` as the entrypoint.
3. Select Python 3.11 in Advanced settings for this pinned dependency set.
4. Deploy and test first with a short MP4 clip. The initial launch downloads the YOLO model. The face detection/recognition ONNX files are downloaded only when face enrollment is used.

Official setup reference: [Streamlit Community Cloud app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Email setup

Open the sidebar's **Email alerts** section, enter the sender Gmail address, Google App Password, and recipient email address, tick **Enable email alerts during video analysis**, and click **Save email settings**.

Gmail SMTP is configured at `smtp.gmail.com:587`, so use a Gmail sender account and a Google App Password rather than your regular Google sign-in password. Credentials entered in the interface are kept only in that app session, not in a source file. **Only enter credentials into a deployment you control.** This form is not intended for collecting credentials from users of a public app.

The original supplied source contained an embedded Gmail app password. Treat that password as exposed: revoke it in the relevant Google account and create a replacement if needed. Never commit a real password or `secrets.toml` file to GitHub.

## Limitations

- Streamlit Community Cloud runs Python remotely, so it cannot use a visitor's local webcam through `VideoCapture(0)`. It can process uploaded recordings.
- The default upload limit is 200 MB. Bigger/longer videos may also exceed CPU, memory, or runtime budgets; start with a short clip.
- Processing uses CPU by default and may be slower than local GPU inference. Increasing the frame stride improves speed but reduces temporal smoothness.
- Annotated output is resized to at most 960 pixels on the longest side and does not preserve source audio.
- Enrolled face encodings live in the current session only; they are not a persistent database.
- Detection/face matching can generate false positives and false negatives. Review alerts manually; do not use this prototype as the sole basis for security or employment decisions.
