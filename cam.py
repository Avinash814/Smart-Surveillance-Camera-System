import streamlit as st
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import math
from twilio.rest import Client
import time
import tempfile

# COCO class names
classNames = [
    'person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Initialize Streamlit app
st.title("Smart Surveillance System")
st.sidebar.header("Settings")

# Twilio credentials (Input via Sidebar)
account_sid = st.sidebar.text_input("Account SID", type="password")
auth_token = st.sidebar.text_input("Auth Token", type="password")
to_number = st.sidebar.text_input("Recipient Number")
from_number = st.sidebar.text_input("Sender Number")
time_interval = st.sidebar.slider("Notification Interval (seconds)", 10, 600, 10)

# YOLO Model selection
model_type = st.sidebar.selectbox("Select YOLO Model", ["yolov8n.pt", "yolov8s.pt"])
model = YOLO(model_type)

# Video source configuration
video_source = st.sidebar.selectbox("Video Source", ["Webcam", "Upload"])
if video_source == "Upload":
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mkv"])
    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name
else:
    video_path = 0  # Webcam

# Initialize PoseDetector
detector = PoseDetector(staticMode=False, detectionCon=0.6)

# Surveillance logic
if st.button("Start Surveillance"):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Placeholder for displaying video
    last_notification_time = time.time() - time_interval  # Allow immediate first notification

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video processing complete or no frames available.")
            break

        results = model(frame, stream=True)
        frame = detector.findPose(frame)
        person_count = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                w, h = x2 - x1, y2 - y1
                bbox = (x1, y1, w, h)

                # Confidence and class detection
                confidence = round(float(box.conf[0]), 2)
                cls_id = int(box.cls[0])
                current_class = classNames[cls_id]

                if current_class == "person" and confidence > 0.4:
                    person_count += 1
                    cvzone.cornerRect(frame, bbox, rt=2)
                    cvzone.putTextRect(frame, f"{current_class} {confidence:.2f}", (x1, y1 - 10), scale=1)

        # Notification logic
        current_time = time.time()
        if person_count > 0 and (current_time - last_notification_time) >= time_interval:
            plurality = "are" if person_count > 1 else "is"
            noun = "people" if person_count > 1 else "person"
            message_body = f"There {plurality} {person_count} {noun} detected."

            st.success(message_body)

            # Send SMS if credentials are provided
            if all([account_sid, auth_token, to_number, from_number]):
                try:
                    client = Client(account_sid, auth_token)
                    client.messages.create(to=to_number, from_=from_number, body=message_body)
                    st.info("SMS notification sent successfully!")
                except Exception as e:
                    st.error(f"Failed to send SMS: {e}")

            last_notification_time = current_time

        # Display video frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()
