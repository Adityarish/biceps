import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def process_frame(frame):
    """Processes a single frame to detect pose and count reps."""
    
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for left arm
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        # Get coordinates for right arm
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Calculate angles
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Update angles in session state for UI display
        st.session_state.left_angle = left_angle
        st.session_state.right_angle = right_angle
        
        # Repetition counter logic for left arm
        if left_angle > 160:
            st.session_state.left_stage = "down"
        if left_angle < 30 and st.session_state.left_stage == 'down':
            st.session_state.left_stage = "up"
            st.session_state.left_counter += 1

        # Repetition counter logic for right arm
        if right_angle > 160:
            st.session_state.right_stage = "down"
        if right_angle < 30 and st.session_state.right_stage == 'down':
            st.session_state.right_stage = "up"
            st.session_state.right_counter += 1

    except Exception as e:
        # Pass if no landmarks are detected
        pass

    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    return image

# --- STREAMLIT UI ---

st.title("AI Fitness Coach: Bicep Curls 💪")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    
    # Initialize session state variables
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False
    if "left_counter" not in st.session_state:
        st.session_state.left_counter = 0
    if "right_counter" not in st.session_state:
        st.session_state.right_counter = 0
    if "left_stage" not in st.session_state:
        st.session_state.left_stage = "down"
    if "right_stage" not in st.session_state:
        st.session_state.right_stage = "down"
    if "left_angle" not in st.session_state:
        st.session_state.left_angle = 0
    if "right_angle" not in st.session_state:
        st.session_state.right_angle = 0
    if "cap" not in st.session_state:
        st.session_state.cap = None

    # Start/Stop Webcam Button
    if st.button("Start Webcam", type="primary", use_container_width=True):
        st.session_state.run_webcam = True
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

    if st.button("Stop Webcam", use_container_width=True):
        st.session_state.run_webcam = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

    # Reset Counter Button
    if st.button("Reset Counters", use_container_width=True):
        st.session_state.left_counter = 0
        st.session_state.right_counter = 0
        st.session_state.left_stage = "down"
        st.session_state.right_stage = "down"
        st.toast("Counters have been reset!", icon="✅")

    st.info("Instructions: \n1. Click 'Start Webcam'. \n2. Allow camera permissions. \n3. Keep your full body in frame. \n4. Perform bicep curls!")

# --- Main Page Layout ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Live Webcam Feed")
    frame_placeholder = st.empty()
    if not st.session_state.run_webcam:
        frame_placeholder.image("static/placeholder.png", caption="Webcam is off. Click 'Start Webcam' in the sidebar.")

with col2:
    st.subheader("Workout Dashboard")
    
    # Repetition Counters
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Left Arm Curls", value=st.session_state.left_counter)
    with metric_col2:
        st.metric(label="Right Arm Curls", value=st.session_state.right_counter)
    
    st.divider()

    # Stage & Angle Display
    stage_col1, stage_col2 = st.columns(2)
    with stage_col1:
        st.markdown(f"**Left Arm Stage:** `{st.session_state.left_stage.upper()}`")
        st.markdown(f"**Left Arm Angle:** `{int(st.session_state.left_angle)}°`")
    with stage_col2:
        st.markdown(f"**Right Arm Stage:** `{st.session_state.right_stage.upper()}`")
        st.markdown(f"**Right Arm Angle:** `{int(st.session_state.right_angle)}°`")

    st.divider()

    st.success("Keep up the great work! Consistency is key. 🚀")


# --- Webcam Loop ---
if st.session_state.run_webcam and st.session_state.cap is not None:
    while st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Could not read frame from webcam. Please restart.")
            break

        processed_frame = process_frame(frame)
        
        # Display the processed frame
        frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        
        # Stop the loop if the run flag is set to False from the sidebar
        if not st.session_state.run_webcam:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            break
    
    # Final release of resources if loop breaks unexpectedly
    if st.session_state.cap is not None and not st.session_state.run_webcam:
        st.session_state.cap.release()
        st.session_state.cap = None

# To make this fully functional, create a folder named `static`
# in the same directory as your script and place a placeholder image
# named `placeholder.png` inside it.