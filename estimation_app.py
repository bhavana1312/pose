import streamlit as st
from PIL import Image
import numpy as np
import mediapipe as mp
import cv2

# Mediapipe pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

DEMO_IMAGE = 'stand.jpg'

st.title("Human Pose Estimation with Mediapipe")

st.text("Make sure you have a clear image with all body parts visible.")

# Upload an image
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Process the image for pose estimation
@st.cache_data
def estimate_pose(image):
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_image)

    # Draw the pose annotations on the image
    annotated_image = image.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
    return annotated_image

# Estimate pose
output_image = estimate_pose(image)

st.subheader("Positions Estimated")
st.image(output_image, caption="Pose Estimation", use_column_width=True)

st.markdown("### Pose estimation performed using Mediapipe.")

