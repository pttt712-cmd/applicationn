import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tempfile

st.title("손 동작 인식 – 이미지 또는 영상 업로드")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 이미지 처리 함수
def process_image(image):
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        return image


# 영상 처리 함수
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            stframe.image(frame, channels="BGR")

    cap.release()


# 업로드 선택
upload_type = st.radio("업로드 유형 선택:", ["이미지", "영상"])

if upload_type == "이미지":
    img_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if img_file:
        img = cv2.imdecode(
            np.frombuffer(img_file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        result = process_image(img)
        st.image(result, channels="BGR")

else:
    video_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "mov", "avi"])
    if video_file:
        st.info("영상을 처리하는 중입니다…")
        process_video(video_file)