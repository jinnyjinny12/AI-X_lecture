import cv2
import numpy as np
from testmodule import PoseProcessor

# Initialize the PoseProcessor
pose_processor = PoseProcessor()

# Paths for the video and background
video_path = "video_for_background/Warrior_flip.mp4"
cap_video = cv2.VideoCapture(0)  # Webcam
cap_video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width
cap_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height
cap_background = cv2.VideoCapture(video_path)

# 웹캠 확인
if not cap_video.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

## 영상 경로 확인
if not cap_background.isOpened():
    print(f"Error: Cannot load background video from {video_path}")
    exit()

while True:
    # Read from webcam
    ret, frame = cap_video.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # Read from background video
    ret_bg, bg_frame = cap_background.read()
    if not ret_bg:
        print("Background video ended. Restarting...")
        cap_background.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
        ret_bg, bg_frame = cap_background.read()

    if not ret_bg or bg_frame is None:
        print("Failed to load background video frame.")
        bg_frame = np.zeros_like(frame)  # Fallback to blank frame

    # 영상을 웹캠 프레임에 맞춰서 재생
    bg_frame_resized = cv2.resize(bg_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Pose detection and background segmentation
    #find_pose 함수의 랜드마크 감지 + 감지 랜드마크 시각화 후 결과 반환
    frame, results = pose_processor.find_pose(frame, draw=True)
    output_frame, condition = pose_processor.segment_background(frame, results, bg_frame_resized)


    # Show the result
    cv2.imshow("Pose Detection with Background", output_frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap_video.release()
cap_background.release()
cv2.destroyAllWindows()
