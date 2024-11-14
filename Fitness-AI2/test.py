import cv2
import numpy as np
import mediapipe as mp
from testmodule import PoseProcessor
from buttonmodule import Button

# Initialize the PoseProcessor
pose_processor = PoseProcessor()

## 버튼 정의
exit_button = Button(50, 50, 150, 50, "Exit", color=(255, 255, 255), active_color=(0, 0, 255))
leg_button = Button(250, 50, 150, 50, "Legs", color=(255, 255, 255), active_color=(0, 255, 0))


#영상 경로 저장
video_paths = {
    "default": "video_for_background/Warrior_flip.mp4",
    "legs": "video_for_background/Legs_flip.mp4"
}
current_video = "default"
cap_video = cv2.VideoCapture(0)  # Webcam
cap_video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width
cap_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height
cap_background = cv2.VideoCapture(video_paths[current_video])


# 웹캠 확인
if not cap_video.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

## 영상 경로 확인
if not cap_background.isOpened():
    print(f"Error: Cannot load background video from {video_paths}")
    exit()

while True:
    # Read from webcam
    ret, frame = cap_video.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # 배경 영상을 읽어옴.
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

    ## 버튼 화면 그리기
    exit_button.draw(output_frame)
    leg_button.draw(output_frame)

    # 버튼이 손으로 오버될시 실행됨.
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        right_index = landmarks[pose_processor.mp_pose.PoseLandmark.RIGHT_INDEX]
        hand_pos = (int(right_index.x * w), int(right_index.y * h))

        # Detect exit action
        if exit_button.detect_action(exit_button.is_hovering(hand_pos)):
            print("Exiting...")
            break

        # Detect legs button action
        if leg_button.detect_action(leg_button.is_hovering(hand_pos)):
            if current_video !="legs":
                print("Switching to legs video...")
                current_video = "legs"
                cap_background.release()
                cap_background = cv2.VideoCapture(video_paths[current_video])
            
    # Show the result
    cv2.imshow("Pose Detection with Background", output_frame)
    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap_video.release()
cap_background.release()
cv2.destroyAllWindows()
