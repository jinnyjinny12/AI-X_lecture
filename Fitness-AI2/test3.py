import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose 추적 객체 생성
pose = mp_pose.Pose(
    static_image_mode=False,        # 실시간 영상 처리
    model_complexity=2,             # 모델 복잡도 (0, 1, 2 중 선택)
    enable_segmentation=True,       # 배경 분리 활성화
    min_detection_confidence=0.5,   # 감지 임계값
    min_tracking_confidence=0.5     # 추적 임계값
)

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # BGR 이미지를 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose 감지
    results = pose.process(frame_rgb)

    # 랜드마크 시각화
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Segmentation mask 시각화
    if results.segmentation_mask is not None:
        mask = (results.segmentation_mask > 0.5).astype("uint8") * 255
        mask = cv2.merge((mask, mask, mask))
        frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

    # 결과 출력
    cv2.imshow('MediaPipe Pose', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
