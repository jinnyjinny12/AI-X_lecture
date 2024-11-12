import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 사용자 설정 임계값 (각도)
left_tilt_threshold = 5  # 왼쪽으로 기울어지는 최소 각도
right_tilt_threshold = 5  # 오른쪽으로 기울어지는 최소 각도
shoulder_offset = -0.05  # 어깨 좌표 Y축 보정값 (값을 낮출수록 어깨가 올라감)

# 상태 감지 변수
forward_head_posture = False
left_tilt = False
right_tilt = False

prev_forward_head_posture = False
prev_left_tilt = False
prev_right_tilt = False

turtle_neck_count = 0
left_tilt_count = 0
right_tilt_count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Pose landmarks가 있으면 처리
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 좌표 추출
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + shoulder_offset]
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y + shoulder_offset]
        ear_left = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        ear_right = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

        # 목 기울기 감지 (각도 계산)
        left_tilt_angle = np.degrees(np.arctan2(shoulder_left[1] - ear_left[1],
                                                shoulder_left[0] - ear_left[0]))
        right_tilt_angle = np.degrees(np.arctan2(shoulder_right[1] - ear_right[1],
                                                 ear_right[0] - shoulder_right[0]))

        # 각도 기준으로 상태 감지
        if left_tilt_angle > left_tilt_threshold:  # 왼쪽 기울기 기준 초과
            left_tilt = True
        else:
            left_tilt = False

        if right_tilt_angle > right_tilt_threshold:  # 오른쪽 기울기 기준 초과
            right_tilt = True
        else:
            right_tilt = False

        # 거북목 감지 (귀와 어깨의 X축 비교)
        if ear_left[0] < shoulder_left[0] - 0.05 or ear_right[0] > shoulder_right[0] + 0.05:
            forward_head_posture = True
        else:
            forward_head_posture = False

        # 상태 변화 감지 및 카운트
        if forward_head_posture and not prev_forward_head_posture:
            turtle_neck_count += 1

        if left_tilt and not prev_left_tilt:
            left_tilt_count += 1

        if right_tilt and not prev_right_tilt:
            right_tilt_count += 1

        # 정상 자세 복귀 알림
        if not forward_head_posture and not left_tilt and not right_tilt:
            feedback_text = "Good Posture! Keep it up!"
        else:
            feedback_text = ""

        prev_forward_head_posture = forward_head_posture
        prev_left_tilt = left_tilt
        prev_right_tilt = right_tilt

        # 상태 텍스트 표시
        status_text = f"Turtle Neck: {turtle_neck_count}, Left Tilt: {left_tilt_count}, Right Tilt: {right_tilt_count}"
        tilt_text = f"Left Tilt: {left_tilt_angle:.1f}°, Right Tilt: {right_tilt_angle:.1f}°"
        cv2.putText(image, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, tilt_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, feedback_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # BGR로 다시 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 화면 출력
    cv2.imshow('Posture Detector', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
