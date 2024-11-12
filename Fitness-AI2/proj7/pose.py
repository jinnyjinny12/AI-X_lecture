import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# 랜드마크 간 거리 계산 함수
def calculate_distances(landmarks):
    # 특정 관절 간 거리 계산
    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    distances = [
        distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]),  # 어깨 간 거리
        distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),  # 엉덩이 간 거리
        distance(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),  # 왼쪽 팔꿈치-손목 거리
        distance(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])  # 오른쪽 팔꿈치-손목 거리
    ]
    return distances

# k-NN 분류기 초기화
knn = KNeighborsClassifier(n_neighbors=1)

# 훈련 데이터 (예제)
# "up"과 "down" 상태를 랜드마크 데이터로 표현 (임의 데이터)
train_data = [

    ([0.5, 0.6, 0.7, 0.8], "up"),# 랜드마크 간 거리 예시 (정상 상태)
    ([0.7, 0.8, 0.5, 0.6], "down"),# 랜드마크 간 거리 예시 (운동 상태)
    ([0.6, 0.7, 0.8, 0.9], "up"),
    ([0.8, 0.9, 0.6, 0.7], "down")
]
X_train = [item[0] for item in train_data]
y_train = [item[1] for item in train_data]
knn.fit(X_train, y_train)

# 반복 횟수 측정 변수
current_state = None
prev_state = None
count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe Pose 처리
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # 랜드마크 추출
        landmarks = results.pose_landmarks.landmark
        landmarks = [(lm.x, lm.y) for lm in landmarks]  # x, y 좌표만 추출

        # 랜드마크 간 거리 계산
        distances = calculate_distances(landmarks)
        distances = np.array(distances).reshape(1, -1)

        # 분류
        current_state = knn.predict(distances)[0]

        # 반복 횟수 증가
        if current_state == "down" and prev_state == "up":
            count += 1

        prev_state = current_state

        # 결과 표시
        cv2.putText(frame, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"State: {current_state}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 랜드마크 시각화
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Classification', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()