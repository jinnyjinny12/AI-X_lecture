import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from exercise_handler import *
from pose_segment import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# 운동 종류 리스트
pose_name_list = ['exit', 'sit-up', 'squat', 'walk', 'push-up', 'pull-up']
cap = cv2.VideoCapture(0)  # webcam
cap.set(3, 800)  # width
cap.set(4, 480)  # height

handler_time_manger = HandlerTimeManager(1.5)
max_counter = 10

#영상 경로 저장
video_paths = {
    "pull-up": "Exercise_videos/pull-up.mp4",
    "push-up": "Exercise_videos/push-up.mp4",
    "sit-up": "Exercise_videos/sit-up.mp4",
    "squat": "Exercise_videos/squat.mp4",
    "walk": "Exercise_videos/walk.mp4",
}

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=True) as pose:
    # 운동 자세 이름을 선택했는지 확인 False = 선택이 안됨. True = 선택 완료.
    is_selected_exercise_name = False 
    # 운동 비디오가 재생됐는지 안됐는지 확인하는 변수 False = 재생이 안됨. True = 재생 됨.
    is_selected_exercise_video = False
    cap_background = None

    counter = 0  # movement of exercise
    status = True  # state of move

    while cap.isOpened():
        ret, frame = cap.read()
        # result_screen = np.zeros((250, 400, 3), np.uint8)
        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
        # recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        # make detection
        results = pose.process(frame)
        # recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not is_selected_exercise_name:
            exercise_name = get_exercise_name(frame, results, mp_drawing, pose_name_list, handler_time_manger)

            if exercise_name == None:
                continue
            elif exercise_name != None:
                is_selected_exercise_name = True

        # 운동 선택 후 운동 영상지정
        current_video = exercise_name if exercise_name != 'exit' else None

        # 사용자가 나가기 선택을 안했으면 선택된 운동에 따라 운동 영상 재생
        if current_video != None and not is_selected_exercise_video:
            is_selected_exercise_video = True
            cap_background = cv2.VideoCapture(video_paths[current_video])
        
        ret_bg, bg_frame = cap_background.read()

         # 리스타트 기능
        if not ret_bg:
            print("Background video ended. Restarting...")
            cap_background.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            ret_bg, bg_frame = cap_background.read()

        # 영상을 웹캠 프레임에 맞춰서 재생
        bg_frame_resized = cv2.resize(bg_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        try:
            landmarks = results.pose_landmarks.landmark
            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                exercise_name, counter, status)
        except:
            pass

        bg_frame_resized = score_table(exercise_name, bg_frame_resized, counter, status, max_counter)

        # render detections (for landmarks)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=4),
            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                   thickness=2,
                                   circle_radius=4),
        )

        exercise_name = exit_handler(bg_frame_resized, results, exercise_name, handler_time_manger)

        #영상 재생과 사용자 객체 분리   
        frame, condition = segment_background(frame, results, bg_frame_resized)

        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif exercise_name == 'exit':
            break
        elif counter == max_counter:
            break

    cap.release()
    cap_background.release()
    cv2.destroyAllWindows()
