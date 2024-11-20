import cv2
import numpy as np
import mediapipe as mp
from utils import *
import time

class HandlerTimeManager():
    def __init__(self, duration):
        self.start_time = None
        self.duration = duration

def get_exercise_name(frame, result_landmarks, mp_drawing, pose_name_list, handler_time_manger:HandlerTimeManager):
    
    # *** 차후에 핸드폰 화면에 맞춰야함. ***
    # 사각형 영역 정의 (오른쪽에서 왼쪽으로 나열)
    frame_width = 800  # 캡처할 프레임의 너비 (화면의 크기를 기준으로 조정)
    box_width = 110    # 각 운동을 위한 박스의 너비
    box_height = 80   # 각 운동을 위한 박스의 높이
    padding = 10      # 각 박스 사이의 간격
    # 오른쪽 끝에서부터 왼쪽으로 배치하는 사각형 박스들
    exercise_boxes = {}

    if handler_time_manger.start_time is None:
        handler_time_manger.start_time = time.time()
    
    duration = handler_time_manger.duration

    for i, exercise in enumerate(pose_name_list):
        x1 = frame_width - (i + 1) * (box_width + padding)
        y1 = 100
        x2 = x1 + box_width
        y2 = y1 + box_height
        exercise_boxes[exercise] = (x1, y1, x2, y2)

    # 선택된 운동 이름
    selected_exercise = None  

    # 사각형 영역 및 운동 목록 표시
    for exercise, (x1, y1, x2, y2) in exercise_boxes.items():
        color = (255, 255, 255) if exercise == selected_exercise else (0, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, exercise, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if result_landmarks.pose_landmarks:
        # 랜드마크 좌표 추출
        landmarks = result_landmarks.pose_landmarks.landmark

        left_hand = detection_hand_wrist(landmarks, 'LEFT_WRIST', frame)
        right_hand = detection_hand_wrist(landmarks, 'RIGHT_WRIST', frame)
        
        # 손목 좌표가 사각형 영역 안에 있는지 확인
        for exercise, (x1, y1, x2, y2) in exercise_boxes.items():
            if handler_time_manger.start_time is not None:
                if time.time() - handler_time_manger.start_time >= duration:
                    if click_button(x1, y1, x2, y2, left_hand):
                        handler_time_manger.start_time = None
                        selected_exercise = exercise
                    elif click_button(x1, y1, x2, y2, right_hand):
                        handler_time_manger.start_time = None
                        selected_exercise = exercise
                    

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(
            frame,
            result_landmarks.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        cv2.imshow('Video', frame)
        cv2.waitKey(10)
    
    return selected_exercise

# 종료 핸들러
def exit_handler(frame, result_pose_process, exercise_name, handler_time_manger:HandlerTimeManager):
    box_width = 50    # 각 운동을 위한 박스의 너비
    box_height = 50   # 각 운동을 위한 박스의 높이
    x1 = 400
    y1 = 40
    x2 = x1 + box_width
    y2 = y1 + box_height

    if handler_time_manger.start_time is None:
        handler_time_manger.start_time = time.time()
    
    duration = handler_time_manger.duration

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, 'exit', (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    if result_pose_process.pose_landmarks:
        landmark = result_pose_process.pose_landmarks.landmark

        left_hand = detection_hand_wrist(landmark, 'LEFT_WRIST', frame)
        right_hand = detection_hand_wrist(landmark, 'RIGHT_WRIST', frame)
        if handler_time_manger.start_time is not None:
            if time.time() - handler_time_manger.start_time >= duration:
                if click_button(x1, y1, x2, y2, left_hand): 
                    handler_time_manger.start_time = None        
                    exercise_name = 'exit'
                elif click_button(x1, y1, x2, y2, right_hand):
                    handler_time_manger.start_time = None
                    exercise_name = 'exit'

                
    return exercise_name


# 손목 랜드마크 좌표 가져오기
def detection_hand_wrist(landmarks, wrist:str, frame):
    hand = detection_body_part(landmarks, wrist)

    return (hand[0] * frame.shape[1], hand[1] * frame.shape[0])

