import mediapipe as mp
import pandas as pd
import numpy as np
import cv2

# utils.py는 편의성을 제공하는 모듈
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def click_button(x1, y1, x2, y2, body_mark):
    if x1 < body_mark[0] < x2 and y1 < body_mark[1] < y2:
        return True
    return False


def detection_body_part(landmarks, body_part_name):
    # NOSE = 0 LEFT_EYE_INNER = 1 LEFT_EYE = 2 LEFT_EYE_OUTER = 3 RIGHT_EYE_INNER = 4 RIGHT_EYE = 5
    # RIGHT_EYE_OUTER = 6 LEFT_EAR = 7 RIGHT_EAR = 8 MOUTH_LEFT = 9 MOUTH_RIGHT = 10 LEFT_SHOULDER = 11
    # RIGHT_SHOULDER = 12 LEFT_ELBOW = 13 RIGHT_ELBOW = 14 LEFT_WRIST = 15 RIGHT_WRIST = 16
    # LEFT_PINKY = 17 RIGHT_PINKY = 18 LEFT_INDEX = 19 RIGHT_INDEX = 20 LEFT_THUMB = 21
    # RIGHT_THUMB = 22 LEFT_HIP = 23 RIGHT_HIP = 24 LEFT_KNEE = 25 RIGHT_KNEE = 26 LEFT_ANKLE = 27 
    # RIGHT_ANKLE = 28 LEFT_HEEL = 29 RIGHT_HEEL = 30 LEFT_FOOT_INDEX = 31 RIGHT_FOOT_INDEX = 32
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

def detection_body_parts(landmarks):
    body_parts = pd.DataFrame(columns=["body_part", "x", "y"])

    for i, lndmrk in enumerate(mp_pose.PoseLandmark):
        lndmrk = str(lndmrk).split(".")[1]
        cord = detection_body_part(landmarks, lndmrk)
        body_parts.loc[i] = lndmrk, cord[0], cord[1]

    return body_parts


def score_table(exercise, frame , counter, status, max_counter):
    cv2.putText(frame, "Activity : " + exercise.replace("-", " "),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "Counter : " + str(counter), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Status : " + str(status), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "max counter : " + str(max_counter), (170, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    return frame
    
