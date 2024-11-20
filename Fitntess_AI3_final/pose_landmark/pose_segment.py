##모듈 추가
import cv2
import mediapipe as mp
import numpy as np

# 메인에서 실행하고있는 기능은 빼야함.

#step1 메서드 초기화

def segment_background(frame, results, bg_frame):
    """Segment the background and place a resized user onto it."""
    # Create segmentation mask condition
    if results.segmentation_mask is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 세분화 마스크가 있을 때
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        # 사용자 영역 추출
        user_only = np.where(condition, rgb_frame, 0)

        # 사용자의 영역을 흑백으로 만들어 감지
        gray_user = cv2.cvtColor(user_only, cv2.COLOR_BGR2GRAY)

        # 사용자 영역의 윤곽선 발견
        contours, _ = cv2.findContours(gray_user, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선이 있을 때만 처리
        if contours:
            # 가장 면적이 큰 윤곽선 계산 : 사용자의 영역
            largest_contour = max(contours, key=cv2.contourArea)

            # 사용자 마스크 생성
            mask = np.zeros_like(gray_user)
            cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            # 사용자 모양 유지
            user_masked = cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask)

            # 배경 이미지를 원본 프레임과 동일한 크기로 조정
            bg_resized_frame = cv2.resize(bg_frame, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_CUBIC)

            # 사용자 배경과 합성
            combined = np.where(condition, user_masked, bg_resized_frame)
            return combined, condition

    print("No contours detected for the user.")
    return frame, None
