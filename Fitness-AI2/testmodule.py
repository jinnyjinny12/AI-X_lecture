##모듈 추가
import cv2
import mediapipe as mp
import numpy as np

##step1 메서드 초기화
class PoseProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

        self.mp_draw = mp.solutions.drawing_utils ## drawing_utils 를 사용해서 랜드마크 연결선을 그리는 기능
        self.previous_mask = None  # 이전 마스크 저장용 변수 추가

    ## step2 포즈감지
    def find_pose(self, img, draw=True):
        """Find pose landmarks in the given image."""

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # 이전 마스크와 현재 마스크 평균화
        if results.segmentation_mask is not None:
            if self.previous_mask is not None:
                results.segmentation_mask = cv2.addWeighted(
                    self.previous_mask, 0.7, results.segmentation_mask, 0.3, 0
                )
            self.previous_mask = results.segmentation_mask

        # 랜드마크 그리기 (스켈레톤)
        if draw and results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                img, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return img, results


    ##step3 배경 분리
    def segment_background(self, img, results, bg_img):
        """Segment the background and place a resized user onto it."""
        if results.segmentation_mask is not None:
            # Create segmentation mask condition

                ##result.segmentation_mask : 상ㅇ자의 포즈와 배경 구분
                ## 0~ 1 사이 값으로 구성된 2D 배열을 기준으로 사용자(1) 배경(0)로 감지
                ##(results.segmentation_mask,) * 3 : 2d 배열을 튜플로 변환 ,3개의 동일한 마스크 생성
                ##np.stack : 2D 배열을 쌓아 3D 배열로 확장
                ## 확률값에 따라 0.1 이상의 값을 ture 로 감지

            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

            # 사용자 영역 추출
            user_only = np.where(condition, img, 0)

            ##condition : 사용자 영역 true, 배경 False
            ##(480, 640, 3D)

            # 사용자의 영역을 흑백으로 만들어 감지
            gray_user = cv2.cvtColor(user_only, cv2.COLOR_BGR2GRAY)

            ##사용자 영역의 윤곽선 발견
            ##cv2.RETR_EXTERNAL 외부 윤곽선 감지
            ##cv2.CHAIN_APPROX_SIMPLE 윤곽선 정보 간소화 저장
            contours, _ = cv2.findContours(gray_user, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ##contours : 발견된 윤곽선의 좌표 리스트
            if contours:
                # 가장 면적이 큰 윤곽선 계산 : 사용자의 영역
                largest_contour = max(contours, key=cv2.contourArea)

                # 사용자 마스크 생성
                mask = np.zeros_like(gray_user)
                cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

                # 사용자 모양 유지
                ## 흑백을 기준으로 유저만 감지해 영상에 임포트
                user_masked = cv2.bitwise_and(img, img, mask=mask)

                #배경 이미지를 원본 프레임과 동일한 크기로 조정
                bg_resized = cv2.resize(bg_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

                #사용자 배경과 합성
                combined = np.where(condition, user_masked, bg_resized)

                return combined, condition

        print("No segmentation mask detected.")
        return img, None


