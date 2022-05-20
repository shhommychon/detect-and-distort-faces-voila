# original code by Baek-Kyun Shin(https://github.com/BaekKyunShin)
#   - https://bkshin.tistory.com/entry/OpenCV-21-블러링을-활용한-모자이크-처리-이미지-스케치-효과-적용하기
#   - https://github.com/BaekKyunShin/OpenCV_Project_Python/blob/master/06.filter/workshop_painting_cam.py

import cv2
import numpy as np

def sketch(img, x, y, w, h, sketch_type="white", env="colab"):
    """스케치 효과
    
    Params:
        sketch_type: "white" 또는 "merged"
    Returns:
        roi: 스케치 효과 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 그레이 스케일로 변경
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 잡음 제거를 위해 가우시안 플러 필터 적용(라플라시안 필터 적용 전에 필수)
    img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
    # 라플라시안 필터로 엣지 검출
    edges = cv2.Laplacian(img_gray, -1, None, 5)
    # 스레시홀드로 경계 값 만 남기고 제거하면서 화면 반전(흰 바탕 검은 선)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    # 경계선 강조를 위해 침식 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    sketch = cv2.erode(sketch, kernel)
    # 경계선 자연스럽게 하기 위해 미디언 블러 필터 적용
    sketch = cv2.medianBlur(sketch, 5)

    if sketch_type=="white":
        # 그레이 스케일에서 BGR 컬러 스케일로 변경
        if env=="colab":
            img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGRA)
        else:
            img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        return img_sketch
    
    elif sketch_type=="merged":
        # 컬러 이미지 선명선을 없애기 위해 평균 블러 필터 적용
        img_paint = cv2.blur(roi, (10,10))
        # 컬러 영상과 스케치 영상과 합성
        img_paint = cv2.bitwise_and(img_paint, img_paint, mask=sketch)

        if env=="colab":
            return cv2.cvtColor(img_paint, cv2.COLOR_RGB2BGRA)
        else:
            return img_paint
    
    else:
        raise ValueError("sketch_type은 white 또는 merged 둘 중 하나여야 합니다.")
