# original code by Baek-Kyun Shin(https://github.com/BaekKyunShin)
#   - https://bkshin.tistory.com/entry/OpenCV-16-모자이크-처리Mosaic-리퀴파이Liquify-왜곡-거울Distortion-Mirror
#   - https://github.com/BaekKyunShin/OpenCV_Project_Python/blob/master/05.geometric_transform/workhop_distotion_camera.py

import cv2
import numpy as np

def wave(img, x, y, w, h, env="colab"):
    """물경 왜곡 효과

    Returns:
        roi: 물결 효과 처리된 Region of Interest
    """
    # 물결 효과
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_wave_x, map_wave_y = map_x.copy(), map_y.copy()
    map_wave_x = map_wave_x + 15*np.sin(map_y/20)
    map_wave_y = map_wave_y + 15*np.sin(map_x/20)

    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 준비한 매핑 좌표로 영상 효과 적용
    roi = cv2.remap(roi, map_wave_x, map_wave_y, cv2.INTER_LINEAR, \
                    None, cv2.BORDER_REPLICATE)

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi

def convex(img, x, y, w, h, env="colab"):
    """볼록 렌즈 효과

    Returns:
        roi: 볼록 렌즈 효과 처리된 Region of Interest
    """
    # 렌즈 효과
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    ## 렌즈 효과, 중심점 이동
    map_lenz_x = 2*map_x/(w-1)-1
    map_lenz_y = 2*map_y/(h-1)-1
    ## 렌즈 효과, 극좌표 변환
    r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
    r_convex = r.copy()
    ## 볼록 렌즈 효과 매핑 좌표 연산
    r_convex[r<1] = r_convex[r<1] ** 2
    ## 렌즈 효과, 직교 좌표 복원
    map_convex_x, map_convex_y = cv2.polarToCart(r_convex, theta)
    ## 렌즈 효과, 좌상단 좌표 복원
    map_convex_x = ((map_convex_x+1)*w - 1)/2
    map_convex_y = ((map_convex_y+1)*h - 1)/2

    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 준비한 매핑 좌표로 영상 효과 적용
    roi = cv2.remap(roi, map_convex_x, map_convex_y, cv2.INTER_LINEAR)

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi

def concave(img, x, y, w, h, env="colab"):
    """오목 렌즈 효과
    
    Returns:
        roi: 오목 렌즈 효과 처리된 Region of Interest
    """
    # 렌즈 효과
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    ## 렌즈 효과, 중심점 이동
    map_lenz_x = 2*map_x/(w-1)-1
    map_lenz_y = 2*map_y/(h-1)-1
    ## 렌즈 효과, 극좌표 변환
    r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
    r_concave = r.copy()
    ## 오목 렌즈 효과 매핑 좌표 연산
    r_concave[r<1] = r_concave[r<1] ** 0.5
    ## 렌즈 효과, 직교 좌표 복원
    map_concave_x, map_concave_y = cv2.polarToCart(r_concave, theta)
    ## 렌즈 효과, 좌상단 좌표 복원
    map_concave_x = ((map_concave_x+1)*w - 1)/2
    map_concave_y = ((map_concave_y+1)*h - 1)/2

    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 준비한 매핑 좌표로 영상 효과 적용
    roi = cv2.remap(roi, map_concave_x, map_concave_y, cv2.INTER_LINEAR)

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi
