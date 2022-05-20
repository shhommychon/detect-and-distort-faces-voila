# original code by 차준영
#   - https://dsbook.tistory.com/194

import cv2
import numpy as np

def averaging(img, x, y, w, h, kernel_size=10, env="colab"):
    """일반 블러

    가장 일반적인 필터링 방법.
    균일한 값을 가진 정규화 된 커널을 이용한 이미지 필터링 방법.

    Params:
        kernel_size: 블러 커널 사이즈
    Returns:
        roi: 평균 블러 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    roi = cv2.blur(roi, (kernel_size, kernel_size))

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi

def gaussian_filtering(img, x, y, w, h, scale=0.1, kernel_size=7, env="colab"):
    """가우시안 블러

    균일한 값을 가진 가우시안 함수를 이용한 커널을 이용한 이미지 필터링 방법.
    중앙 위치에 놓인 픽셀과 가까울 수록 가중치가 높아지고 멀어질 수록 가중치가 작아져
    중앙 픽셀과 멀어질 수록 해당 픽셀값에 대한 영향력이 작아진다.
    전체적으로 밀도가 동일한 노이즈 등을 제거하는데 가장 효과적.

    Params:
        kernel_size: 블러 커널 사이즈
    Returns:
        roi: 가우시안 블러 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    img_gaussian = np.clip(
        (roi/255 + np.random.normal(scale=scale, size=roi.shape)) * 255, 
        0, 
        255
    ).astype("uint8")

    roi = cv2.GaussianBlur(img_gaussian, (kernel_size, kernel_size), 0)

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi

def median_filtering(img, x, y, w, h, kernel_size=9, env="colab"):
    """중앙값 블러

    kernel window에 있는 모든 픽셀들을 정렬한 후 중간값을 선택하여 적용.
    항상 이미지의 일부 픽셀 값으로 대체된다.
    점 잡음(salt-and-pepper noise) 제거에 효과적이다.

    Params:
        kernel_size: 블러 커널 사이즈
    Returns:
        roi: 중앙값 블러 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    roi = cv2.medianBlur(roi, kernel_size)

    return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)

def bilateral_filtering(img, x, y, w, h, scale=0.1, kernel_size=9, env="colab"):
    """양방향 블러

    두 픽셀의 거리 차이를 고려하며
    두 픽셀의 명암값 차이 또한 커널에 넣어서 가중치로 곱한다. 
    픽셀 값의 차이가 너무 크면 가중치가 0에 가까운 값이 되어 합쳐지지 않으므로
    영역과 영역 사이의 경계선이 잘 보존될 수 있다.

    Params:
        kernel_size: 블러 커널 사이즈
    Returns:
        roi: 양방향 블러 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    img_bilateral = np.clip(
        (roi/255 + np.random.normal(scale=scale, size=roi.shape)) * 255, 
        0, 
        255
    ).astype("uint8")

    roi = cv2.bilateralFilter(img_bilateral, kernel_size, 75, 75)

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi
