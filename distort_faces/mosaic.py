# original code by Baek-Kyun Shin(https://github.com/BaekKyunShin)
#   - https://bkshin.tistory.com/entry/OpenCV-16-모자이크-처리Mosaic-리퀴파이Liquify-왜곡-거울Distortion-Mirror
#   - https://github.com/BaekKyunShin/OpenCV_Project_Python/blob/master/05.geometric_transform/workshop_mosaic.py

import cv2

def mosaic(img, x, y, w, h, rate=15, env="colab"):
    """모자이크
    
    Params:
        rate: 모자이크에 사용할 축소 비율 (1/rate)
    Returns:
        roi: 모자이크 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 1/rate 비율로 축소
    roi = cv2.resize(roi, (w//rate, h//rate))
    # 원래 크기로 확대
    roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)

    if env=="colab":
        return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)
    else:
        return roi
