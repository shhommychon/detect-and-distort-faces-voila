import cv2
import numpy as np

from deepface.basemodels import DeepID
from tensorflow.keras.preprocessing import image


def build_deepface_deepid():
    return DeepID.loadModel()


def preprocess_deepid_input(img, region, target_size=(55, 47), grayscale=False, enforce_detection=True, detector_backend="ssd", return_region=False, align=True):
    """
    
    original code: https://github.com/serengil/deepface/blob/eeaf1253da024e9045a9c0b76771413ed1a9e805/deepface/commons/functions.py#L172-L235
    """
    if (img.shape[0] == 0 or img.shape[1] == 0) and enforce_detection == True:
        raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")

    #--------------------------

    #post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #---------------------------------------------------

    #resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    #------------------------------------------

    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    #---------------------------------------------------

    #normalizing the image pixels

    img_pixels = image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]

    #---------------------------------------------------

    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels


def face_to_embedding(face_recognizer, img):
    return face_recognizer.predict(img)


def recognize_face(face_recognizer, img):
    pass
