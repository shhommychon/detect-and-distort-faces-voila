from deepface.detectors import FaceDetector


def build_deepface_ssd():
    return FaceDetector.build_model("ssd")

def detect_faces(face_detector, img):
    try:
        # faces store list of detected_face and region pair
        faces = FaceDetector.detect_faces(face_detector, "ssd", img, align=False)
    except:
        return None
    
    return faces
