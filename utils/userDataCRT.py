from .dataStruct import UserData
try:
    from .featureExtractor import FeatureExtractor
except ImportError:
    from featureExtractor import FeatureExtractor
import cv2
def createuserData(nom,prenom,path_image,face_image):
    extractor = FeatureExtractor()
    embedding = extractor.process_image(cv2.imread(path_image))
    if embedding is None:
        raise ValueError("Failed to extract embedding from image")
    else:
        user={
            "nom": nom,
            "prenom": prenom,
            "path_image": path_image,
            "face_image": face_image,
            "embedding": embedding,
        }
        return UserData(**user)