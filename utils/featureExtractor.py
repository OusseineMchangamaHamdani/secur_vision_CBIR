import os
import sys

# Supprimer tous les logs et avertissements parasites TensorFlow, oneDNN, absl et Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
# pyrefly: ignore [missing-import]
import mtcnn
import numpy as np
import time
from sklearn.preprocessing import normalize
# pyrefly: ignore [missing-import]
from keras_facenet import FaceNet


class FeatureExtractor:

    def __init__(self, model_name="facenet"):
        self.detector = mtcnn.MTCNN()
        self.model_name = model_name.lower()

        if self.model_name == "facenet":
            self.embedder = FaceNet()
        else:
            raise ValueError("Modèle inconnu. Utilisez 'facenet'.")

    def detect_faces(self, image):
        faces = self.detector.detect_faces(image)
        return faces

    def crop_face(self, image, face):
        x, y, w, h = face["box"]
        x = max(0, x)
        y = max(0, y)
        face_crop = image[y:y+h, x:x+w]
        return face_crop

    def process_image(self, image):
        faces = self.detect_faces(image)

        if len(faces) == 0:
            return None

        # Découpage du visage
        face = self.crop_face(image, faces[0])
        if face is None or face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            return None

        # keras-facenet attend des images en RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Extraction
        start = time.perf_counter()
        # La méthode embeddings prend une liste d'images et retourne un array (N, 512)
        embeddings = self.embedder.embeddings([face_rgb])
        elapsed = time.perf_counter() - start

        # Normalisation L2 pour la similarité cosinus
        features = normalize(embeddings)


        return features.flatten()