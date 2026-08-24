import os
import sys
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Supprimer tous les logs et avertissements parasites TensorFlow, oneDNN, absl et Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .featureExtractor import FeatureExtractor
    from .db_manager import DatabaseManager
except (ImportError, ValueError):
    from featureExtractor import FeatureExtractor
    from db_manager import DatabaseManager


def search_face(
    image_input: Union[str, np.ndarray],
    top_k: int = 5,
    threshold: float = 0.75,
    extractor: Optional[FeatureExtractor] = None,
    db: Optional[DatabaseManager] = None,
) -> Dict[str, Any]:
    """
    Pipeline complet de recherche faciale vectorielle (Image / Frame -> Embedding 512D -> pgvector search) :
    1. image path / frame : Chargement de l'image (fichier ou frame caméra numpy)
    2. load image : Prétraitement et détection du visage
    3. extract embedding : Extraction du vecteur d'embedding 512D FaceNet
    4. search nearest embeddings in PostgreSQL : Requête cosinus pgvector
    5. compute similarity & return best match : Retourne l'employé reconnu
    """
    start_time = time.perf_counter()

    # 1. image path & load image
    if isinstance(image_input, str):
        resolved_path = image_input
        if not os.path.exists(resolved_path):
            cand1 = os.path.abspath(os.path.join(parent_dir, image_input))
            root_dir = os.path.dirname(parent_dir)
            cand2 = os.path.abspath(os.path.join(root_dir, image_input))
            if os.path.exists(cand1):
                resolved_path = cand1
            elif os.path.exists(cand2):
                resolved_path = cand2

        image = cv2.imread(resolved_path)
        if image is None:
            return {"status": "error", "message": f"Impossible de charger l'image {image_input}"}
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        return {"status": "error", "message": "Format d'entrée d'image non valide."}

    # 2. preprocess & extract embedding
    if extractor is None:
        extractor = FeatureExtractor(model_name="facenet")

    embedding = extractor.process_image(image)
    if embedding is None:
        return {"status": "no_face_detected", "message": "Aucun visage détecté sur l'image."}

    # 3. search nearest embeddings in PostgreSQL
    close_db_on_finish = False
    if db is None:
        try:
            db = DatabaseManager()
            close_db_on_finish = True
        except Exception as e:
            return {"status": "db_error", "message": f"Erreur de connexion à la base : {e}"}

    try:
        results = db.search_nearest_face(embedding, top_k=top_k, threshold=threshold)
        elapsed = time.perf_counter() - start_time

        if not results:
            return {
                "status": "unknown_person",
                "message": "Aucun profil correspondant trouvé dans la base de données (Seuil non atteint).",
                "similarity_score": 0.0,
                "elapsed_seconds": elapsed,
            }

        best_match = results[0]
        return {
            "status": "found",
            "best_match": best_match,
            "top_matches": results,
            "elapsed_seconds": round(elapsed, 4),
        }
    finally:
        if close_db_on_finish and db:
            db.close()


if __name__ == "__main__":
    print("--- Test du pipeline de recherche faciale (search_face) ---")
    test_image = "captures/Ousseine/2026-07-10_16-29-13.jpg"
    print(f"Recherche pour l'image : {test_image}")

    res = search_face(test_image)
    print("Résultat :", res.get("status"))
    if res.get("status") == "found":
        match = res["best_match"]
        print(f"-> Employé reconnu : {match['first_name']} {match['last_name']} ({match['similarity_percent']}% de similarité)")