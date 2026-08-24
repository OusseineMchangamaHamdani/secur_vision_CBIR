import os
import sys

# Supprimer tous les avertissements et logs parasites TensorFlow / oneDNN / Keras / absl
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

try:
    # pyrefly: ignore [missing-import]
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import cv2
import time
from typing import Dict, Any, Optional

# Ajouter le chemin courant et parent pour permettre l'exécution directe en script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .featureExtractor import FeatureExtractor
    from .db_manager import DatabaseManager
    from .dataStruct import UserData
    from .saveInDB import saveInDB
except (ImportError, ValueError):
    from featureExtractor import FeatureExtractor
    from db_manager import DatabaseManager
    from dataStruct import UserData
    from saveInDB import saveInDB


def index_image(
    path_image: str,
    info_emp: Dict[str, Any],
    extractor: Optional[FeatureExtractor] = None,
    db: Optional[DatabaseManager] = None,
) -> Dict[str, Any]:
    """
    Pipeline complet d'indexation d'une image faciale :
    1. Vérification du chemin de l'image (image path)
    2. Chargement de l'image (load image)
    3. Prétraitement et détection (preprocess image)
    4. Extraction du vecteur d'embedding 512D (extract embedding)
    5. Sauvegarde de l'embedding et des métadonnées dans PostgreSQL (save embedding + metadata in PostgreSQL)
    """
    print(f"\n=================== PHASE D'INDEXATION ===================")
    start_time = time.perf_counter()

    # --- 1. Image path ---
    resolved_path = path_image
    if not os.path.exists(resolved_path):
        # 1. Tester par rapport au dossier pgvectorTest
        candidate1 = os.path.abspath(os.path.join(parent_dir, path_image))
        # 2. Tester par rapport au dossier racine pointage Facial
        root_dir = os.path.dirname(parent_dir)
        candidate2 = os.path.abspath(os.path.join(root_dir, path_image))

        if os.path.exists(candidate1):
            resolved_path = candidate1
        elif os.path.exists(candidate2):
            resolved_path = candidate2
        else:
            err_msg = f"Erreur [Indexation] : Fichier image introuvable à {path_image}"
            print(err_msg)
            return {"status": "error", "message": err_msg}

    # --- 2. Load image ---
    image = cv2.imread(resolved_path)
    if image is None:
        err_msg = f"Erreur [Indexation] : Impossible de lire l'image {resolved_path}"
        print(err_msg)
        return {"status": "error", "message": err_msg}

    print(f"[1/5] Image chargée avec succès depuis : {resolved_path}")

    # --- 3. Preprocess image & 4. Extract embedding ---
    if extractor is None:
        print("[2/5] Initialisation du modèle FeatureExtractor (MTCNN + FaceNet)...")
        extractor = FeatureExtractor(model_name="facenet")

    print("[3/5 & 4/5] Détection du visage et extraction de l'embedding...")
    embedding = extractor.process_image(image)

    if embedding is None:
        err_msg = f"Erreur [Indexation] : Aucun visage valide détecté dans {path_image}"
        print(err_msg)
        return {"status": "error", "message": err_msg}

    print(f"[4/5] Embedding extrait avec succès (Dimension : {len(embedding)})")

    # --- 5. Save embedding + metadata in PostgreSQL ---
    print("[5/5] Sauvegarde en base de données PostgreSQL...")
    close_db_on_finish = False
    if db is None:
        try:
            db = DatabaseManager()
            close_db_on_finish = True
        except Exception as e:
            print(f"[Attention] Connexion DB échouée ({e}). L'embedding a bien été extrait localement.")
            elapsed = time.perf_counter() - start_time
            return {
                "status": "extracted_without_db",
                "embedding": embedding,
                "path_image": path_image,
                "info_emp": info_emp,
                "elapsed_seconds": elapsed,
            }

    try:
        employee_id = db.insert_new_employee(
            employee_data=info_emp,
            image_path=path_image,
            embedding=embedding,
            model_name="facenet",
            dimension=len(embedding),
        )
        elapsed = time.perf_counter() - start_time
        print(f"==========================================================")
        print(f" Indexation réussie pour l'employé ID {employee_id} en {elapsed:.4f} s")
        print(f"==========================================================\n")

        return {
            "status": "success",
            "employee_id": employee_id,
            "embedding": embedding,
            "path_image": path_image,
            "info_emp": info_emp,
            "elapsed_seconds": elapsed,
        }
    finally:
        if close_db_on_finish and db:
            db.close()


def index_user(
    nom: str,
    prenom: str,
    path_image: str,
    employee_number: str = "EMP001",
    email: Optional[str] = None,
    department: str = "General",
    position: str = "Employé",
    extractor: Optional[FeatureExtractor] = None,
    db: Optional[DatabaseManager] = None,
) -> Dict[str, Any]:
    """
    Raccourci pour indexer un employé à partir de ses informations de base.
    """
    info_emp = {
        "number": employee_number,
        "first_name": prenom,
        "last_name": nom,
        "email": email or f"{prenom.lower()}.{nom.lower()}@example.com",
        "department": department,
        "position": position,
    }
    return index_image(path_image=path_image, info_emp=info_emp, extractor=extractor, db=db)


def index_folder(
    folder_path: str,
    info_emp: Dict[str, Any],
    extractor: Optional[FeatureExtractor] = None,
    db: Optional[DatabaseManager] = None,
) -> Dict[str, Any]:
    """
    Indexe l'ensemble des images contenues dans un dossier pour une même personne.
    1. Crée/Récupère la fiche personne dans la base de données.
    2. Prétraite et extrait le vecteur d'embedding pour chaque photo du dossier.
    3. Associe tous les vecteurs extraits à cette personne dans PostgreSQL.
    """
    resolved_folder = folder_path
    if not os.path.exists(resolved_folder):
        candidate1 = os.path.abspath(os.path.join(parent_dir, folder_path))
        root_dir = os.path.dirname(parent_dir)
        candidate2 = os.path.abspath(os.path.join(root_dir, folder_path))
        if os.path.exists(candidate1):
            resolved_folder = candidate1
        elif os.path.exists(candidate2):
            resolved_folder = candidate2
        else:
            err_msg = f"Erreur [Indexation Dossier] : Dossier introuvable {folder_path}"
            print(err_msg)
            return {"status": "error", "message": err_msg}

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        f for f in os.listdir(resolved_folder)
        if os.path.isfile(os.path.join(resolved_folder, f)) and f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        err_msg = f"Aucune image valide trouvée dans {resolved_folder}"
        print(err_msg)
        return {"status": "error", "message": err_msg}

    print(f"\n=================== INDEXATION DU DOSSIER ===================")
    print(f"Employé : {info_emp.get('first_name')} {info_emp.get('last_name')}")
    print(f"Dossier : {resolved_folder} ({len(image_files)} images trouvées)")

    if extractor is None:
        print("Initialisation du modèle FeatureExtractor (MTCNN + FaceNet)...")
        extractor = FeatureExtractor(model_name="facenet")

    close_db_on_finish = False
    if db is None:
        try:
            db = DatabaseManager()
            close_db_on_finish = True
        except Exception as e:
            print(f"[DB Warning] Impossible de se connecter à la base DB: {e}")

    try:
        user_id = db.get_or_create_employee(info_emp) if db else None
        if user_id:
            print(f"ID Employé dans la Base de Données : {user_id}")

        indexed_count = 0
        for idx, filename in enumerate(image_files, 1):
            img_path = os.path.join(resolved_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            vector = extractor.process_image(image)
            if vector is not None:
                if db and user_id:
                    db.add_embedding(user_id, img_path, vector, model_name="facenet", dimension=len(vector))
                indexed_count += 1
                print(f"  [{idx}/{len(image_files)}] {filename} -> Visage détecté & Vectorisé 512D")
            else:
                print(f"  [{idx}/{len(image_files)}] {filename} -> Aucun visage détecté (ignoré)")

        print(f"===========================================================")
        print(f" Bilan : {indexed_count}/{len(image_files)} images indexées avec succès pour {info_emp.get('first_name')} {info_emp.get('last_name')}")
        print(f"===========================================================\n")

        return {
            "status": "success",
            "user_id": user_id,
            "total_images": len(image_files),
            "indexed_images": indexed_count,
        }
    finally:
        if close_db_on_finish and db:
            db.close()


if __name__ == "__main__":
    print("--- Batch Indexation de dossiers de personnes ---")

    # 1. Personne 1 : Ousseine Mchangama
    emp1 = {
        "number": "EMP_001",
        "first_name": "Ousseine",
        "last_name": "Mchangama",
        "email": "ousseine.mchangama@example.com",
        "department": "R&D",
        "position": "Ingénieur IA",
    }
    index_folder("captures/Ousseine", emp1)

    # 2. Personne 2 : Abdou Ali ELAnzaff
    emp2 = {
        "number": "EMP_002",
        "first_name": "Abdou Ali",
        "last_name": "ELAnzaff",
        "email": "elanzaff.abdou@example.com",
        "department": "Direction",
        "position": "Manager",
    }
    index_folder("captures/Anzaff", emp2)