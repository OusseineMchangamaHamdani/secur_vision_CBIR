import cv2
import time
try:
    from .featureExtractor import FeatureExtractor
    from .db_manager import DatabaseManager
except ImportError:
    from featureExtractor import FeatureExtractor
    from db_manager import DatabaseManager

def saveInDB(path_img, infoEmp, extractor):
    
    image_capturee_path = path_img
    print(f"\n--- Démarrage de l'extraction pour {image_capturee_path} ---")
    start = time.perf_counter()
    
    # --- 2. Phase d'Extraction vectorielle ---
    image = cv2.imread(image_capturee_path)
    
    # On extrait le vecteur de 512 dimensions avec FaceNet
    vecteur_facial = extractor.process_image(image)
    
    if vecteur_facial is not None:
        # --- 3. Phase d'Envoi vers la Base de Données ---
        print("\n--- Envoi vers PostgreSQL ---")
        db = DatabaseManager()
        
        # Insertion complète : Employé + Photo + Vecteur
        db.insert_new_employee(
            employee_data=infoEmp, # <-- 2. Utilisation du paramètre infoEmp ici
            image_path=image_capturee_path,
            embedding=vecteur_facial,
            model_name="facenet",
            dimension=512
        )
        db.close()
        
        elapsed = time.perf_counter() - start
        print(f"Extraction et insertion terminées avec succès en {elapsed:.4f} s.")
    else:
        elapsed = time.perf_counter() - start
        print(f"Erreur : Aucun visage valide détecté lors de l'extraction. (Temps écoulé : {elapsed:.4f} s)")