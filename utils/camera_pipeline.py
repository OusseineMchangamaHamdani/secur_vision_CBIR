import os
import sys
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional

# Supprimer tous les logs parasites
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
    from .eye_detector import eyes_are_open
    from .stability import FaceStability
    from .capture import CaptureManager
    from .search import search_face
    from .featureExtractor import FeatureExtractor
    from .db_manager import DatabaseManager
except (ImportError, ValueError):
    from eye_detector import eyes_are_open
    from stability import FaceStability
    from capture import CaptureManager
    from search import search_face
    from featureExtractor import FeatureExtractor
    from db_manager import DatabaseManager


class CameraPipeline:
    def __init__(self, countdown_duration: float = 5.0):
        self.countdown_duration = countdown_duration
        
        # Localisation des fichiers cascade (dossier local utils ou cv2.data)
        face_xml = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
        eye_xml = os.path.join(current_dir, 'haarcascade_eye.xml')

        if not os.path.exists(face_xml):
            face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(eye_xml):
            eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'

        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        
        self.stability = FaceStability(threshold=25)
        self.capture_mgr = CaptureManager()
        
        self.extractor = FeatureExtractor(model_name="facenet")
        self.db = DatabaseManager()

        # Charger l'historique persistant depuis PostgreSQL
        try:
            self.attendance_history = self.db.get_search_history(limit=50)
        except Exception:
            self.attendance_history = []

        # État du pipeline
        self.timer_start = None
        self.is_analyzing = False
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_captured_image: Optional[str] = None
        self.result_display_timer = None
        self.attendance_history = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Traite un frame vidéo : détection faciale, vérification des yeux et stabilité,
        affichage du compte à rebours HUD et déclenchement de la recherche à 5s.
        """
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        # Masquer temporairement le dernier résultat après 6 secondes d'affichage
        if self.result_display_timer and (time.time() - self.result_display_timer > 6.0):
            self.last_result = None
            self.result_display_timer = None

        if len(faces) == 0:
            self.timer_start = None
            cv2.putText(display_frame, "CHURCH OF FACE - RECHERCHE DE VISAGE...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return display_frame

        # Prendre le visage principal (le plus grand)
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        (x, y, w, h) = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]

        # Détection des yeux dans le visage
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        eyes_open = len(eyes) >= 1  # Au moins un œil ou les deux yeux détectés

        # Stabilité de la tête
        face_center = (x + w // 2, y + h // 2)
        is_stable = self.stability.is_stable(face_center)

        # Dessiner le rectangle HUD autour du visage
        box_color = (0, 255, 0) if (eyes_open and is_stable) else (0, 215, 255)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

        # Condition de déclenchement du compte à rebours de 5 secondes
        if eyes_open and not self.is_analyzing and self.result_display_timer is None:
            if self.timer_start is None:
                self.timer_start = time.time()

            elapsed = time.time() - self.timer_start
            remaining = max(0.0, self.countdown_duration - elapsed)

            # Dessiner le Compte à Rebours 5s sur le HUD
            countdown_text = f"CAPTURE DANS : {remaining:.1f}s"
            cv2.rectangle(display_frame, (x, y - 40), (x + w, y - 5), (0, 0, 0), -1)
            cv2.putText(display_frame, countdown_text, (x + 10, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Barre de progression
            progress = min(1.0, elapsed / self.countdown_duration)
            bar_w = int(w * progress)
            cv2.rectangle(display_frame, (x, y - 5), (x + bar_w, y), (0, 255, 0), -1)

            # Déclenchement de la capture et de la recherche vectorielle à 5s
            if remaining <= 0:
                self.is_analyzing = True
                cv2.putText(display_frame, "ANALYSE VECTORIELLE EN COURS...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                # Capture de l'image
                cap_filename = self.capture_mgr.capture(frame)
                self.last_captured_image = cap_filename
                print(f"\n[Caméra Pipeline] Capture automatique enregistrée : {cap_filename}")

                # Recherche vectorielle
                search_res = search_face(frame, top_k=5, threshold=0.75, extractor=self.extractor, db=self.db)
                self.last_result = search_res
                self.result_display_timer = time.time()
                self.timer_start = None
                self.is_analyzing = False

                if search_res.get("status") == "found":
                    bm = search_res["best_match"]
                    entry = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "first_name": bm["first_name"],
                        "last_name": bm["last_name"],
                        "department": bm.get("department", "Général"),
                        "position": bm.get("position", "Employé"),
                        "similarity": bm.get("similarity_percent", 0.0),
                        "status": "APPROVED",
                        "image_path": cap_filename
                    }
                    self.attendance_history.insert(0, entry)
                    self.db.log_search_event(entry)
                elif search_res.get("status") in ["unknown_person", "no_face_detected"]:
                    entry = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "first_name": "Personne",
                        "last_name": "Inconnue",
                        "department": "---",
                        "position": "Accès Refusé",
                        "similarity": 0.0,
                        "status": "REJECTED",
                        "image_path": cap_filename
                    }
                    self.attendance_history.insert(0, entry)
                    self.db.log_search_event(entry)
        else:
            if not self.is_analyzing:
                self.timer_start = None
                status_txt = "GARDEZ LES YEUX OUVERTS & RESTEZ IMMOBILE" if not eyes_open else "STABILISANT..."
                cv2.putText(display_frame, status_txt, (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Affichage du badge de résultat sur l'image si récent
        if self.last_result and self.last_result.get("status") == "found":
            bm = self.last_result["best_match"]
            res_txt = f"IDENTIFIE : {bm['first_name']} {bm['last_name']} ({bm['similarity_percent']}%)"
            cv2.rectangle(display_frame, (20, 20), (580, 70), (0, 100, 0), -1)
            cv2.putText(display_frame, res_txt, (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        elif self.last_result and self.last_result.get("status") == "unknown_person":
            cv2.rectangle(display_frame, (20, 20), (520, 70), (0, 0, 150), -1)
            cv2.putText(display_frame, "PERSONNE INCONNUE - ACCES REFUSE", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return display_frame

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état courant pour l'interface web SaaS."""
        return {
            "last_result": self.last_result,
            "last_captured_image": self.last_captured_image,
            "attendance_history": self.attendance_history[:10],
            "timer_active": self.timer_start is not None,
            "is_analyzing": self.is_analyzing
        }
