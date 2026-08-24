import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file

# Supprimer tous les logs parasites TensorFlow / oneDNN / Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.camera_pipeline import CameraPipeline
from utils.search import search_face

app = Flask(__name__, template_folder="templates")

# Initialiser le pipeline caméra & recherche
pipeline = CameraPipeline(countdown_duration=3.0)

# Capture vidéo globale
cap = cv2.VideoCapture(0)


def generate_frames():
    """Générateur de flux vidéo MJPEG avec superposition du HUD et compte à rebours 5s."""
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMÉRA INDISPONIBLE - MODE DEMO", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            time.sleep(0.05)
        else:
            frame = cv2.flip(frame, 1)

        processed_frame = pipeline.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Interface Web SaaS Bento Grid."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint de flux vidéo en direct (MJPEG)."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_image')
def get_image():
    """Sert une image à partir de son chemin de fichier."""
    image_path = request.args.get('path')
    if image_path:
        if os.path.exists(image_path):
            return send_file(os.path.abspath(image_path))
        cand1 = os.path.abspath(os.path.join(current_dir, image_path))
        parent_dir = os.path.dirname(current_dir)
        cand2 = os.path.abspath(os.path.join(parent_dir, image_path))
        if os.path.exists(cand1):
            return send_file(cand1)
        elif os.path.exists(cand2):
            return send_file(cand2)
    return jsonify({"error": "Image introuvable"}), 404


@app.route('/api/status', methods=['GET'])
def get_status():
    """Retourne l'état courant de l'identification, la capture et l'historique."""
    return jsonify(pipeline.get_status())


@app.route('/api/trigger_search', methods=['POST'])
def trigger_search():
    """Déclenche immédiatement une capture et recherche manuelle sur le flux courant."""
    success, frame = cap.read()
    if success and frame is not None:
        frame = cv2.flip(frame, 1)
        cap_filename = pipeline.capture_mgr.capture(frame)
        pipeline.last_captured_image = cap_filename
        res = search_face(frame, top_k=5, threshold=0.75, extractor=pipeline.extractor, db=pipeline.db)
        pipeline.last_result = res
        pipeline.result_display_timer = time.time()
        return jsonify(res)
    return jsonify({"status": "error", "message": "Impossible d'accéder à la caméra."})


if __name__ == "__main__":
    if "--web" in sys.argv:
        print("\n==================================================================")
        print(" Lancement de BioVector SaaS - Serveur Web Flask")
        print(" Accédez à l'interface visuelle : http://127.0.0.1:5000")
        print("==================================================================\n")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        print("\n==================================================================")
        print(" Lancement de BioVector Desktop App (PySide6 / Qt pour Python)")
        print("==================================================================\n")
        from gui_app import BioVectorApp, QtWidgets
        pyside_app = QtWidgets.QApplication(sys.argv)
        window = BioVectorApp()
        window.show()
        sys.exit(pyside_app.exec())
