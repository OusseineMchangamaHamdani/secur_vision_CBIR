import os
import sys
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional

# Configuration des variables d'environnement TensorFlow / Keras pour désactiver les avertissements
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

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QDateTime, QPropertyAnimation, QVariantAnimation, QEasingCurve, QRectF
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QIcon, QPainter, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QFrame,
    QLineEdit, QComboBox, QMessageBox, QCheckBox
)


class LucideIcon:
    """Générateur d'icônes vectorielles épurées style Lucide pour l'interface SVA."""

    @staticmethod
    def draw_dashboard(color="#8DA2C3", size=18) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        p.setPen(pen)
        p.drawRect(2, 2, 6, 6)
        p.drawRect(10, 2, 6, 6)
        p.drawRect(2, 10, 6, 6)
        p.drawRect(10, 10, 6, 6)
        p.end()
        return pix

    @staticmethod
    def draw_history(color="#8DA2C3", size=18) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        p.setPen(pen)
        p.drawEllipse(2, 2, 14, 14)
        p.drawLine(9, 5, 9, 9)
        p.drawLine(9, 9, 12, 11)
        p.end()
        return pix

    @staticmethod
    def draw_user(color="#8DA2C3", size=18) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        p.setPen(pen)
        p.drawEllipse(5, 2, 8, 8)
        path = QPainterPath()
        path.moveTo(2, 16)
        path.quadTo(9, 10, 16, 16)
        p.drawPath(path)
        p.end()
        return pix

    @staticmethod
    def draw_database(color="#8DA2C3", size=18) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        p.setPen(pen)
        p.drawEllipse(2, 2, 14, 4)
        path1 = QPainterPath()
        path1.moveTo(2, 4)
        path1.quadTo(9, 8, 16, 4)
        path1.lineTo(16, 9)
        path1.quadTo(9, 13, 2, 9)
        path1.lineTo(2, 4)
        p.drawPath(path1)
        p.end()
        return pix

    @staticmethod
    def draw_camera(color="#8DA2C3", size=18) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        p.setPen(pen)
        p.drawRect(2, 5, 10, 8)
        path = QPainterPath()
        path.moveTo(12, 7)
        path.lineTo(16, 5)
        path.lineTo(16, 13)
        path.lineTo(12, 11)
        p.drawPath(path)
        p.end()
        return pix

    @staticmethod
    def draw_settings(color="#8DA2C3", size=18) -> QPixmap:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color), 1.8)
        p.setPen(pen)
        p.drawEllipse(5, 5, 8, 8)
        p.drawLine(9, 1, 9, 4)
        p.drawLine(9, 14, 9, 17)
        p.drawLine(1, 9, 4, 9)
        p.drawLine(14, 9, 17, 9)
        p.end()
        return pix


class CircularProgressWidget(QWidget):
    """Widget vectoriel personnalisé pour afficher l'anneau de similarité animé."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(110, 110)
        self._progress = 0.0
        self._status_color = QColor("#18C7F5")

        self.animation = QVariantAnimation(self)
        self.animation.setDuration(800)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.valueChanged.connect(self._on_anim_value_changed)

    def set_score(self, target_score: float, is_found: bool = True):
        self._status_color = QColor("#20D67B") if is_found else QColor("#FF5A6F")
        self.animation.stop()
        self.animation.setStartValue(self._progress)
        self.animation.setEndValue(target_score)
        self.animation.start()

    def _on_anim_value_changed(self, value):
        self._progress = float(value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(8, 8, 94, 94)

        # Fond de piste neutre
        pen_bg = QPen(QColor(255, 255, 255, 15), 6)
        painter.setPen(pen_bg)
        painter.drawEllipse(rect)

        # Arc actif animé
        if self._progress > 0:
            pen_active = QPen(self._status_color, 6)
            pen_active.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_active)
            angle = int(-self._progress / 100.0 * 360 * 16)
            painter.drawArc(rect, 90 * 16, angle)

        # Texte central
        painter.setPen(QColor("#FFFFFF"))
        font_val = QFont("Inter", 13, QFont.Bold)
        painter.setFont(font_val)
        val_text = f"{self._progress:.1f}%"
        painter.drawText(QRectF(0, 32, 110, 24), Qt.AlignCenter, val_text)

        painter.setPen(QColor("#8DA2C3"))
        font_sub = QFont("Inter", 7, QFont.Bold)
        painter.setFont(font_sub)
        painter.drawText(QRectF(0, 58, 110, 16), Qt.AlignCenter, "SIMILARITÉ")
        painter.end()


class CameraWorker(QThread):
    """Worker Thread pour la capture vidéo webcam et le traitement en temps réel avec effet miroir."""
    frame_processed = Signal(np.ndarray, dict)

    def __init__(self, countdown_duration: float = 5.0):
        super().__init__()
        self.running = False
        self.pipeline = CameraPipeline(countdown_duration=countdown_duration)
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True

        while self.running:
            success, frame = self.cap.read()
            if not success or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA INDISPONIBLE - MODE SYSTEME SVA", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (141, 162, 195), 2)
                time.sleep(0.05)
            else:
                frame = cv2.flip(frame, 1)

            processed_frame = self.pipeline.process_frame(frame)
            status_data = self.pipeline.get_status()

            self.frame_processed.emit(processed_frame, status_data)
            time.sleep(0.03)

        if self.cap and self.cap.isOpened():
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def trigger_manual_search(self, current_thresh: float = 0.75) -> Optional[Dict[str, Any]]:
        if self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if success and frame is not None:
                frame = cv2.flip(frame, 1)
                cap_filename = self.pipeline.capture_mgr.capture(frame)
                self.pipeline.last_captured_image = cap_filename
                res = search_face(
                    frame,
                    top_k=5,
                    threshold=current_thresh,
                    extractor=self.pipeline.extractor,
                    db=self.pipeline.db
                )
                self.pipeline.last_result = res
                self.pipeline.result_display_timer = time.time()
                return res
        return None


class BioVectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVA - Secure Vision AI Platform")
        self.resize(1420, 900)
        self.setMinimumSize(1200, 780)

        self.camera_thread = CameraWorker(countdown_duration=5.0)
        self.camera_thread.frame_processed.connect(self.on_frame_processed)

        self.last_known_captured_path = None
        self.full_attendance_data = []

        self.init_ui()
        self.apply_sva_enterprise_theme()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

        self.camera_thread.start()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # =========================================================================
        # 1. SIDEBAR NAVIGATION ENTERPRISE SVA
        # =========================================================================
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("sidebarFrame")
        sidebar_frame.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(18, 22, 18, 22)
        sidebar_layout.setSpacing(14)

        # Header Logo SVA
        logo_layout = QHBoxLayout()
        logo_badge = QLabel("SVA")
        logo_badge.setObjectName("logoBadgeSVA")

        logo_text_box = QVBoxLayout()
        logo_title = QLabel("Secure Vision AI")
        logo_title.setObjectName("logoTitle")
        logo_sub = QLabel("Enterprise Platform")
        logo_sub.setObjectName("logoSub")
        logo_text_box.addWidget(logo_title)
        logo_text_box.addWidget(logo_sub)

        logo_layout.addWidget(logo_badge)
        logo_layout.addLayout(logo_text_box)
        logo_layout.addStretch()

        sidebar_layout.addLayout(logo_layout)
        sidebar_layout.addSpacing(10)

        # Entrées du menu principal SVA
        menu_items = [
            ("Dashboard", True, LucideIcon.draw_dashboard),
            ("Historique", False, LucideIcon.draw_history),
            ("Employés", False, LucideIcon.draw_user),
            ("Base biométrique", False, LucideIcon.draw_database),
            ("Caméras", False, LucideIcon.draw_camera),
            ("Paramètres", False, LucideIcon.draw_settings),
            ("Rapports", False, LucideIcon.draw_dashboard),
            ("Logs système", False, LucideIcon.draw_history)
        ]

        for text, is_active, draw_fn in menu_items:
            btn = QPushButton(f"  {text}")
            btn.setIcon(QIcon(draw_fn("#18C7F5" if is_active else "#8DA2C3", 16)))
            btn.setIconSize(QtCore.QSize(16, 16))
            btn.setObjectName("navBtnActive" if is_active else "navBtn")
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # Bloc Système
        sys_box = QFrame()
        sys_box.setObjectName("sysBox")
        sys_layout = QVBoxLayout(sys_box)
        sys_layout.setSpacing(6)

        sys_title = QLabel("SYSTEM STATUS")
        sys_title.setObjectName("sysTitle")
        sys_layout.addWidget(sys_title)

        sys_rows = [
            ("Statut serveur", "🟢 En ligne"),
            ("PostgreSQL", "🟢 Connecté"),
            ("pgvector", "🟢 512D Active"),
            ("Caméra", "Logitech C920")
        ]

        for label, val in sys_rows:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setObjectName("sysLbl")
            v = QLabel(val)
            v.setObjectName("sysVal")
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(v)
            sys_layout.addLayout(row)

        sidebar_layout.addWidget(sys_box)

        # Carte Administrateur
        user_box = QFrame()
        user_box.setObjectName("userBox")
        user_layout = QHBoxLayout(user_box)
        user_layout.setContentsMargins(10, 8, 10, 8)

        u_avatar = QLabel()
        u_avatar.setPixmap(LucideIcon.draw_user("#18C7F5", 22))

        u_info = QVBoxLayout()
        u_name = QLabel("Admin")
        u_name.setObjectName("uName")
        u_role = QLabel("Administrateur")
        u_role.setObjectName("uRole")
        u_info.addWidget(u_name)
        u_info.addWidget(u_role)

        user_layout.addWidget(u_avatar)
        user_layout.addLayout(u_info)
        user_layout.addStretch()

        sidebar_layout.addWidget(user_box)
        root_layout.addWidget(sidebar_frame)

        # =========================================================================
        # 2. PANNEAU PRINCIPAL SVA
        # =========================================================================
        main_content = QFrame()
        main_content.setObjectName("mainContent")
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(24, 18, 24, 20)
        main_layout.setSpacing(16)

        # Header Bar SVA
        header_bar = QHBoxLayout()

        head_title_box = QVBoxLayout()
        head_title = QLabel("SVA — Secure Vision AI")
        head_title.setObjectName("headerTitle")
        head_desc = QLabel("Enterprise Biometric Identification Platform")
        head_desc.setObjectName("headerDesc")
        head_title_box.addWidget(head_title)
        head_title_box.addWidget(head_desc)

        header_bar.addLayout(head_title_box)
        header_bar.addStretch()

        # Badge d'état + Horloge + Contrôles
        status_pill = QLabel("🟢 Système opérationnel")
        status_pill.setObjectName("sysStatusPill")
        header_bar.addWidget(status_pill)
        header_bar.addSpacing(15)

        self.clock_time_lbl = QLabel("17:02:45")
        self.clock_time_lbl.setObjectName("clockTime")
        self.clock_date_lbl = QLabel("20 Mai 2025")
        self.clock_date_lbl.setObjectName("clockDate")

        clock_box = QVBoxLayout()
        clock_box.addWidget(self.clock_time_lbl, alignment=Qt.AlignRight)
        clock_box.addWidget(self.clock_date_lbl, alignment=Qt.AlignRight)

        header_bar.addLayout(clock_box)
        header_bar.addSpacing(15)

        btn_notif = QPushButton("🔔")
        btn_notif.setObjectName("btnHeaderIcon")
        btn_fullscreen = QPushButton("⤢")
        btn_fullscreen.setObjectName("btnHeaderIcon")
        btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        header_bar.addWidget(btn_notif)
        header_bar.addWidget(btn_fullscreen)

        main_layout.addLayout(header_bar)

        # Grille à 3 Panneaux Bento
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        # -------------------------------------------------------------
        # PANNEAU 1 : FLUX VIDÉO & CAPTURE
        # -------------------------------------------------------------
        card1 = QFrame()
        card1.setObjectName("bentoCard")
        card1_layout = QVBoxLayout(card1)
        card1_layout.setSpacing(12)

        c1_head = QHBoxLayout()
        c1_title = QLabel("Flux Vidéo & Capture")
        c1_title.setObjectName("cardTitle")
        c1_live = QLabel("● EN DIRECT")
        c1_live.setObjectName("liveBadge")
        c1_head.addWidget(c1_title)
        c1_head.addStretch()
        c1_head.addWidget(c1_live)
        card1_layout.addLayout(c1_head)

        # Video Box avec overlay
        self.video_label = QLabel()
        self.video_label.setObjectName("videoBox")
        self.video_label.setMinimumSize(350, 240)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Initialisation du flux caméra SVA...")
        card1_layout.addWidget(self.video_label, stretch=1)

        # Boutons d'Action Principaux
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_capture = QPushButton("Capturer")
        self.btn_capture.setObjectName("btnPrimaryCyan")
        self.btn_capture.clicked.connect(self.trigger_manual_search)

        self.btn_select = QPushButton("Sélectionner une image")
        self.btn_select.setObjectName("btnOutlineDark")
        self.btn_select.clicked.connect(self.open_file_and_search)

        self.btn_stop_cam = QPushButton("Arrêter la caméra")
        self.btn_stop_cam.setObjectName("btnOutlineRed")
        self.btn_stop_cam.clicked.connect(self.toggle_camera)

        btn_row.addWidget(self.btn_capture)
        btn_row.addWidget(self.btn_select)
        btn_row.addWidget(self.btn_stop_cam)
        card1_layout.addLayout(btn_row)

        # Paramètres Card
        param_box = QFrame()
        param_box.setObjectName("innerParamBox")
        param_layout = QVBoxLayout(param_box)
        param_layout.setSpacing(8)

        param_title = QLabel("PARAMÈTRES DE RECHERCHE")
        param_title.setObjectName("paramSubTitle")
        param_layout.addWidget(param_title)

        # Slider Seuil
        sl_head = QHBoxLayout()
        sl_head.addWidget(QLabel("Seuil de similarité"))
        sl_head.addStretch()
        self.lbl_thresh_val = QLabel("75.00%")
        self.lbl_thresh_val.setObjectName("threshPill")
        sl_head.addWidget(self.lbl_thresh_val)
        param_layout.addLayout(sl_head)

        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(50, 100)
        self.slider_thresh.setValue(75)
        self.slider_thresh.valueChanged.connect(self.on_slider_changed)
        param_layout.addWidget(self.slider_thresh)

        # Options et Switches
        sw_row1 = QHBoxLayout()
        self.chk_auto = QCheckBox("Recherche automatique")
        self.chk_auto.setChecked(True)
        self.chk_cont = QCheckBox("Capture continue")
        sw_row1.addWidget(self.chk_auto)
        sw_row1.addWidget(self.chk_cont)
        param_layout.addLayout(sw_row1)

        sw_row2 = QHBoxLayout()
        self.chk_multi = QCheckBox("Détection multi-visages")
        self.chk_multi.setChecked(True)
        sw_row2.addWidget(self.chk_multi)
        sw_row2.addStretch()
        param_layout.addLayout(sw_row2)

        # Métriques secondaires
        m_row = QHBoxLayout()
        m_box1 = QVBoxLayout()
        m_box1.addWidget(QLabel("Temps d'exécution estimé"))
        self.lbl_execution_time = QLabel("⏱ 0.08 s")
        self.lbl_execution_time.setObjectName("valMetrics")
        m_box1.addWidget(self.lbl_execution_time)

        m_box2 = QVBoxLayout()
        m_box2.addWidget(QLabel("Méthode de comparaison"))
        m_combo = QComboBox()
        m_combo.addItem("Cosinus Vectoriel (pgvector)")
        m_combo.addItem("Distance Euclidienne")
        m_box2.addWidget(m_combo)

        m_row.addLayout(m_box1)
        m_row.addStretch()
        m_row.addLayout(m_box2)
        param_layout.addLayout(m_row)

        card1_layout.addWidget(param_box)
        cards_layout.addWidget(card1, stretch=3)

        # -------------------------------------------------------------
        # PANNEAU 2 : RÉSULTAT DE L'IDENTIFICATION
        # -------------------------------------------------------------
        card2 = QFrame()
        card2.setObjectName("bentoCard")
        card2_layout = QVBoxLayout(card2)
        card2_layout.setSpacing(12)

        c2_title = QLabel("Résultat de l'identification")
        c2_title.setObjectName("cardTitle")
        card2_layout.addWidget(c2_title)

        # Comparaison côte à côte + Anneau Circulaire Animé central
        comp_layout = QHBoxLayout()
        comp_layout.setSpacing(6)

        # Photo Capturée
        cap_box = QVBoxLayout()
        cap_box.addWidget(QLabel("IMAGE CAPTURÉE"), alignment=Qt.AlignCenter)
        self.cap_img_label = QLabel()
        self.cap_img_label.setObjectName("photoFrame")
        self.cap_img_label.setFixedSize(115, 145)
        self.cap_img_label.setAlignment(Qt.AlignCenter)
        self.cap_img_label.setText("Aucune")
        cap_box.addWidget(self.cap_img_label)

        # Anneau de score central vectoriel animé
        self.circle_progress = CircularProgressWidget()
        comp_layout.addLayout(cap_box)
        comp_layout.addWidget(self.circle_progress, alignment=Qt.AlignCenter)

        # Photo Matchée DB
        db_box = QVBoxLayout()
        db_box.addWidget(QLabel("MEILLEURE CORRESPONDANCE"), alignment=Qt.AlignCenter)
        self.db_img_label = QLabel()
        self.db_img_label.setObjectName("photoFrameMatched")
        self.db_img_label.setFixedSize(115, 145)
        self.db_img_label.setAlignment(Qt.AlignCenter)
        self.db_img_label.setText("Aucun match")
        db_box.addWidget(self.db_img_label)

        comp_layout.addLayout(db_box)
        card2_layout.addLayout(comp_layout)

        # Bannière de résultat
        self.lbl_banner = QLabel("✔ Correspondance trouvée")
        self.lbl_banner.setObjectName("bannerFound")
        self.lbl_banner.setAlignment(Qt.AlignCenter)
        card2_layout.addWidget(self.lbl_banner)

        # Liste des détails employé avec icônes Lucide
        details_layout = QVBoxLayout()
        details_layout.setSpacing(8)

        fields = [
            (LucideIcon.draw_user, "Nom complet", "lbl_full_name", "---"),
            (LucideIcon.draw_database, "Matricule", "lbl_matricule", "---"),
            (LucideIcon.draw_dashboard, "Département", "lbl_dept", "---"),
            (LucideIcon.draw_settings, "Poste", "lbl_poste", "---"),
            (LucideIcon.draw_history, "Email", "lbl_email", "---"),
            (LucideIcon.draw_camera, "Statut d'accès", "lbl_access_status", "EN ATTENTE")
        ]

        for draw_fn, label_title, attr_name, default_val in fields:
            row = QHBoxLayout()
            icon_lbl = QLabel()
            icon_lbl.setPixmap(draw_fn("#18C7F5", 16))

            text_box = QVBoxLayout()
            text_box.setSpacing(0)
            lbl_t = QLabel(label_title)
            lbl_t.setObjectName("sysLbl")
            lbl_v = QLabel(default_val)
            lbl_v.setObjectName("valDetail")
            setattr(self, attr_name, lbl_v)

            text_box.addWidget(lbl_t)
            text_box.addWidget(lbl_v)

            row.addWidget(icon_lbl)
            row.addLayout(text_box)
            row.addStretch()
            details_layout.addLayout(row)

        card2_layout.addLayout(details_layout)
        card2_layout.addStretch()
        cards_layout.addWidget(card2, stretch=3)

        # -------------------------------------------------------------
        # PANNEAU 3 : HISTORIQUE DES RECHERCHES
        # -------------------------------------------------------------
        card3 = QFrame()
        card3.setObjectName("bentoCard")
        card3_layout = QVBoxLayout(card3)
        card3_layout.setSpacing(12)

        c3_title = QLabel("Historique des recherches")
        c3_title.setObjectName("cardTitle")
        card3_layout.addWidget(c3_title)

        # Barre de recherche & Filtres
        s_bar = QHBoxLayout()
        self.input_search = QLineEdit()
        self.input_search.setPlaceholderText("Rechercher un employé...")
        self.input_search.setObjectName("searchInput")
        self.input_search.textChanged.connect(self.filter_table_history)

        btn_filter = QPushButton("🌪")
        btn_filter.setObjectName("btnSquareDark")

        s_bar.addWidget(self.input_search)
        s_bar.addWidget(btn_filter)
        card3_layout.addLayout(s_bar)

        # Tableau
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Heure", "Employé", "Similarité", "Statut"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setObjectName("svaTable")
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        card3_layout.addWidget(self.table, stretch=1)

        # Pagination
        pag_layout = QHBoxLayout()
        pag_layout.addStretch()
        for p in ["<", "1", "2", "3", "4", "5", ">"]:
            btn_p = QPushButton(p)
            btn_p.setObjectName("pagActive" if p == "1" else "pagBtn")
            btn_p.setFixedSize(26, 26)
            pag_layout.addWidget(btn_p)
        pag_layout.addStretch()
        card3_layout.addLayout(pag_layout)

        cards_layout.addWidget(card3, stretch=4)

        main_layout.addLayout(cards_layout)
        root_layout.addWidget(main_content, stretch=1)

    def update_clock(self):
        now = QDateTime.currentDateTime()
        self.clock_time_lbl.setText(now.toString("hh:mm:ss"))
        self.clock_date_lbl.setText(now.toString("dd MMM yyyy"))

    def on_slider_changed(self, val: int):
        self.lbl_thresh_val.setText(f"{val:.2f}%")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @Slot(np.ndarray, dict)
    def on_frame_processed(self, frame: np.ndarray, status: dict):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        captured_path = status.get("last_captured_image")
        if captured_path and captured_path != self.last_known_captured_path:
            self.last_known_captured_path = captured_path
            self.update_image_preview(self.cap_img_label, captured_path)

        last_result = status.get("last_result")
        if last_result:
            self.display_search_result(last_result)

        attendance_list = status.get("attendance_history", [])
        if attendance_list:
            self.full_attendance_data = attendance_list
            self.update_history_table(attendance_list)

    def display_search_result(self, res: dict):
        elapsed = res.get("elapsed_seconds", 0.08)
        self.lbl_execution_time.setText(f"⏱ {elapsed:.2f} s")

        if res.get("status") == "found":
            bm = res["best_match"]
            full_name = f"{bm.get('first_name', '')} {bm.get('last_name', '')}".strip()
            self.lbl_full_name.setText(full_name if full_name else "Inconnu")
            self.lbl_matricule.setText(str(bm.get("employee_number", "EMP-001")))
            self.lbl_dept.setText(bm.get("department", "Général"))
            self.lbl_poste.setText(bm.get("position", "Employé"))
            self.lbl_email.setText(bm.get("email", f"{full_name.lower().replace(' ', '.')}@example.com"))

            sim_score = bm.get("similarity_percent", 0.0)
            self.circle_progress.set_score(sim_score, is_found=True)

            self.lbl_banner.setText("✔ Correspondance trouvée")
            self.lbl_banner.setStyleSheet("background-color: rgba(32, 214, 123, 0.15); color: #20D67B; border: 1px solid #20D67B; border-radius: 8px; padding: 8px;")

            self.lbl_access_status.setText("ACCÈS AUTORISÉ")
            self.lbl_access_status.setStyleSheet("color: #20D67B; font-weight: bold;")

            if bm.get("image_path"):
                self.update_image_preview(self.db_img_label, bm["image_path"])

        elif res.get("status") in ["unknown_person", "no_face_detected"]:
            self.lbl_full_name.setText("Personne Inconnue")
            self.lbl_matricule.setText("---")
            self.lbl_dept.setText("---")
            self.lbl_poste.setText("---")
            self.lbl_email.setText("---")

            self.circle_progress.set_score(0.0, is_found=False)

            self.lbl_banner.setText("✖ Aucune correspondance")
            self.lbl_banner.setStyleSheet("background-color: rgba(255, 90, 111, 0.15); color: #FF5A6F; border: 1px solid #FF5A6F; border-radius: 8px; padding: 8px;")

            self.lbl_access_status.setText("ACCÈS REFUSÉ")
            self.lbl_access_status.setStyleSheet("color: #FF5A6F; font-weight: bold;")

    def update_image_preview(self, label: QLabel, img_path: str):
        if not os.path.exists(img_path):
            cand1 = os.path.abspath(os.path.join(current_dir, img_path))
            parent_dir = os.path.dirname(current_dir)
            cand2 = os.path.abspath(os.path.join(parent_dir, img_path))
            if os.path.exists(cand1):
                img_path = cand1
            elif os.path.exists(cand2):
                img_path = cand2

        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

    def update_history_table(self, history_list: list):
        self.table.setRowCount(0)
        filter_text = self.input_search.text().lower()

        for item in history_list:
            full_name = f"{item.get('first_name', '')} {item.get('last_name', '')}"
            if filter_text and filter_text not in full_name.lower():
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(item.get("timestamp", "")))
            self.table.setItem(row, 1, QTableWidgetItem(full_name))
            self.table.setItem(row, 2, QTableWidgetItem(f"{item.get('similarity', 0):.2f}%"))

            status = item.get("status", "APPROVED")
            status_item = QTableWidgetItem("APPROUVÉ" if "APPROVED" in status else "REJETÉ")
            status_item.setForeground(QColor("#20D67B" if "APPROVED" in status else "#FF5A6F"))
            self.table.setItem(row, 3, status_item)

    def filter_table_history(self):
        self.update_history_table(self.full_attendance_data)

    def trigger_manual_search(self):
        current_thresh = float(self.slider_thresh.value()) / 100.0
        self.btn_capture.setText("Analyse...")
        QApplication.processEvents()
        res = self.camera_thread.trigger_manual_search(current_thresh=current_thresh)
        self.btn_capture.setText("Capturer")
        if res:
            self.display_search_result(res)

    def open_file_and_search(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une Image pour la recherche vectorielle SVA",
            current_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not file_path:
            return

        self.update_image_preview(self.cap_img_label, file_path)
        current_thresh = float(self.slider_thresh.value()) / 100.0

        res = search_face(
            image_input=file_path,
            top_k=5,
            threshold=current_thresh,
            extractor=self.camera_thread.pipeline.extractor,
            db=self.camera_thread.pipeline.db
        )

        self.display_search_result(res)

        if res.get("status") == "found":
            bm = res["best_match"]
            new_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "first_name": bm.get("first_name", ""),
                "last_name": bm.get("last_name", ""),
                "department": bm.get("department", "Général"),
                "position": bm.get("position", "Employé"),
                "similarity": bm.get("similarity_percent", 0.0),
                "status": "APPROVED (FICHIER)",
                "image_path": file_path
            }
            self.camera_thread.pipeline.attendance_history.insert(0, new_entry)
            self.camera_thread.pipeline.db.log_search_event(new_entry)
            self.update_history_table(self.camera_thread.pipeline.attendance_history)
        elif res.get("status") in ["unknown_person", "no_face_detected"]:
            new_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "first_name": "Personne",
                "last_name": "Inconnue",
                "department": "---",
                "position": "Accès Refusé",
                "similarity": 0.0,
                "status": "REJECTED (FICHIER)",
                "image_path": file_path
            }
            self.camera_thread.pipeline.attendance_history.insert(0, new_entry)
            self.camera_thread.pipeline.db.log_search_event(new_entry)
            self.update_history_table(self.camera_thread.pipeline.attendance_history)
            msg = res.get("message", "Aucun visage correspondant trouvé.")
            QMessageBox.information(self, "Résultat SVA", msg)

    def toggle_camera(self):
        if self.camera_thread.running:
            self.camera_thread.stop()
            self.btn_stop_cam.setText("Démarrer la caméra")
            self.video_label.setText("Caméra suspendue.")
        else:
            self.camera_thread.start()
            self.btn_stop_cam.setText("Arrêter la caméra")

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

    def apply_sva_enterprise_theme(self):
        """Charte Graphique Officielle Enterprise SVA (Secure Vision AI)."""
        qss = """
        /* Base globale */
        QMainWindow, QWidget#centralWidget {
            background-color: #08111F;
            color: #FFFFFF;
            font-family: "Segoe UI", "Inter", -apple-system, sans-serif;
        }

        /* Sidebar Nav */
        QFrame#sidebarFrame {
            background-color: #101827;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        #logoBadgeSVA {
            background-color: #18C7F5;
            color: #08111F;
            border-radius: 8px;
            padding: 4px 8px;
            font-weight: bold;
            font-size: 14px;
        }
        #logoTitle {
            color: #FFFFFF;
            font-size: 15px;
            font-weight: bold;
        }
        #logoSub {
            color: #8DA2C3;
            font-size: 10px;
        }

        QPushButton#navBtn {
            background-color: transparent;
            color: #8DA2C3;
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            text-align: left;
            font-size: 13px;
        }
        QPushButton#navBtn:hover {
            background-color: rgba(255, 255, 255, 0.04);
            color: #FFFFFF;
        }
        QPushButton#navBtnActive {
            background-color: #111C2D;
            color: #18C7F5;
            border-left: 3px solid #18C7F5;
            border-radius: 6px;
            padding: 10px 14px;
            text-align: left;
            font-size: 13px;
            font-weight: bold;
        }

        /* Boîtes d'Information Système et Profil Sidebar */
        QFrame#sysBox, QFrame#userBox {
            background-color: #111C2D;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 10px;
        }
        #sysTitle {
            color: #8DA2C3;
            font-size: 9px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        #sysLbl {
            color: #8DA2C3;
            font-size: 11px;
        }
        #sysVal {
            color: #FFFFFF;
            font-size: 11px;
        }
        #uName {
            color: #FFFFFF;
            font-size: 12px;
            font-weight: bold;
        }
        #uRole {
            color: #8DA2C3;
            font-size: 10px;
        }

        /* Top Header */
        #headerTitle {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: bold;
        }
        #headerDesc {
            color: #8DA2C3;
            font-size: 11px;
        }
        #sysStatusPill {
            background-color: rgba(32, 214, 123, 0.12);
            color: #20D67B;
            border: 1px solid #20D67B;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: bold;
        }
        #clockTime {
            color: #FFFFFF;
            font-size: 14px;
            font-weight: bold;
        }
        #clockDate {
            color: #8DA2C3;
            font-size: 11px;
        }
        QPushButton#btnHeaderIcon {
            background-color: #111C2D;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: #FFFFFF;
            padding: 6px 10px;
        }

        /* Bento Cards (Flat Style SVA) */
        QFrame#bentoCard {
            background-color: #111C2D;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 16px;
        }
        #cardTitle {
            color: #FFFFFF;
            font-size: 15px;
            font-weight: bold;
        }
        #liveBadge {
            background-color: rgba(32, 214, 123, 0.15);
            color: #20D67B;
            border: 1px solid #20D67B;
            border-radius: 12px;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: bold;
        }

        /* Video Feed Box */
        #videoBox {
            background-color: #08111F;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
        }

        /* Boutons d'Action */
        QPushButton#btnPrimaryCyan {
            background-color: #18C7F5;
            color: #08111F;
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton#btnPrimaryCyan:hover {
            background-color: #22D3A6;
        }
        QPushButton#btnOutlineDark {
            background-color: transparent;
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 12px;
        }
        QPushButton#btnOutlineDark:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
        QPushButton#btnOutlineRed {
            background-color: transparent;
            color: #FF5A6F;
            border: 1px solid #FF5A6F;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 12px;
        }
        QPushButton#btnOutlineRed:hover {
            background-color: rgba(255, 90, 111, 0.15);
        }

        /* Inner Param Box */
        QFrame#innerParamBox {
            background-color: #101827;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
        }
        #paramSubTitle {
            color: #8DA2C3;
            font-size: 9px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        #threshPill {
            background-color: rgba(24, 199, 245, 0.15);
            color: #18C7F5;
            border: 1px solid #18C7F5;
            border-radius: 6px;
            padding: 2px 6px;
            font-size: 11px;
            font-weight: bold;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background-color: #18C7F5;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background-color: #18C7F5;
            width: 14px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 7px;
        }

        QCheckBox {
            color: #8DA2C3;
            font-size: 11px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            background-color: #18C7F5;
            border-color: #18C7F5;
        }

        QComboBox {
            background-color: #111C2D;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            color: #FFFFFF;
            padding: 4px 8px;
            font-size: 11px;
        }
        #valMetrics {
            color: #FFFFFF;
            font-size: 12px;
            font-weight: bold;
        }

        /* Photo Frames */
        #photoFrame, #photoFrameMatched {
            background-color: #08111F;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            color: #8DA2C3;
        }
        #photoFrameMatched {
            border: 1px solid #20D67B;
        }
        #valDetail {
            color: #FFFFFF;
            font-size: 13px;
            font-weight: bold;
        }

        /* Table SVA */
        QLineEdit#searchInput {
            background-color: #101827;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: #FFFFFF;
            padding: 6px 12px;
            font-size: 12px;
        }
        QPushButton#btnSquareDark {
            background-color: #101827;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: #8DA2C3;
            padding: 6px 10px;
        }
        QTableWidget#svaTable {
            background-color: #101827;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            gridline-color: rgba(255, 255, 255, 0.03);
            color: #FFFFFF;
            font-size: 12px;
        }
        QHeaderView::section {
            background-color: #111C2D;
            color: #8DA2C3;
            padding: 8px;
            border: none;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-weight: bold;
            font-size: 10px;
            letter-spacing: 0.5px;
        }
        QPushButton#pagBtn {
            background-color: transparent;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            color: #8DA2C3;
            font-size: 11px;
        }
        QPushButton#pagActive {
            background-color: #18C7F5;
            border: none;
            border-radius: 6px;
            color: #08111F;
            font-size: 11px;
            font-weight: bold;
        }
        """
        self.setStyleSheet(qss)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BioVectorApp()
    window.show()
    sys.exit(app.exec())
