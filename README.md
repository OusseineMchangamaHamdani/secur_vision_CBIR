## 🛠️ Technologies

- **Langage :** Python
- **Framework Web :** Flask
- **Interface Desktop :** PySide6
- **Deep Learning :** FaceNet / MTCNN
- **Base de données :** PostgreSQL
- **Recherche vectorielle :** pgvector
- **Indexation :** HNSW
- **Conteneurisation :** Docker

# BioVector - Système de Pointage Facial par Recherche d'Images par le Contenu (CBIR)

## 📌 Présentation Générale

**BioVector** est un système moderne de pointage biométrique basé sur le principe du **CBIR (Content-Based Image Retrieval)** ou *Recherche d'Images par le Contenu*. 

Contrairement aux systèmes traditionnels fondés sur des requêtes textuelles ou des identifiants (badges, identifiants employés), BioVector utilise l'image faciale capturée en temps réel (via webcam ou flux vidéo) comme requête. Le système extrait l'empreinte visuelle haute dimension (embedding faciale) du visage détecté, puis effectue une recherche de plus proches voisins (*Nearest Neighbor Search*) au sein d'une base de données PostgreSQL accélérée par l'extension vectorielle **pgvector**.

---

## 🏗️ Architecture et Pipeline CBIR

Le système repose sur un pipeline CBIR en 4 étapes majeures :

```mermaid
flowchart LR
    A[📷 Acquisition Image / Flux Vidéo] --> B[🔍 Détection & Rognage Facia (MTCNN)]
    B --> C[🧠 Extraction de Descripteurs (FaceNet 512D)]
    C --> D[📐 Normalisation L2 du Vecteur]
    D --> E[⚡ Recherche Vectorielle pgvector (HNSW / Cosine)]
    E --> F[🎯 Identification & Calcul du Score de Similarité]
```

### 1. Acquisition et Prétraitement Visuel
- **Entrée** : Image fixe (`.jpg`, `.png`) ou flux vidéo en direct (Webcam OpenCV).
- **Détection Faciale** : Utilisation du réseau neuronal multi-tâches **MTCNN** (*Multi-task Cascaded Convolutional Networks*) pour localiser le visage avec précision et en extraire la boîte englobante (*bounding box*).
- **Rognage Dynamique** : Extraction de la région d'intérêt (*Crop*) du visage.

### 2. Extraction de Descripteurs (Feature Extraction)
- **Modèle Deep Learning** : **Keras-FaceNet** (architecture Inception-ResNet-v1 entraînée sur de larges jeux de données de visages).
- **Représentation Vectorielle** : Transformation de la sous-image faciale en un vecteur dense en 512 dimensions :
  $$\mathbf{v} \in \mathbb{R}^{512}$$
- **Normalisation L2** : Conversion de l'embedding en un vecteur unitaire pour rendre la distance euclidienne équivalente à la similarité cosinus :
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### 3. Stockage et Indexation Vectorielle (pgvector)
- **Base de Données** : PostgreSQL 16 avec l'extension `pgvector`.
- **Schéma SQL** (`iniDB.sql`) :
  - Table `users` : Métadonnées de l'employé (nom, prénom, email, département, etc.).
  - Table `user_embeddings` : Contient le vecteur `VECTOR(512)` et le chemin de l'image de référence.
- **Indexation HNSW** (*Hierarchical Navigable Small World*) :
  - Utilisation de l'opérateur `vector_cosine_ops`.
  - Permet des recherches ultra-rapides en complexité logarithmique $\mathcal{O}(\log N)$, idéales pour les bases de données volumineuses.

### 4. Recherche et Appariement (Matching Engine)
- **Requête Cosinus** : Mesure de la distance cosinus entre l'embedding de requête $\mathbf{q}$ et les embeddings enregistrés $\mathbf{d}_i$ :
  $$d_{\text{cosine}}(\mathbf{q}, \mathbf{d}_i) = 1 - (\mathbf{q} \cdot \mathbf{d}_i)$$
- **Score de Similarité** : Conversion en pourcentage de similarité :
  $$\text{Similarité (\%)} = (1 - d_{\text{cosine}}) \times 100$$
- **Validation par Seuil (Thresholding)** : Vérification de la correspondance (seuil configurable, ex. $\ge 75\%$). Si le score dépasse le seuil, l'identité est validée ; sinon, la personne est marquée comme non reconnue.

---

## 🛠️ Structure des Modules du Projet

| Fichier / Dossier | Rôle et Description |
| :--- | :--- |
| [`app.py`](app.py) | Point d'entrée principal. Gère l'application Web Flask (mode SaaS) et fait le relais vers l'application Desktop Qt PySide6. |
| [`gui_app.py`](gui_app.py) | Application Desktop complète sous **PySide6 (Qt pour Python)** avec tableau de bord, flux caméra en direct, affichage HUD et enregistrement d'utilisateurs. |
| [`utils/featureExtractor.py`](utils/featureExtractor.py) | Module d'extraction de caractéristiques visuelles (MTCNN + Keras-FaceNet + Normalisation L2). |
| [`utils/db_manager.py`](utils/db_manager.py) | Gestionnaire SQL PostgreSQL (`psycopg3`). Exécute les requêtes de recherche vectorielle avec `<=>` (distance cosinus) et la gestion des utilisateurs/historique. |
| [`utils/search.py`](utils/search.py) | Fonction `search_face` orchestrant le pipeline complet de recherche CBIR (Frame/Image -> Embedding -> pgvector -> Best Match). |
| [`utils/camera_pipeline.py`](utils/camera_pipeline.py) | Pipeline caméra temps réel avec compte à rebours, détection de présence faciale, HUD graphique et déclenchement automatique du pointage. |
| [`utils/index.py`](utils/index.py) | Module d'indexation des nouveaux profil employés (génération et insertion de l'embedding dans la base de données). |
| [`iniDB.sql`](iniDB.sql) | Script d'initialisation de la base PostgreSQL avec l'extension `vector`, les tables `users`, `user_embeddings` et l'index HNSW. |
| [`docker-compose.yml`](docker-compose.yml) | Configuration Docker avec le conteneur PostgreSQL `pgvector/pgvector:pg16` et l'application Python. |

---

## 🚀 Modes de Lancement

### 1. Démarrage de la Base de Données (Docker)
```bash
docker-compose up -d db
```

### 2. Application Desktop GUI (PySide6 / Qt)
```bash
python app.py
```

### 3. Application Web SaaS (Flask + Bento Grid)
```bash
python app.py --web
```
L'interface Web est ensuite accessible sur `http://127.0.0.1:5000`.

---

## ⚡ Points Forts et Innovation CBIR

1. **Recherche Vectorielle Haute Performance** : Grâce à l'indexation HNSW de `pgvector`, la recherche de similarité parmi des milliers d'empreintes se fait en une fraction de milliseconde.
2. **Robustesse de la Représentation Visuelle** : L'espace d'embedding 512D de FaceNet garantit une invariance aux légères variations d'éclairage, d'angle et d'expression faciale.
3. **Double Interface Utilisateur** : Une version Desktop ergonomique (PySide6) pour une borne d'émargement locale et une version Web (Flask SaaS) pour une gestion à distance.
4. **Anti-Spoofing & Contrôle Qualité** : Détection des yeux et contrôle de présence intégrés dans le pipeline de saisie pour prévenir la fraude.
