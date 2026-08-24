from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from datetime import datetime


# Modèle Pydantic pour la structure d'un utilisateur / empreinte faciale
class UserData(BaseModel):
    nom: str = Field(..., description="Nom de famille")
    prenom: str = Field(..., description="Prénom")
    face_image: str = Field(..., description="Nom ou identifiant de l'image faciale")
    path_image: str = Field(..., description="Chemin d'accès au fichier image")
    embedding: Optional[List[float]] = Field(default=None, description="Vecteur d'embedding facial (ex: pgvector)")
    created_at: datetime = Field(default_factory=datetime.now, description="Date d'enregistrement")


# Modèle pour la recherche par similarité vectorielle
class SearchQuery(BaseModel):
    query_embedding: List[float] = Field(..., description="Vecteur d'embedding à comparer")
    top_k: int = Field(default=5, ge=1, description="Nombre maximum de résultats")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Seuil de similarité minimale")


# Données brutes (dictionnaires avec virgules corrigées)
data1 = {
    "nom": "Ousseine",
    "prenom": "Mchangama",
    "face_image": "image1",
    "path_image": "captures/Ousseine/2026-07-10_16-28-31.jpg",
    "embedding": np.random.rand(1536).tolist(),
}

data2 = {
    "nom": "ELAnzaff",
    "prenom": "Abdou Ali",
    "face_image": "image2",
    "path_image": "captures/Anzaff/100202657.jpeg",
    "embedding": np.random.rand(1536).tolist(),
}

# Instanciation et validation avec Pydantic
user1 = UserData(**data1)
user2 = UserData(**data2)
# Liste des utilisateurs structurés
users: List[UserData] = [user1, user2]


from utils.index import index_user

if __name__ == "__main__":
    print("--- Données Utilisateurs Structurées ---")
    for u in users:
        print(u.model_dump_json(indent=2))

    print("\n--- Démonstration de la Phase d'Indexation ---")
    # Indexation d'un utilisateur exemple
    result = index_user(
        nom=user1.nom,
        prenom=user1.prenom,
        path_image=user1.path_image,
        employee_number="EMP_001",
        department="Informatique",
        position="Développeur"
    )
    print("Statut de l'indexation :", result.get("status"))

