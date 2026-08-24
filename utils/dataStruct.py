from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserData(BaseModel):
    nom: str = Field(..., description="Nom de famille")
    prenom: str = Field(..., description="Prénom")
    face_image: str = Field(..., description="Nom ou identifiant de l'image faciale")
    path_image: str = Field(..., description="Chemin d'accès au fichier image")
    embedding: Optional[List[float]] = Field(default=None, description="Vecteur d'embedding facial (ex: pgvector)")
    created_at: datetime = Field(default_factory=datetime.now, description="Date d'enregistrement")

