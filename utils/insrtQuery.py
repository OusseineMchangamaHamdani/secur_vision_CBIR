from .dataStruct import UserData
# pyrefly: ignore [missing-import]
import asyncio
# pyrefly: ignore [missing-import]
from .db import get_connection


async def insertUser(user: UserData):
    try:
        conn = await get_connection()
        await conn.execute(
            "INSERT INTO users (nom, prenom, path_image, face_image, embedding) VALUES ($1, $2, $3, $4, $5)",
            (user.nom, user.prenom, user.path_image, user.face_image, user.embedding)
        )
        print(f"User {user.nom} {user.prenom} inserted successfully")
    except Exception as e:
        print(f"Error inserting user {user.nom} {user.prenom}: {e}")