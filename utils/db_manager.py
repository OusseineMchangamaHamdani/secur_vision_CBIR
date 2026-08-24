import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self, dbname=None, user=None, password=None, host=None, port=None):
        self.dbname = dbname or os.getenv("POSTGRES_DB", "my_database")
        self.user = user or os.getenv("POSTGRES_USER", "admin")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "securepasswordhere")
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = int(port or os.getenv("DB_PORT", 5433))

        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            client_encoding='utf8'
        )
        self.cur = self.conn.cursor()

    def get_or_create_employee(self, employee_data):
        """
        Récupère l'ID d'un utilisateur/employé existant ou le crée s'il n'existe pas encore.
        """
        first_name = employee_data.get('first_name', '')
        last_name = employee_data.get('last_name', '')
        email = employee_data.get('email', f"{first_name.lower()}.{last_name.lower()}@example.com")
        emp_num = employee_data.get('number', 'EMP001')

        try:
            self.cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'employees'
                );
            """)
            has_employees_table = self.cur.fetchone()[0]

            if has_employees_table:
                # Chercher l'employé existant
                self.cur.execute("""
                    SELECT id FROM employees WHERE email = %s OR employee_number = %s LIMIT 1;
                """, (email, emp_num))
                row = self.cur.fetchone()
                if row:
                    return row[0]

                # Créer le nouvel employé
                self.cur.execute("""
                    INSERT INTO employees (employee_number, first_name, last_name, email, department, position)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
                """, (emp_num, first_name, last_name, email, employee_data.get('department', 'General'), employee_data.get('position', 'Staff')))
                emp_id = self.cur.fetchone()[0]
                self.conn.commit()
                return emp_id
            else:
                # Chercher l'utilisateur existant
                self.cur.execute("""
                    SELECT user_id FROM users WHERE email = %s OR (first_name = %s AND last_name = %s) LIMIT 1;
                """, (email, first_name, last_name))
                row = self.cur.fetchone()
                if row:
                    return row[0]

                # Créer le nouvel utilisateur
                self.cur.execute("""
                    INSERT INTO users (first_name, last_name, email)
                    VALUES (%s, %s, %s) RETURNING user_id;
                """, (first_name, last_name, email))
                u_id = self.cur.fetchone()[0]
                self.conn.commit()
                return u_id
        except Exception as e:
            self.conn.rollback()
            print(f"[ERREUR DB get_or_create] {e}")
            return None

    def add_embedding(self, user_id, image_path, embedding, model_name="facenet", dimension=512):
        """
        Ajoute une photo et son embedding facial pour un utilisateur/employé existant.
        """
        try:
            self.cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'employees'
                );
            """)
            has_employees_table = self.cur.fetchone()[0]
            emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            if has_employees_table:
                self.cur.execute("""
                    INSERT INTO photos (employee_id, image_path)
                    VALUES (%s, %s) RETURNING id;
                """, (user_id, image_path))
                photo_id = self.cur.fetchone()[0]

                self.cur.execute("""
                    INSERT INTO face_embeddings (employee_id, photo_id, model_name, embedding, dimension)
                    VALUES (%s, %s, %s, %s, %s) RETURNING id;
                """, (user_id, photo_id, model_name, str(emb_list), dimension))
                emb_id = self.cur.fetchone()[0]
                self.conn.commit()
                return emb_id
            else:
                self.cur.execute("""
                    INSERT INTO user_embeddings (user_id, image_path, embedding)
                    VALUES (%s, %s, %s) RETURNING embedding_id;
                """, (user_id, image_path, str(emb_list)))
                emb_id = self.cur.fetchone()[0]
                self.conn.commit()
                return emb_id
        except Exception as e:
            self.conn.rollback()
            print(f"[ERREUR DB add_embedding] {e}")
            return None

    def insert_new_employee(self, employee_data, image_path, embedding, model_name="facenet", dimension=512):
        """
        Insère un nouvel employé/utilisateur (ou réutilise si existant) avec sa photo et son vecteur.
        """
        user_id = self.get_or_create_employee(employee_data)
        if user_id:
            return self.add_embedding(user_id, image_path, embedding, model_name=model_name, dimension=dimension)
        return None

    def search_nearest_face(self, query_embedding, top_k=5, threshold=0.5):
        """
        Recherche les visages les plus similaires dans PostgreSQL avec pgvector.
        Retourne une liste de dictionnaires contenant les métadonnées de l'employé et la similarité (0 à 1).
        """
        try:
            emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

            self.cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'employees'
                );
            """)
            has_employees_table = self.cur.fetchone()[0]

            if has_employees_table:
                sql = """
                    SELECT 
                        e.id AS employee_id,
                        e.first_name,
                        e.last_name,
                        e.email,
                        e.department,
                        e.position,
                        e.employee_number,
                        fe.image_path,
                        1 - (fe.embedding <=> %s::vector) AS similarity
                    FROM face_embeddings fe
                    JOIN employees e ON fe.employee_id = e.id
                    ORDER BY fe.embedding <=> %s::vector ASC
                    LIMIT %s;
                """
            else:
                sql = """
                    SELECT 
                        u.user_id AS employee_id,
                        u.first_name,
                        u.last_name,
                        u.email,
                        'General' AS department,
                        'Employé' AS position,
                        CONCAT('EMP_', u.user_id) AS employee_number,
                        ue.image_path,
                        1 - (ue.embedding <=> %s::vector) AS similarity
                    FROM user_embeddings ue
                    JOIN users u ON ue.user_id = u.user_id
                    ORDER BY ue.embedding <=> %s::vector ASC
                    LIMIT %s;
                """

            self.cur.execute(sql, (str(emb_list), str(emb_list), top_k))
            rows = self.cur.fetchall()

            results = []
            for r in rows:
                similarity = float(r[8]) if r[8] is not None else 0.0
                if similarity >= threshold:
                    results.append({
                        "employee_id": r[0],
                        "first_name": r[1],
                        "last_name": r[2],
                        "email": r[3],
                        "department": r[4],
                        "position": r[5],
                        "employee_number": r[6],
                        "image_path": r[7],
                        "similarity": round(similarity, 4),
                        "similarity_percent": round(similarity * 100, 2)
                    })

            return results
        except Exception as e:
            self.conn.rollback()
            print(f"[ERREUR DB search_nearest_face] {e}")
            return []

    def init_history_table(self):
        """Crée la table search_history si elle n'existe pas encore dans PostgreSQL."""
        try:
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id SERIAL PRIMARY KEY,
                    timestamp VARCHAR(50),
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    department VARCHAR(100),
                    position VARCHAR(100),
                    similarity_percent FLOAT,
                    status VARCHAR(50),
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[ERREUR DB init_history_table] {e}")

    def log_search_event(self, event_data: dict):
        """Enregistre un événement de recherche ou pointage dans la table PostgreSQL search_history."""
        try:
            self.init_history_table()
            import time
            timestamp = event_data.get("timestamp") or time.strftime("%H:%M:%S")
            first_name = event_data.get("first_name", "Inconnu")
            last_name = event_data.get("last_name", "")
            department = event_data.get("department", "Général")
            position = event_data.get("position", "Employé")
            similarity = float(event_data.get("similarity", 0.0))
            status = event_data.get("status", "APPROVED")
            image_path = event_data.get("image_path", "")

            self.cur.execute("""
                INSERT INTO search_history (timestamp, first_name, last_name, department, position, similarity_percent, status, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (timestamp, first_name, last_name, department, position, similarity, status, image_path))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[ERREUR DB log_search_event] {e}")

    def get_search_history(self, limit: int = 50) -> list:
        """Récupère l'historique des recherches/pointages depuis la base PostgreSQL."""
        try:
            self.init_history_table()
            self.cur.execute("""
                SELECT timestamp, first_name, last_name, department, position, similarity_percent, status, image_path
                FROM search_history
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            rows = self.cur.fetchall()
            history = []
            for r in rows:
                history.append({
                    "timestamp": r[0],
                    "first_name": r[1],
                    "last_name": r[2],
                    "department": r[3],
                    "position": r[4],
                    "similarity": float(r[5]) if r[5] is not None else 0.0,
                    "status": r[6],
                    "image_path": r[7]
                })
            return history
        except Exception as e:
            self.conn.rollback()
            print(f"[ERREUR DB get_search_history] {e}")
            return []

    def close(self):
        self.cur.close()
        self.conn.close()