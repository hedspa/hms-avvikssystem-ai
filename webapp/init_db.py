import sqlite3
from pathlib import Path

DB_PATH = Path("reports.db")

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            helmet INTEGER NOT NULL,
            vest INTEGER NOT NULL,
            glasses INTEGER NOT NULL,
            helmet_conf REAL NOT NULL,
            vest_conf REAL NOT NULL,
            glasses_conf REAL NOT NULL,
            deviation TEXT NOT NULL,
            status TEXT NOT NULL,
            comment TEXT,
            assigned_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("Database opprettet: reports.db")

if __name__ == "__main__":
    init_database()