import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "reports.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports(
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


def create_report(
    image_name: str,
    image_path: str,
    helmet: int,
    vest: int,
    glasses: int,
    helmet_conf: float,
    vest_conf: float,
    glasses_conf: float,
    deviation: str,
    comment: str = ""
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reports (
            image_name,
            image_path,
            helmet,
            vest,
            glasses,
            helmet_conf,
            vest_conf,
            glasses_conf,
            deviation,
            status,
            comment,
            assigned_to
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        image_path,
        helmet,
        vest,
        glasses,
        helmet_conf,
        vest_conf,
        glasses_conf,
        deviation,
        "Til vurdering",
        comment,
        None
    ))

    conn.commit()
    conn.close()


def get_pending_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM reports
        WHERE status = 'Til vurdering'
    """)

    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_reports_for_review():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, image_name, image_path, deviation, comment, created_at
        FROM reports
        WHERE status = 'Til vurdering'
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def update_report_status(report_id: int, new_status: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE reports
        SET status = ?
        WHERE id = ?
    """, (new_status, report_id))

    conn.commit()
    conn.close()


def get_all_reports():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, image_name, image_path, deviation, status, comment, created_at, assigned_to
        FROM reports
        ORDER BY
            CASE
                WHEN status = 'Til vurdering' THEN 1
                WHEN status = 'Rapportert' THEN 2
                WHEN status = 'Sendt til person' THEN 3
                WHEN status LIKE 'Lukket av %' THEN 4
                WHEN status = 'Avvist' THEN 5
                ELSE 6
            END,
            created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def update_report_status_and_comment(report_id: int, new_status: str, comment: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE reports
        SET status = ?, comment = ?
        WHERE id = ?
    """, (new_status, comment, report_id))

    conn.commit()
    conn.close()

def report_with_comment(report_id: int, comment: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
                    UPDATE reports
                    SET status  = ?,
                        comment = ?
                    WHERE id = ?
                    """, ("Rapportert", comment, report_id))

    conn.commit()
    conn.close()

def assign_report_to_person(report_id: int, person: str, comment: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE reports
        SET status = ?, assigned_to = ?, comment = ?
        WHERE id = ?
    """, ("Sendt til person", person, comment, report_id))

    conn.commit()
    conn.close()

def reject_report(report_id: int, comment: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE reports
        SET status = ?, comment = ?
        WHERE id = ?
    """, ("Avvist", comment, report_id))

    conn.commit()
    conn.close()

def get_reports_for_person(person: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, image_name, image_path, deviation, status, comment, created_at, assigned_to
        FROM reports
        WHERE assigned_to = ?
        ORDER BY created_at DESC
    """, (person,))

    rows = cursor.fetchall()
    conn.close()
    return rows

def close_report(report_id: int, person: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE reports
        SET status = ?
        WHERE id = ?
    """, (f"Lukket av {person}", report_id))

    conn.commit()
    conn.close()

def delete_report(report_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM reports
        WHERE id = ?
    """, (report_id,))

    conn.commit()
    conn.close()
