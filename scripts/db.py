\
"""
Rôle :
- Gérer le stockage local dans une base SQLite (table : 'papers')
- Fournir des fonctions 
  1) connect() : ouvre la DB + crée la table
  2) upsert_paper() : insère ou met à jour une ligne (anti-doublons grâce au sha256)
  3) fetch_all() : lit toutes les lignes et renvoie une liste de dictionnaires Python

"""

from __future__ import annotations
import sqlite3
from typing import Dict, Any, Optional, List

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT UNIQUE,
    file_name TEXT,
    file_path TEXT,
    n_pages INTEGER,
    text_method TEXT,
    region_label TEXT,
    region_scores TEXT,
    fracture_type_label TEXT,
    fracture_type_scores TEXT,
    location_label TEXT,
    location_scores TEXT,
    abstract TEXT,
    conclusion TEXT,
    summary TEXT
    
);
"""

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA)
    conn.commit()
    return conn

def upsert_paper(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = [
        "sha256","file_name","file_path","n_pages","text_method",
        "region_label","region_scores",
        "fracture_type_label","fracture_type_scores",
        "location_label","location_scores",
        "abstract","conclusion","summary"
    ]
    placeholders = ",".join(["?"] * len(cols))
    sql = f"""
    INSERT INTO papers ({",".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(sha256) DO UPDATE SET
      file_name=excluded.file_name,
      file_path=excluded.file_path,
      n_pages=excluded.n_pages,
      text_method=excluded.text_method,
      region_label=excluded.region_label,
      region_scores=excluded.region_scores,
      fracture_type_label=excluded.fracture_type_label,
      fracture_type_scores=excluded.fracture_type_scores,
      location_label=excluded.location_label,
      location_scores=excluded.location_scores,
      abstract=excluded.abstract,
      conclusion=excluded.conclusion,
      summary=excluded.summary
    ;
    """
    conn.execute(sql, [row.get(c) for c in cols])
    conn.commit()

def fetch_all(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT sha256,file_name,file_path,n_pages,text_method,region_label,region_scores,fracture_type_label,fracture_type_scores,location_label,location_scores,abstract,conclusion,summary FROM papers ORDER BY id ASC;")
    out = []
    for r in cur.fetchall():
        out.append({
            "file_name": r[1],
            "file_path": r[2],
            "n_pages": r[3],

            "region": r[5],
            "fracture_type": r[7],
            "location": r[9],

            "abstract": r[11],
            "conclusion": r[12],
            "summary": r[13],
           
        })
    return out
