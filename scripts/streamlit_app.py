\
"""
Rôle :
Créer une application Web (Streamlit) pour :

1) Explorer la base SQLite :
   - liste des PDFs indexés
   - affichage region / fracture_type / location (TOP1)
   - affichage abstract / conclusion / summary

2) Tester l'inférence sur un nouveau PDF (upload) :
   - extraction texte
   - classification mots-clés
   - résumé DrBERT 
   - affichage immédiat dans l'interface
   - export JSON téléchargeable

"""


from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import datetime
from typing import List, Dict, Any

import streamlit as st
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Fix OpenMP Windows (évite libiomp5md.dll crash) ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# --- Imports projet ---
from scripts.keywords import REGION_KEYWORDS, FRACTURE_TYPE_KEYWORDS, LOCATION_KEYWORDS
from scripts.rules import score_keywords
from scripts.pdf_utils import extract_text_pymupdf, extract_sections

# DrBERT (résumé extractif)
from scripts.drbert_embed import load_drbert
from scripts.drbert_extractive_summarizer import summarize_extractive_drbert


DEFAULT_DB = "outputs/fractures.sqlite"


def db_connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def db_fetch_all(db_path: str) -> List[Dict[str, Any]]:
    """
    Lit la table SQLite et renvoie une liste de dicts.
    """
    conn = db_connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          sha256, file_name, file_path, n_pages,
          region_label, fracture_type_label, location_label,
          abstract, conclusion, summary
        FROM papers
        ORDER BY id DESC;
        """
    )
    rows = []
    for r in cur.fetchall():
        rows.append(
            {
                "sha256": r[0],
                "file_name": r[1],
                "file_path": r[2],
                "n_pages": r[3],
                "region": r[4],
                "fracture_type": r[5],
                "location": r[6],
                "abstract": r[7],
                "conclusion": r[8],
                "summary": r[9],
               
            }
        )
    conn.close()
    return rows


@st.cache_resource
def get_drbert():
    return load_drbert("Dr-BERT/DrBERT-7GB")


def run_inference_on_pdf(pdf_path: str, max_pages: int = 30, do_summarize: bool = True) -> Dict[str, Any]:
    text = extract_text_pymupdf(pdf_path, max_pages=max_pages)
    sections = extract_sections(text)

    region_res = score_keywords(sections.full_text, REGION_KEYWORDS)
    type_res = score_keywords(sections.full_text, FRACTURE_TYPE_KEYWORDS)
    loc_res = score_keywords(sections.full_text, LOCATION_KEYWORDS)

    summary = None
    if do_summarize:
        dr_tok, dr_model, dr_device = get_drbert()

        payload_parts = []
        if sections.abstract:
            payload_parts.append(sections.abstract)
        if sections.conclusion:
            payload_parts.append(sections.conclusion)
        payload = "\n\n".join(payload_parts).strip()

        if not payload:
            payload = sections.full_text[:6000].strip()

        if not payload:
            summary = "SUMMARY_UNAVAILABLE: no extractable text (maybe scanned PDF, OCR needed)."
        else:
            summary = summarize_extractive_drbert(dr_tok, dr_model, dr_device, payload, k=7)
            if not summary.strip():
                summary = "SUMMARY_UNAVAILABLE: summarizer returned empty output."

    return {
        "file_name": os.path.basename(pdf_path),
        "file_path": pdf_path,
        "region": region_res.best_label,
        "fracture_type": type_res.best_label,
        "location": loc_res.best_label,
        "abstract": sections.abstract,
        "conclusion": sections.conclusion,
        "summary": summary,
       
    }


def save_json(obj: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- UI ----------------
st.set_page_config(page_title="PDF Fractures Explorer", layout="wide")
st.title("Système d'analyse de fractions")

with st.sidebar:
    st.header("Paramètres")
    db_path = st.text_input("Chemin DB SQLite", value=DEFAULT_DB)
    max_pages = st.slider("Max pages à extraire (infer)", 1, 80, 30)
    do_summarize = st.checkbox("Activer résumé", value=True)

tabs = st.tabs(["📚 Articles", "🧪 Upload nouveaux PDF)"])

# --- Tab 1: DB Explorer ---
with tabs[0]:
    st.subheader("Contenu de la base")
    if not os.path.exists(db_path):
        st.error(f"DB introuvable: {db_path}\n\n👉 Lance d'abord build_db pour créer outputs/fractures.sqlite")
    else:
        rows = db_fetch_all(db_path)

        if not rows:
            st.warning("Base vide. Ajoute des PDFs dans data/pdfs puis relance build_db.")
        else:
            # Tableau compact pour sélection
            display_rows = [
                {
                    "Nom d'article": r["file_name"],
                    "N° de pages": r["n_pages"],
                    "Region": r["region"],
                    "Type de Fracture": r["fracture_type"],
                    "Location": r["location"],
                    
                }
                for r in rows
            ]

            try:
                import pandas as pd

                df = pd.DataFrame(display_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                st.write(display_rows)

            choices = [r["file_name"] for r in rows]
            selected = st.selectbox("Choisir un PDF", choices, index=0)

            rec = next(r for r in rows if r["file_name"] == selected)

            c1, c2, c3 = st.columns(3)
            c1.metric("Region", rec["region"] or "None")
            c2.metric("Fracture type", rec["fracture_type"] or "None")
            c3.metric("Location", rec["location"] or "None")

            st.markdown("### Abstract")
            st.write(rec["abstract"] if rec["abstract"] else "Abstract non détecté.")

            st.markdown("### Conclusion")
            st.write(rec["conclusion"] if rec["conclusion"] else "Conclusion non détectée.")

            st.markdown("### Résumer ")
            st.write(rec["summary"] if rec["summary"] else "Summary = null (relance build_db avec --summarize)")

# --- Tab 2: Upload + Infer ---
with tabs[1]:
    st.subheader("Upload un PDF")
    uploaded = st.file_uploader("Choisir un fichier PDF", type=["pdf"])

    if uploaded is not None:
        st.info("Fichier reçu. Lancement inference…")

        # Sauvegarder temporairement
        os.makedirs("outputs/tmp", exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="outputs/tmp") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            result = run_inference_on_pdf(tmp_path, max_pages=max_pages, do_summarize=do_summarize)
            st.success("Inference terminée ✅")

            c1, c2, c3 = st.columns(3)
            c1.metric("Region", result["region"] or "None")
            c2.metric("Fracture type", result["fracture_type"] or "None")
            c3.metric("Location", result["location"] or "None")

            st.markdown("### Abstract")
            st.write(result["abstract"] if result["abstract"] else "Abstract non détecté.")

            st.markdown("### Conclusion")
            st.write(result["conclusion"] if result["conclusion"] else "Conclusion non détectée.")

            st.markdown("### Résumer")
            st.write(result["summary"] if result["summary"] else "Summary vide / indisponible.")

            out_json = "outputs/infer_result.json"
            save_json(result, out_json)
            st.caption(f"JSON sauvegardé: {out_json}")

            st.download_button(
                label="Télécharger infer_result.json",
                data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="infer_result.json",
                mime="application/json",
            )
        finally:
            pass