\
"""

But :
- Construire automatiquement une base de données (SQLite + JSON) à partir d'un dossier de PDF
  d'articles scientifiques sur des fractures (plateau tibial / pelvis / radius).

Fonctionnalités :
1) Parcourir un dossier de PDFs
2) Justifier le nombre de PDFs :
   - compter les fichiers trouvés
   - sauvegarder un rapport JSON ("pdf_count_report.json") avec la liste des PDFs
3) Pour chaque PDF :
   - extraire le texte (PyMuPDF)
   - extraire "Abstract" et "Conclusion" (regex heuristique)
   - classifier selon des mots-clés (région / type / localisation)
   - produire un résumé extractif avec DrBERT (optionnel)
   - stocker les résultats dans SQLite
4) Exporter toute la base en JSON

Usage:
python -m scripts.build_db --pdf_dir data/pdfs --db outputs/fractures.sqlite --json outputs/fractures.json --summarize"""

from __future__ import annotations
import argparse, os, json, hashlib, datetime
from tqdm import tqdm

from scripts.keywords import REGION_KEYWORDS, FRACTURE_TYPE_KEYWORDS, LOCATION_KEYWORDS
from scripts.rules import score_keywords
from scripts.pdf_utils import extract_text_pymupdf, extract_sections

from scripts.db import connect, upsert_paper, fetch_all

from scripts.drbert_embed import load_drbert
from scripts.drbert_extractive_summarizer import summarize_extractive_drbert

import fitz



def sha256_file(path: str) -> str:

    """
    Calcule le hash SHA256 du fichier :
    - créer un identifiant unique basé sur le contenu du fichier,
      ce qui permet d'éviter les doublons en base même si le nom change.
    - SHA256 = empreinte (fingerprint) du fichier.

    """

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


#Parcourt récursivement un dossier et renvoie la liste de tous les PDFs trouvés.
def list_pdfs(pdf_dir: str):
    out = []
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                out.append(os.path.join(root, fn))
    return sorted(out)

def main():
    """
    Fonction principale :
    - parse les arguments
    - construit le rapport de comptage
    - charge DrBERT si nécessaire
    - traite chaque PDF (extraction + classification + résumé)
    - stocke dans SQLite et exporte en JSON
    """

    # 1) Lecture des arguments CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Dossier contenant les PDFs")
    ap.add_argument("--db", default="outputs/fractures.sqlite", help="Chemin SQLite")
    ap.add_argument("--json", default="outputs/fractures.json", help="Export JSON")
    ap.add_argument("--max_pages", type=int, default=30, help="Max pages extraites par PDF (simple, rapide)")
    ap.add_argument("--summarize", action="store_true", help="Activer le résumé local")
    ap.add_argument("--model_name", default="google/flan-t5-base", help="Modèle local pour résumé")
    args = ap.parse_args()

    # 2) Recherche des PDFs + justification 
    pdfs = list_pdfs(args.pdf_dir)
    n = len(pdfs)

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)

    # Rapport de justification du nombre de PDFs
    report = {
        "pdf_dir": args.pdf_dir,
        "n_pdfs": n,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "files": [{"file_path": p, "file_name": os.path.basename(p)} for p in pdfs],
    }
    with open(os.path.join(os.path.dirname(args.json), "pdf_count_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] PDFs trouvés: {n}")
    if n == 0:
        print("[WARN] Aucun PDF. Mets tes fichiers dans data/pdfs/")
        return


    # 3) Chargement de DrBERT
    dr_tok = dr_model = dr_device = None
    if args.summarize:
        print("[INFO] Chargement DrBERT (résumé extractif)")
        dr_tok, dr_model, dr_device = load_drbert("Dr-BERT/DrBERT-7GB")


    # 4) Connexion à la base SQLite
    conn = connect(args.db)


    # 5) Traitement de chaque PDF
    for pdf_path in tqdm(pdfs, desc="Indexation"):
        try:
            # 5.1) SHA256 pour identifier le PDF de façon unique
            file_hash = sha256_file(pdf_path)
            # 5.2) Extraction texte brut (limité à max_pages)
            text = extract_text_pymupdf(pdf_path, max_pages=args.max_pages)
            # 5.3) Extraction des sections
            sections = extract_sections(text)

            # 5.4) Classification par mots-clés
            region_res = score_keywords(sections.full_text, REGION_KEYWORDS)
            type_res = score_keywords(sections.full_text, FRACTURE_TYPE_KEYWORDS)
            loc_res = score_keywords(sections.full_text, LOCATION_KEYWORDS)

            # 5.5) Résumé on résume prioritairement abstract+conclusion si disponibles, sinon début du texte.
            summary = None
            if dr_tok is not None:
                payload_parts = []
                if sections.abstract:
                    payload_parts.append(sections.abstract)
                if sections.conclusion:
                    payload_parts.append(sections.conclusion)

                payload = "\n\n".join(payload_parts).strip()

                # fallback si abstract/conclusion absents
                if not payload:
                    payload = sections.full_text[:6000].strip()

                # fallback si texte vide
                if not payload:
                    summary = "SUMMARY_UNAVAILABLE: no extractable text (maybe scanned PDF, OCR needed)."
                else:
                    summary = summarize_extractive_drbert(dr_tok, dr_model, dr_device, payload, k=7)
                    if not summary.strip():
                        summary = "SUMMARY_UNAVAILABLE: summarizer returned empty output."


            # 5.6) Nombre de pages
            doc = fitz.open(pdf_path)
            n_pages = doc.page_count
            doc.close()
            
            # 5.7) Construction de la ligne à stocker en base
            row = {
                "sha256": file_hash,
                "file_name": os.path.basename(pdf_path),
                "file_path": pdf_path,
                "n_pages": n_pages,  # on garde simple (option: récupérer avec fitz)
                "text_method": "pymupdf_text",
                "region_label": region_res.best_label,
                "region_scores": json.dumps(region_res.scores, ensure_ascii=False),
                "fracture_type_label": type_res.best_label,
                "fracture_type_scores": json.dumps(type_res.scores, ensure_ascii=False),
                "location_label": loc_res.best_label,
                "location_scores": json.dumps(loc_res.scores, ensure_ascii=False),
                "abstract": sections.abstract,
                "conclusion": sections.conclusion,
                "summary": summary,
               
            }

             # 5.8) Insertion / mise à jour dans la base SQLite
            upsert_paper(conn, row)
        except Exception as e:
            print(f"[ERROR] {pdf_path}: {e}")


    # 6) Export final de la base en JSON
    all_rows = fetch_all(conn)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"[DONE] SQLite: {args.db}")
    print(f"[DONE] JSON:   {args.json}")
    print(f"[DONE] Rapport nb PDFs: {os.path.join(os.path.dirname(args.json), 'pdf_count_report.json')}")

if __name__ == "__main__":
    main()
