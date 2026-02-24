\
"""
- infer.py sert à tester rapidement le pipeline sur un seul document,
  contrairement à build_db.py qui traite tout un dossier.

analyser UN nouveau PDF et produire immédiatement :
- classification TOP1 : region / fracture_type / location
- extraction : abstract + conclusion
- résumé local (DrBERT, résumé extractif)
- sauvegarde du résultat en JSON
- insertion en base SQLite pour garder une trace

Usage  :
python -m scripts.infer --pdf "path/to/file.pdf" --db outputs/fractures.sqlite --json outputs/infer_result.json --summarize

"""

from __future__ import annotations
import argparse, os, json, hashlib

from scripts.keywords import REGION_KEYWORDS, FRACTURE_TYPE_KEYWORDS, LOCATION_KEYWORDS
from scripts.rules import score_keywords
from scripts.pdf_utils import extract_text_pymupdf, extract_sections

from scripts.db import connect, upsert_paper

from scripts.drbert_embed import load_drbert
from scripts.drbert_extractive_summarizer import summarize_extractive_drbert

import fitz

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Chemin du PDF à analyser")
    ap.add_argument("--db", default="outputs/fractures.sqlite", help="SQLite existant (ou à créer)")
    ap.add_argument("--json", default="outputs/infer_result.json", help="Sortie JSON")
    ap.add_argument("--max_pages", type=int, default=30)
    ap.add_argument("--summarize", action="store_true", help="Activer le résumé local")
    ap.add_argument("--model_name", default="google/flan-t5-base")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)

    summarizer = None
    dr_tok = dr_model = dr_device = None
    if args.summarize:
        print("[INFO] Chargement DrBERT (résumé extractif)")
        dr_tok, dr_model, dr_device = load_drbert("Dr-BERT/DrBERT-7GB")

    pdf_path = args.pdf
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()
    file_hash = sha256_file(pdf_path)
    text = extract_text_pymupdf(pdf_path, max_pages=args.max_pages)
    sections = extract_sections(text)

    region_res = score_keywords(sections.full_text, REGION_KEYWORDS)
    type_res = score_keywords(sections.full_text, FRACTURE_TYPE_KEYWORDS)
    loc_res = score_keywords(sections.full_text, LOCATION_KEYWORDS)

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

        # fallback si texte vide (pdf scanné ou extraction ratée)
        if not payload:
            summary = "SUMMARY_UNAVAILABLE: no extractable text (maybe scanned PDF, OCR needed)."
        else:
            summary = summarize_extractive_drbert(dr_tok, dr_model, dr_device, payload, k=7)
            if not summary.strip():
                summary = "SUMMARY_UNAVAILABLE: summarizer returned empty output."

    result = {
        "file_name": os.path.basename(pdf_path),
        "file_path": pdf_path,
        "n_pages": n_pages,

        "region": region_res.best_label,
        "fracture_type": type_res.best_label,
        "location": loc_res.best_label,

        "abstract": sections.abstract,
        "conclusion": sections.conclusion,
        "summary": summary,
        
    }

    # Sauvegarde JSON
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # on insère aussi en DB pour garder trace
    conn = connect(args.db)
    upsert_paper(conn, {
        "sha256": file_hash,
        "file_name": result["file_name"],
        "file_path": result["file_path"],
        "n_pages": n_pages,
        "text_method": "pymupdf_text",
        "region_label": region_res.best_label,
        "region_scores": json.dumps(region_res.scores, ensure_ascii=False),
        "fracture_type_label": type_res.best_label,
        "fracture_type_scores": json.dumps(type_res.scores, ensure_ascii=False),
        "location_label": loc_res.best_label,
        "location_scores": json.dumps(loc_res.scores, ensure_ascii=False),
        "abstract": result["abstract"],
        "conclusion": result["conclusion"],
        "summary": result["summary"],
        
    })

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
