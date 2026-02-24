\
"""


- Extraire le texte brut d'un fichier PDF scientifique
- Extraire de manière heuristique les sections :
    • Abstract
    • Conclusion


- Extraction texte avec PyMuPDF (fitz)
- Détection de sections via expressions régulières (regex)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import re
import fitz  # PyMuPDF

@dataclass
class ExtractedSections:
    full_text: str
    abstract: Optional[str]
    conclusion: Optional[str]

def extract_text_pymupdf(pdf_path: str, max_pages: int | None = None) -> str:
    """
    Extrait le texte brut d'un PDF.

    Étapes :
    1) Ouverture du document avec fitz
    2) Lecture page par page
    3) Extraction du texte brut avec get_text("text")
    4) Concaténation des pages

    Retour :
    - Texte brut complet sous forme d'une seule string
    """


    doc = fitz.open(pdf_path)
    texts = []
    n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    for i in range(n):
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)
# Fonction de nettoyage
def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Extraction heuristique des sections Abstract / Conclusion
def extract_sections(text: str) -> ExtractedSections:
    t = text

    # On cherche un bloc après "Abstract" jusqu'à "Introduction/Background/Methods/Keywords"
    abstract = None
    abs_patterns = [
        r"\babstract\b[:\s]*",
        r"\brésumé\b[:\s]*", 
    ]
    stop_after = r"\b(introduction|background|methods?|materials?|keywords?|mots[-\s]?clés)\b"

    for p in abs_patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            start = m.end()
            # Cherche où s'arrêter
            m2 = re.search(stop_after, t[start:], flags=re.IGNORECASE)
            end = start + (m2.start() if m2 else min(2000, len(t[start:])))
            abstract = _clean(t[start:end])[:2500]
            break

    # Conclusion: on cherche "Conclusion(s)" ou "Discussion"->fin
    conclusion = None
    concl_patterns = [
        r"\bconclusions?\b[:\s]*",
        r"\bconcluding remarks\b[:\s]*",
    ]
    for p in concl_patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            start = m.end()
            # souvent après conclusion: references/acknowledgments
            stop = r"\b(references?|bibliography|acknowledg(e)?ments?|funding|conflicts? of interest)\b"
            m2 = re.search(stop, t[start:], flags=re.IGNORECASE)
            end = start + (m2.start() if m2 else min(2000, len(t[start:])))
            conclusion = _clean(t[start:end])[:2500]
            break

    return ExtractedSections(full_text=t, abstract=abstract, conclusion=conclusion)
