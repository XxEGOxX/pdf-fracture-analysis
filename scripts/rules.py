\
"""

Rôle :
classification simple basée sur des règles (rule-based).

Principe :
- Chaque classe (pelvis, tibial_plateau, radius)
  possède une liste de mots-clés associés.
- On calcule un score par classe :
      score = somme des occurrences des mots-clés dans le texte.
- On choisit la classe ayant le score maximal .

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class LabelResult:
    best_label: str | None
    scores: Dict[str, int]


# Fonction principale de scoring
def score_keywords(text: str, keyword_map: Dict[str, List[str]]) -> LabelResult:
    """
    Retourne le max score et les scores détaillés.
    """
    t = text.lower()
    scores: Dict[str, int] = {}
    for label, kws in keyword_map.items():
        s = 0
        for kw in kws:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue
            
            s += t.count(kw_l)
        scores[label] = s

    best = None
    best_score = 0
    for label, s in scores.items():
        if s > best_score:
            best_score = s
            best = label

    return LabelResult(best_label=best if best_score > 0 else None, scores=scores)

# Fonction utilitaire pour récupérer les top-k labels
def top_labels(scores: Dict[str, int], k: int = 5) -> List[Tuple[str, int]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
