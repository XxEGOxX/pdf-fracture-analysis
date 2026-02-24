\
"""

Rôle :
- Générer un résumé EXTRACTIF d'un texte scientifique
- Basé sur des embeddings biomédicaux DrBERT
- Le résumé est composé de phrases originales du texte (pas de génération)

Principe général :
1) Découper le texte en phrases
2) Encoder chaque phrase avec DrBERT (embeddings)
3) Calculer un vecteur central (centroid) représentant le thème du document
4) Sélectionner les phrases les plus proches du centroid (similarité cosinus)
5) Retourner les phrases sélectionnées sous forme de résumé
"""
from __future__ import annotations
import re
import torch
from scripts.drbert_embed import embed_text

# Expression régulière pour découper les phrases
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

def split_sentences(text: str, max_sentences: int = 80):
    # Nettoyage : supprime retours ligne multiples, espaces inutiles
    text = re.sub(r"\s+", " ", text).strip()
    # Découpage en phrases
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 30]
    return sents[:max_sentences]

@torch.no_grad()
# Génère un résumé extractif basé sur DrBERT.
def summarize_extractive_drbert(tok, model, device: str, text: str, k: int = 6) -> str:
    
    # 1) Découpage du texte en phrases
    sents = split_sentences(text)
    if not sents:
        return ""

    # 2) Calcul des embeddings DrBERT pour chaque phrase
    embs = []
    for s in sents:
        embs.append(embed_text(tok, model, device, s, max_length=128))
    E = torch.cat(embs, dim=0)  # (n, hidden)

    # 3) Calcul du centroid du document
    # Le centroid est la moyenne des embeddings de toutes les phrases.
    # - Il représente le "thème central" du document
    # Mathématiquement : centroid = (e1 + e2 + ... + en) / n

    centroid = torch.nn.functional.normalize(E.mean(dim=0, keepdim=True), p=2, dim=1)  # (1, hidden)
    
    # 4) Calcul de l’importance des phrases
    # Produit scalaire entre chaque phrase et le centroïde :
    scores = (E @ centroid.T).squeeze(1)  # cosine similarity, (n,)


    # 5) Sélection les phrases les plus importantes
    topk = min(k, len(sents))
    idx = torch.topk(scores, k=topk).indices.tolist()
    idx_sorted = sorted(idx)  # garder l’ordre du texte
    bullets = ["- " + sents[i] for i in idx_sorted]
    return "\n".join(bullets)