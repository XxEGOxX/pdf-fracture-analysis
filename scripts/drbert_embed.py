\
"""
Rôle :
- Charger le modèle biomédical DrBERT
- Transformer un texte en embedding (vecteur sémantique)
- Fournir une fonction de mean pooling pour obtenir une représentation
  unique de document
"""

from __future__ import annotations
import re
import torch
# HuggingFace Transformers :
# AutoTokenizer : convertit texte -> tokens numériques
# AutoModel : charge le modèle Transformer (encodeur)
from transformers import AutoTokenizer, AutoModel


# Charge le modèle DrBERT + tokenizer.
def load_drbert(model_name: str = "Dr-BERT/DrBERT-7GB", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tok, model, device

@torch.no_grad()
# Calcule la moyenne des embeddings token-level pour obtenir un embedding phrase-level.
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.no_grad()
# Transforme un texte en embedding normalisé.
def embed_text(tok, model, device: str, text: str, max_length: int = 256) -> torch.Tensor:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb  