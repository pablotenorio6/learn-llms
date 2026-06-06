"""Índice BM25 en memoria (solo stdlib) para la rama léxica del híbrido.

Por qué BM25 además del denso: el embedding (nomic) difumina queries que
contienen términos exactos o "meta" ausentes en el texto del documento
(nombres propios, siglas, "currículum"). BM25 puntúa por coincidencia léxica
exacta con saturación de term-frequency y normalización por longitud, así que
rescata justo esos casos. El retriever fusiona ambos rankings con RRF.

El corpus es pequeño (notas/documentos del usuario), así que un índice en
memoria reconstruido cuando cambia la colección es más que suficiente; no
justifica vectores sparse en Qdrant todavía.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_ACCENTS = str.maketrans("áéíóúüñ", "aeiouun")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().translate(_ACCENTS))


class BM25Index:
    """BM25 Okapi. fit() con (id, texto); search() devuelve [(id, score)] desc."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._tf: list[Counter] = []
        self._dl: list[int] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0

    def fit(self, ids: list[str], texts: list[str]) -> None:
        self.ids = list(ids)
        self._tokens = [tokenize(t) for t in texts]
        self._tf = [Counter(tl) for tl in self._tokens]
        self._dl = [len(tl) for tl in self._tokens]
        self._n = len(self._tokens)
        self._avgdl = (sum(self._dl) / self._n) if self._n else 0.0
        self._df = Counter()
        for tl in self._tokens:
            for w in set(tl):
                self._df[w] += 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # idf con suavizado; siempre positivo
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._n == 0:
            return []
        q_terms = tokenize(query)
        scored: list[tuple[str, float]] = []
        for i in range(self._n):
            tf = self._tf[i]
            dl = self._dl[i]
            s = 0.0
            for w in q_terms:
                f = tf.get(w)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += self._idf(w) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scored.append((self.ids[i], s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
