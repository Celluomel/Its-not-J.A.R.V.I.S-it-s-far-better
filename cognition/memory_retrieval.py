"""Deterministic lexical retrieval used when vector embeddings are unavailable."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset(("""
a about after again all also am an and any are as at be been before but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours yourself yourselves
a au aux avec ce ces dans de des du elle elles en et eux il ils je la le les leur lui ma mais me meme mes moi mon
ne nos notre nous on ou par pas pour qu que quel quelle qui sa sans se ses son sur ta te tes toi ton tu un une vos
votre vous y etre avoir faire suis sont doit dois
""").split())


def tokenize(text: str) -> List[str]:
    """Normalize accents and remove common English/French function words."""
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    plain = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [term for term in _TOKEN_RE.findall(plain) if len(term) > 1 and term not in _STOPWORDS]


def rank_documents(query: str, documents: Sequence[str], limit: int = 5) -> List[Tuple[int, float]]:
    """Return (document index, normalized BM25 relevance), best first."""
    query_terms = Counter(tokenize(query))
    if not query_terms or not documents or limit <= 0:
        return []

    tokenized = [tokenize(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    if average_length <= 0:
        return []

    k1, b = 1.5, 0.75
    scored = []
    for index, terms in enumerate(tokenized):
        frequencies = Counter(terms)
        length = len(terms)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1.0 + (len(tokenized) - df + 0.5) / (df + 0.5))
            saturation = frequency * (k1 + 1.0) / (
                frequency + k1 * (1.0 - b + b * length / average_length)
            )
            score += inverse_frequency * saturation * min(2, query_frequency)
        if score > 0:
            scored.append((index, score / (score + 1.0)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]
