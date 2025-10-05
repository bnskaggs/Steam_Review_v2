"""Rule-based review classification."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

_TOPIC_PRIORITY: Tuple[str, ...] = (
    "monetization",
    "performance",
    "combat",
    "difficulty",
    "progression",
)

_NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "without",
    "isnt",
    "isn't",
    "arent",
    "aren't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "cant",
    "can't",
    "won't",
    "wont",
    "shouldn't",
    "shouldnt",
    "couldn't",
    "couldnt",
}

_POSITIVE_CUES: Tuple[str, ...] = (
    "cheap",
    "fair",
    "smooth",
    "stable",
    "fast",
    "responsive",
    "balanced",
    "good",
    "great",
    "awesome",
    "amazing",
    "love",
    "fun",
    "enjoy",
    "solid",
)

_NEGATIVE_TOPIC_CUES: Dict[str, Tuple[str, ...]] = {
    "monetization": (
        "expensive",
        "overpriced",
        "paywall",
        "greedy",
        "predatory",
        "grindy",
        "terrible",
        "bad",
    ),
    "performance": ("crash", "stutter", "lag", "freeze", "fps drop"),
    "difficulty": ("impossible", "unfair", "broken", "too hard", "cheap"),
    "combat": ("clunky", "unresponsive", "delayed", "rollback"),
    "progression": tuple(),
}

_VARIANT_MAP: Dict[str, Tuple[str, ...]] = {
    "price": ("price", "prices", "priced", "pricing"),
    "microtransaction": ("microtransaction", "microtransactions"),
    "dlc": ("dlc",),
    "shop": ("shop", "shops", "store", "stores"),
    "store": ("store", "stores", "shop", "shops"),
    "currency": ("currency", "currencies"),
    "coin": ("coin", "coins"),
    "coins": ("coin", "coins"),
    "battle pass": ("battle pass", "battle-pass", "battlepass"),
}

_WORD_RE = re.compile(r"\b\w+(?:'\w+)?\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class _Pattern:
    term: str
    regex: re.Pattern[str]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_taxonomy(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    topics = payload.get("topics", {})
    return {str(topic): [str(kw).lower() for kw in kws] for topic, kws in topics.items()}


def _tokenize(text: str) -> List[str]:
    return [token.strip("'") for token in _WORD_RE.findall(text.lower())]


def _phrase_variants(keyword: str) -> Iterable[str]:
    base = keyword.lower().strip()
    if not base:
        return []
    if base in _VARIANT_MAP:
        return _VARIANT_MAP[base]

    variants: set[str] = {base}
    if " " not in base and base.isalpha():
        if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
            variants.add(f"{base[:-1]}ies")
        elif base.endswith(("s", "x", "z", "ch", "sh")):
            variants.add(f"{base}es")
        else:
            variants.add(f"{base}s")
    return tuple(sorted(variants))


def _compile_patterns(taxonomy: Dict[str, List[str]]) -> Dict[str, List[_Pattern]]:
    compiled: Dict[str, List[_Pattern]] = {}
    for topic, keywords in taxonomy.items():
        topic_patterns: List[_Pattern] = []
        seen: set[str] = set()
        for keyword in keywords:
            for variant in _phrase_variants(keyword):
                if variant in seen:
                    continue
                seen.add(variant)
                escaped = re.escape(variant)
                pattern_text = escaped.replace(r"\ ", r"\s+")
                regex = re.compile(rf"(?i)\b{pattern_text}\b")
                topic_patterns.append(_Pattern(term=variant, regex=regex))
        compiled[topic] = topic_patterns
    return compiled


def _sentence_hits(sentence: str, patterns: Sequence[_Pattern]) -> int:
    hits = 0
    for pattern in patterns:
        hits += len(list(pattern.regex.finditer(sentence)))
    return hits


def _token_matches(token: str, cue_token: str) -> bool:
    return token == cue_token or (token.startswith(cue_token) and len(token) > len(cue_token))


def _phrase_occurrences(tokens: Sequence[str], phrase: str) -> List[int]:
    cue_tokens = [tok for tok in phrase.split() if tok]
    if not cue_tokens:
        return []
    window = len(cue_tokens)
    hits: List[int] = []
    for idx in range(len(tokens) - window + 1):
        segment = tokens[idx : idx + window]
        if all(_token_matches(seg, cue_tokens[pos]) for pos, seg in enumerate(segment)):
            hits.append(idx)
    return hits


def _has_negation(tokens: Sequence[str], start: int, span: int) -> bool:
    window_start = max(0, start - 3)
    window_end = min(len(tokens), start + span + 3)
    for idx in range(window_start, window_end):
        if tokens[idx] in _NEGATION_TERMS:
            return True
    return False


def _count_sentiment(tokens: Sequence[str], topic: str) -> Tuple[int, int, int]:
    positive_hits = 0
    negative_hits = 0
    cue_matches = 0

    negatives = _NEGATIVE_TOPIC_CUES.get(topic, tuple())
    negative_set = set(negatives)

    for cue in _POSITIVE_CUES:
        if cue in negative_set:
            continue
        occurrences = _phrase_occurrences(tokens, cue)
        for occ in occurrences:
            cue_matches += 1
            flipped = _has_negation(tokens, occ, len(cue.split()))
            if flipped:
                negative_hits += 1
            else:
                positive_hits += 1

    bad_tokens = {"bad", "poor"}
    for cue in negatives:
        occurrences = _phrase_occurrences(tokens, cue)
        if cue == "rollback" and occurrences:
            # Require nearby negative qualifier
            has_bad = any(
                any(tokens[j] in bad_tokens for j in range(max(0, occ - 2), min(len(tokens), occ + 3)))
                for occ in occurrences
            )
            if not has_bad:
                continue
        for occ in occurrences:
            cue_matches += 1
            flipped = _has_negation(tokens, occ, len(cue.split()))
            if flipped:
                positive_hits += 1
            else:
                negative_hits += 1

    return positive_hits, negative_hits, cue_matches


def _sentence_confidence(topic_hits: int, cue_hits: int, tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    length = len(tokens)
    effective_length = max(3, length - 2)
    clamp_len = max(5.0, min(float(effective_length), 60.0))
    score = (topic_hits + cue_hits) / clamp_len
    return max(0.0, min(1.0, score))


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def classify(
    in_parquet: str,
    out_parquet: str,
    taxonomy: str,
    min_conf: float = 0.45,
) -> Dict[str, object]:
    """Assign topics to reviews using keyword matching."""
    src = Path(in_parquet)
    if not src.exists():
        raise FileNotFoundError(in_parquet)

    tax_path = Path(taxonomy)
    if not tax_path.exists():
        raise FileNotFoundError(taxonomy)

    taxonomy_map = _load_taxonomy(tax_path)
    pattern_map = _compile_patterns(taxonomy_map)

    df = pd.read_parquet(src)
    rows_in = int(df.shape[0])

    sentence_records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        review_id = row.get("review_id")
        text = row.get("clean_text", "") or ""
        sentences = _split_sentences(str(text))
        for sentence in sentences:
            lower_sentence = sentence.lower()
            topic_hits_map: Dict[str, int] = {}
            for topic, patterns in pattern_map.items():
                hits = _sentence_hits(lower_sentence, patterns)
                if hits:
                    topic_hits_map[topic] = hits
            if not topic_hits_map:
                continue

            selected_topic = None
            for topic in _TOPIC_PRIORITY:
                if topic in topic_hits_map:
                    selected_topic = topic
                    break
            if selected_topic is None:
                # If taxonomy contains additional topics outside priority order
                selected_topic = next(iter(topic_hits_map.keys()))

            topic_hits = topic_hits_map[selected_topic]
            tokens = _tokenize(sentence)
            pos_hits, neg_hits, cue_hits = _count_sentiment(tokens, selected_topic)
            if pos_hits > neg_hits:
                sentiment = 1
            elif neg_hits > pos_hits:
                sentiment = -1
            else:
                sentiment = 0

            confidence = _sentence_confidence(topic_hits, cue_hits, tokens)
            if confidence < min_conf:
                continue

            sentence_records.append(
                {
                    "review_id": review_id,
                    "topic": selected_topic,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4),
                    "rationale": sentence.strip(),
                }
            )

    aggregated: Dict[Tuple[object, str], Dict[str, object]] = {}
    for record in sentence_records:
        key = (record["review_id"], record["topic"])
        existing = aggregated.get(key)
        if existing is None or record["confidence"] > existing["confidence"]:
            aggregated[key] = record

    if aggregated:
        labeled_df = pd.DataFrame(aggregated.values())
    else:
        labeled_df = pd.DataFrame(columns=["review_id", "topic", "sentiment", "confidence", "rationale"])

    dest = Path(out_parquet)
    _ensure_parent(dest)
    labeled_df.to_parquet(dest, index=False)

    labeled_reviews = set(labeled_df["review_id"].dropna().tolist()) if not labeled_df.empty else set()
    rows_labeled = len(labeled_reviews)
    blank_pct = 1 - (rows_labeled / rows_in) if rows_in else 0.0
    avg_conf = float(labeled_df["confidence"].mean()) if not labeled_df.empty else 0.0

    metrics = {
        "rows_in": rows_in,
        "rows_labeled": rows_labeled,
        "blank_pct": round(float(blank_pct), 4),
        "avg_conf": round(float(avg_conf), 4),
        "min_conf": min_conf,
    }
    LOGGER.info("Classification complete: %s", metrics)
    return metrics


__all__ = ["classify"]
