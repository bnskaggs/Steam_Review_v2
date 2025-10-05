"""Command line tool to expand taxonomy keywords from review data."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd
import yaml

WINDOW_RADIUS = 6
MAX_NGRAM = 3
MAX_SUGGESTIONS = 30

# A lightweight English stopword list. Keeping it local avoids pulling optional dependencies.
STOPWORDS: Set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "couldn",
    "did",
    "didn",
    "do",
    "does",
    "doesn",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "has",
    "hasn",
    "have",
    "haven",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "it",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "might",
    "more",
    "most",
    "must",
    "my",
    "myself",
    "needn",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "she",
    "should",
    "shouldn",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn",
    "we",
    "were",
    "weren",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "would",
    "wouldn",
    "y",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in re.findall(r"[\w']+", text.lower()):
        token = raw.strip("'")
        if token:
            tokens.append(token)
    return tokens


def _normalize_phrase(text: str) -> str:
    return " ".join(_tokenize(text))


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"[\n\r]+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part for part in parts if part]


def _find_matches(tokens: Sequence[str], seed_tokens: Sequence[str]) -> List[int]:
    if not seed_tokens:
        return []
    window = len(seed_tokens)
    limit = len(tokens) - window + 1
    if limit <= 0:
        return []
    matches: List[int] = []
    for idx in range(limit):
        if tokens[idx : idx + window] == list(seed_tokens):
            matches.append(idx)
    return matches


def _window_tokens(tokens: Sequence[str], start: int, seed_len: int, radius: int = WINDOW_RADIUS) -> List[str]:
    lo = max(0, start - radius)
    hi = min(len(tokens), start + seed_len + radius)
    return list(tokens[lo:hi])


def _extract_ngrams(tokens: Sequence[str], max_n: int = MAX_NGRAM) -> Set[str]:
    output: Set[str] = set()
    length = len(tokens)
    for n in range(1, max_n + 1):
        if length < n:
            break
        for idx in range(length - n + 1):
            ngram_tokens = tokens[idx : idx + n]
            ngram = " ".join(ngram_tokens)
            if ngram:
                output.add(ngram)
    return output


def _load_taxonomy(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    topics = data.get("topics", data)
    if not isinstance(topics, dict):
        raise ValueError("taxonomy must contain a 'topics' mapping")
    normalized: Dict[str, List[str]] = {}
    for topic, keywords in topics.items():
        if keywords is None:
            normalized[topic] = []
        elif isinstance(keywords, list):
            normalized[topic] = [str(keyword) for keyword in keywords]
        else:
            raise ValueError(f"keywords for topic '{topic}' must be a list")
    return normalized


def _select_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "clean_text",
        "text",
        "review",
        "body",
        "content",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError("No supported text column found in parquet. Expected one of: " + ", ".join(candidates))


def _normalize_lang(value) -> str:
    if isinstance(value, str) and value:
        return value.lower()
    return "unknown"


def expand_taxonomy(
    parquet_path: Path,
    taxonomy_path: Path,
    max_suggestions: int = MAX_SUGGESTIONS,
) -> Dict[str, List[str]]:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return {topic: [] for topic in _load_taxonomy(taxonomy_path)}

    text_column = _select_text_column(df)
    taxonomy = _load_taxonomy(taxonomy_path)

    topic_seed_tokens: Dict[str, List[List[str]]] = {}
    topic_seed_normalized: Dict[str, Set[str]] = {}
    for topic, seeds in taxonomy.items():
        normalized_seeds: Set[str] = set()
        seed_token_list: List[List[str]] = []
        for seed in seeds:
            normalized_seed = _normalize_phrase(seed)
            if not normalized_seed:
                continue
            normalized_seeds.add(normalized_seed)
            seed_token_list.append(_tokenize(seed))
        topic_seed_tokens[topic] = seed_token_list
        topic_seed_normalized[topic] = normalized_seeds

    topic_windows: Dict[str, Dict[str, List[List[str]]]] = defaultdict(lambda: defaultdict(list))

    for record in df.itertuples(index=False):
        text = getattr(record, text_column, None)
        if not isinstance(text, str) or not text.strip():
            continue
        lang = _normalize_lang(getattr(record, "lang", "unknown"))
        for sentence in _split_sentences(text):
            tokens = _tokenize(sentence)
            if not tokens:
                continue
            for topic, seeds in topic_seed_tokens.items():
                for seed_tokens in seeds:
                    for match_index in _find_matches(tokens, seed_tokens):
                        window = _window_tokens(tokens, match_index, len(seed_tokens))
                        if window:
                            topic_windows[topic][lang].append(window)

    topic_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    total_windows = 0
    for topic, lang_windows in topic_windows.items():
        for windows in lang_windows.values():
            for window_tokens in windows:
                total_windows += 1
                for ngram in _extract_ngrams(window_tokens):
                    topic_counts[topic][ngram] += 1
                    global_counts[ngram] += 1

    if total_windows == 0:
        return {topic: [] for topic in taxonomy.keys()}

    suggestions: Dict[str, List[str]] = {}
    for topic, counts in topic_counts.items():
        candidates = []
        seed_norm = topic_seed_normalized.get(topic, set())
        for ngram, freq in counts.items():
            tokens = ngram.split()
            if all(token in STOPWORDS for token in tokens):
                continue
            if ngram in seed_norm:
                continue
            if not any(char.isalpha() for char in ngram):
                continue
            global_freq = global_counts.get(ngram, 0)
            score = freq * math.log(total_windows / (1 + global_freq))
            candidates.append((score, freq, ngram))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        top_terms = [term for _, _, term in candidates[:max_suggestions]]
        suggestions[topic] = top_terms

    # Ensure every topic from taxonomy has an entry, even if empty.
    for topic in taxonomy.keys():
        suggestions.setdefault(topic, [])

    return suggestions


def write_suggestions(path: Path, suggestions: Dict[str, List[str]]) -> None:
    payload = {"topics": {topic: {"add": terms} for topic, terms in suggestions.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True))


def apply_suggestions(taxonomy_path: Path, suggestions: Dict[str, List[str]]) -> None:
    taxonomy_data = yaml.safe_load(taxonomy_path.read_text()) or {}
    topics = taxonomy_data.get("topics", taxonomy_data)
    if not isinstance(topics, dict):
        raise ValueError("taxonomy must contain a 'topics' mapping")

    for topic, new_terms in suggestions.items():
        existing = topics.get(topic, [])
        if existing is None:
            existing = []
        if not isinstance(existing, list):
            raise ValueError(f"keywords for topic '{topic}' must be a list")
        normalized: Dict[str, str] = {str(term).lower(): str(term) for term in existing}
        for term in new_terms:
            normalized.setdefault(term.lower(), term)
        topics[topic] = sorted(normalized.values(), key=lambda value: (value.lower(), value))

    taxonomy_data["topics"] = topics
    taxonomy_path.write_text(yaml.safe_dump(taxonomy_data, sort_keys=False, allow_unicode=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand taxonomy seeds from review data")
    parser.add_argument("--in-parquet", required=True, dest="in_parquet", help="Path to filtered parquet of reviews")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.yaml")
    parser.add_argument(
        "--out",
        required=True,
        help="Destination path for taxonomy_suggestions.yaml",
    )
    parser.add_argument("--max", type=int, default=MAX_SUGGESTIONS, help="Maximum suggestions per topic")
    parser.add_argument("--apply", action="store_true", help="Merge suggestions back into taxonomy.yaml")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parquet_path = Path(args.in_parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(args.in_parquet)
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.exists():
        raise FileNotFoundError(args.taxonomy)
    out_path = Path(args.out)

    suggestions = expand_taxonomy(parquet_path, taxonomy_path, max_suggestions=args.max)
    write_suggestions(out_path, suggestions)

    if args.apply:
        apply_suggestions(taxonomy_path, suggestions)


if __name__ == "__main__":
    main()
