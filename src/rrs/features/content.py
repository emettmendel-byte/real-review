"""Per-review content (text) features.

Three layers:

1. **Polars-vectorized** char/word/sentence counts + caps/exclamation/first-person ratios.
   Runs in a few seconds over 1.9M reviews.
2. **Multiprocessed** VADER sentiment + Flesch reading-ease. Both are per-text Python;
   with ~10 worker processes wall-clock drops from ~12 min to ~1-2 min. Flesch is
   computed directly from the standard formula (a vowel-group syllable count) — using
   `textstat` would drag NLTK's CMU dictionary in, which fails in worker subprocesses.
3. **Embeddings + similarity scalars** live in `embeddings.py` (they need the GPU).

`compute_text_features(df)` takes a DataFrame with `review_id` + `text` and returns it
augmented with the layer-1 + layer-2 columns. Embeddings are joined separately in
`build.py` so we don't load 2.96 GB of float32 unless we have to.
"""

from __future__ import annotations

import os
import re
from multiprocessing import Pool

import polars as pl

# Module-global VADER analyzer — created once per worker via the pool initializer.
_SIA = None
_WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
_VOWEL_GROUP_RE = re.compile(r"[aeiouyAEIOUY]+")


def _init_worker() -> None:
    global _SIA
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _SIA = SentimentIntensityAnalyzer()


def _count_syllables(word: str) -> int:
    """Vowel-group count, with the silent-trailing-e adjustment. ≥1 for any non-empty word."""
    n = len(_VOWEL_GROUP_RE.findall(word))
    if n > 1 and word.lower().endswith("e"):
        n -= 1
    return max(n, 1)


def _flesch_reading_ease(text: str) -> float:
    """Standard Flesch RE formula. Higher = easier to read. Typical English ~60-70."""
    sentences = text.count(".") + text.count("!") + text.count("?")
    sentences = max(sentences, 1)
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def _text_metrics(text: str | None) -> tuple[float, float, float, float, float]:
    """Return (vader_compound, vader_pos, vader_neg, vader_neu, flesch). Safe on null/empty."""
    if not text:
        return (0.0, 0.0, 0.0, 1.0, 0.0)  # neutral
    s = _SIA.polarity_scores(text)
    try:
        f = _flesch_reading_ease(text)
    except (ZeroDivisionError, ValueError, TypeError):
        f = 0.0
    return (s["compound"], s["pos"], s["neg"], s["neu"], f)


def _add_polars_text_features(df: pl.DataFrame) -> pl.DataFrame:
    """Char/word/sentence counts + ratios — pure polars, microseconds per million rows."""
    text = pl.col("text").fill_null("")
    char_len = text.str.len_chars()
    # \w+ counts alphanumeric tokens; close enough to "words" for feature use.
    word_count = text.str.count_matches(r"\w+")
    sent_count = text.str.count_matches(r"[.!?]+")
    excl = text.str.count_matches("!")
    caps = text.str.count_matches(r"[A-Z]")
    # Case-insensitive: "I", "i", "we"… all count toward first-person voice.
    first_person = text.str.count_matches(r"(?i)\b(?:I|me|my|mine|we|us|our|ours)\b")

    def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
        return pl.when(den > 0).then(num / den).otherwise(0.0)

    return df.with_columns(
        char_len.alias("char_length"),
        word_count.alias("word_count"),
        sent_count.alias("sentence_count"),
        safe_div(excl, char_len).alias("exclamation_ratio"),
        safe_div(caps, char_len).alias("caps_ratio"),
        safe_div(first_person, word_count).alias("first_person_ratio"),
    )


def _add_vader_flesch(df: pl.DataFrame, processes: int | None = None) -> pl.DataFrame:
    """Fan out VADER + Flesch across worker processes; gather results in order."""
    processes = processes or max(1, (os.cpu_count() or 2) - 2)
    texts = df["text"].to_list()
    with Pool(processes=processes, initializer=_init_worker) as p:
        # chunksize=2000 keeps IPC overhead well under the per-text compute cost.
        results = list(p.imap(_text_metrics, texts, chunksize=2000))
    arr = list(zip(*results, strict=True))  # tuple of five columns
    return df.with_columns(
        pl.Series("vader_compound", arr[0], dtype=pl.Float32),
        pl.Series("vader_pos",      arr[1], dtype=pl.Float32),
        pl.Series("vader_neg",      arr[2], dtype=pl.Float32),
        pl.Series("vader_neu",      arr[3], dtype=pl.Float32),
        pl.Series("flesch_reading_ease", arr[4], dtype=pl.Float32),
    )


def compute_text_features(df: pl.DataFrame, processes: int | None = None) -> pl.DataFrame:
    """Augment a (review_id, text) DataFrame with all content features except embeddings."""
    if "text" not in df.columns:
        raise ValueError("compute_text_features needs a `text` column.")
    df = _add_polars_text_features(df)
    df = _add_vader_flesch(df, processes=processes)
    return df
