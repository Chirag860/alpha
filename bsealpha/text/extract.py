"""Phase 2: turn each announcement's text into a small numeric feature vector.

The disclosure agent reads **only the announcement text** and returns four scores:

* ``direction``   -- signed expected price impact, -1 (very bad) .. +1 (very good)
* ``materiality`` -- how market-moving the news is, 0 (noise) .. 1 (major)
* ``surprise``    -- how unexpected vs a reasonable prior, 0 .. 1
* ``novelty``     -- genuinely new information vs routine/procedural filing, 0 .. 1

Two disciplines make this safe to feed the validated model downstream:

1. **Identity is stripped before the model sees the text** (:func:`sanitize_text`).
   BSE headlines embed the company name and scrip code ("Jindal Poly Films Ltd -
   500227 - Board Meeting ..."); handing those to an LLM lets it pattern-match "oh,
   this is <name> in <period>, which mooned" -- exactly the parametric look-ahead that
   sinks LLM-trading backtests. The model scores the *text*, never the identity.
2. **Deterministic disk cache** keyed on a hash of (model, schema, sanitized text). A
   given filing is scored once; re-runs and backtests read the cache, so the feature is
   reproducible and the API bill is paid once. The LLM never sizes a trade -- it only
   emits a feature that must still earn its place in :mod:`bsealpha.validation`.

The live path uses the Anthropic SDK (``claude-opus-5`` by default -- pass ``model=`` to
use a cheaper model such as ``claude-haiku-4-5`` for bulk runs). :func:`stub_extract_frame`
is a zero-network deterministic fallback so the whole pipeline (and CI) runs offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Bump when the prompt/schema changes so stale cache entries are not reused.
SCHEMA_VERSION = "v1"
DEFAULT_MODEL = "claude-opus-5"

_SCORES = ("direction", "materiality", "surprise", "novelty")

# JSON-schema for structured outputs. Numeric range constraints (min/max) are not
# supported by structured outputs, so ranges are stated in the prompt and clipped below.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "number"},
        "materiality": {"type": "number"},
        "surprise": {"type": "number"},
        "novelty": {"type": "number"},
    },
    "required": list(_SCORES),
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are a financial disclosure analyst. You receive the text of a single corporate "
    "filing with the company's identity removed. Judge only the text; never guess which "
    "company or date it is, and do not use any outside knowledge of specific companies. "
    "Return four numbers:\n"
    "- direction: signed expected short-term share-price impact, from -1.0 (very negative) "
    "to +1.0 (very positive); 0.0 if neutral or unclear.\n"
    "- materiality: how market-moving this is, 0.0 (routine/noise) to 1.0 (major).\n"
    "- surprise: how unexpected versus a reasonable prior, 0.0 (fully expected) to 1.0.\n"
    "- novelty: how much genuinely new information it carries, 0.0 (procedural/duplicate) "
    "to 1.0 (substantive new fact).\n"
    "Judge conservatively: most filings are routine and should score low on materiality, "
    "surprise, and novelty."
)

# "Company Ltd - 500227 - <subject>" -> keep only <subject>
_BSE_PREFIX = re.compile(r"^\s*.+?\s-\s\d{4,7}\s-\s")


def sanitize_text(headline: str, body: str, scrip_code: int) -> str:
    """Strip company identity from the announcement text before the model sees it.

    Removes the "Name - <scrip> - " prefix BSE prepends, any standalone occurrence of the
    scrip code, and collapses whitespace. Leak control, not cosmetics (see module docstring).
    """
    text = (headline or "").strip()
    text = _BSE_PREFIX.sub("", text)
    if body:
        text = f"{text}\n\n{body.strip()}"
    # drop bare scrip-code numbers anywhere they survived
    text = re.sub(rf"\b{scrip_code}\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _cache_key(model: str, sanitized: str) -> str:
    h = hashlib.sha256(f"{model}\0{SCHEMA_VERSION}\0{sanitized}".encode()).hexdigest()
    return h[:32]


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _normalize_scores(raw: dict) -> dict:
    """Clip model output into the documented ranges (structured outputs can't enforce them)."""
    return {
        "direction": float(min(1.0, max(-1.0, float(raw.get("direction", 0.0))))),
        "materiality": _clip01(float(raw.get("materiality", 0.0))),
        "surprise": _clip01(float(raw.get("surprise", 0.0))),
        "novelty": _clip01(float(raw.get("novelty", 0.0))),
    }


@dataclass
class ExtractionCache:
    """On-disk cache of extractions: one JSON file per content hash under ``dir``.

    The cache is the reproducibility contract -- a filing is scored once and every later
    run (including backtests) reads the same numbers, so the LLM feature is a deterministic
    function of the text, exactly like every other feature on the grid.
    """

    dir: Path

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        f = self.dir / f"{key}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def put(self, key: str, scores: dict) -> None:
        (self.dir / f"{key}.json").write_text(json.dumps(scores))


class AnnouncementExtractor:
    """Score announcements with Claude, cached and identity-stripped.

    Parameters
    ----------
    cache_dir:
        Directory for the deterministic extraction cache.
    model:
        Anthropic model id. Defaults to ``claude-opus-5``; for bulk historical runs pass a
        cheaper model (e.g. ``claude-haiku-4-5``) -- the score is advisory and must still
        clear the validation gate, so the model is a cost lever, not a correctness one.
    effort:
        Reasoning effort for the extraction (``low`` is plenty for a scoped scoring task).
    """

    def __init__(self, cache_dir: str | Path, *, model: str = DEFAULT_MODEL,
                 effort: str = "low") -> None:
        self.cache = ExtractionCache(Path(cache_dir))
        self.model = model
        self.effort = effort
        self._client = None  # lazily constructed so importing needs no anthropic/API key

    def _api(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _extract_one(self, sanitized: str) -> dict:
        """One cached extraction. Refusals/parse errors -> neutral zeros (never crash a batch)."""
        key = _cache_key(self.model, sanitized)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        resp = self._api().messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            messages=[{"role": "user", "content": sanitized or "(empty filing)"}],
        )
        if resp.stop_reason == "refusal":
            scores = {k: 0.0 for k in _SCORES} | {"_refused": True}
        else:
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            scores = _normalize_scores(json.loads(text))
        self.cache.put(key, scores)
        return scores

    def extract_frame(self, anns: pl.DataFrame, *, progress: bool = False) -> pl.DataFrame:
        """Return ``(ann_id, direction, materiality, surprise, novelty)`` for each row."""
        rows = []
        n = anns.height
        for i, r in enumerate(anns.iter_rows(named=True)):
            sanitized = sanitize_text(r["headline"], r.get("body", ""), r["scrip_code"])
            s = self._extract_one(sanitized)
            rows.append({"ann_id": r["ann_id"], **{k: s.get(k, 0.0) for k in _SCORES}})
            if progress and (i % 100 == 0 or i == n - 1):
                print(f"  extracted {i + 1}/{n}", flush=True)
        return _scores_frame(rows)


def _scores_frame(rows: list[dict]) -> pl.DataFrame:
    schema = {"ann_id": pl.Utf8, **{k: pl.Float64 for k in _SCORES}}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def stub_extract_frame(anns: pl.DataFrame, *, seed: int = 0) -> pl.DataFrame:
    """Deterministic offline extractor (no API) for CI and the synthetic pipeline.

    Derives plausible-looking scores from a hash of the *sanitized* text so the output is
    stable and identity-free, mirroring the price layer's synthetic generator. Crucially it
    embeds **no real signal** -- Phase 4 must still show the feature adds nothing here, which
    is the honest baseline (a stub that faked predictive power would defeat the whole test).
    """
    rows = []
    for r in anns.iter_rows(named=True):
        sanitized = sanitize_text(r["headline"], r.get("body", ""), r["scrip_code"])
        h = int(hashlib.sha256(f"{seed}\0{sanitized}".encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        rows.append({
            "ann_id": r["ann_id"],
            "direction": float(rng.uniform(-1.0, 1.0)),
            "materiality": _clip01(float(rng.beta(1.5, 4.0))),   # skewed low, like reality
            "surprise": _clip01(float(rng.beta(1.5, 4.0))),
            "novelty": _clip01(float(rng.beta(2.0, 3.0))),
        })
    return _scores_frame(rows)
