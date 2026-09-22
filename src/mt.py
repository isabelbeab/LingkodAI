"""Stages 4 and 6 -- Machine translation, native <-> English.

facebook/nllb-200-3.3B, fp16 on CUDA and fp32 on CPU. Sentence-boundary
chunking at 800 characters, batched translation (batch size 8, chunks
flattened across turns), greedy decoding: `generate()` receives only
`forced_bos_token_id` and `max_new_tokens`, nothing else. Identical settings
both directions.

On a whole-batch failure, translation retries per chunk so partial progress
isn't lost. A turn counts as successful only if it gets back the same number
of chunks it was split into.

Ported from reference/bea_stage4_mt_multi_turn.ipynb (native -> English) and
reference/bea_stage5_backtranslate_w_lid.ipynb (English -> native), which use
an identical chunk-batch-reassemble pattern in both directions.
"""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL_NAME = "facebook/nllb-200-3.3B"
LANG_CODES = {"ceb": "ceb_Latn", "fil": "tgl_Latn", "eng": "eng_Latn"}

MT_BATCH_SIZE = 8  # tune down if you hit OOM, up if you have headroom
MAX_CHARS = 800
MAX_NEW_TOKENS = 256


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    device = device or get_device()
    return torch.float16 if device == "cuda" else torch.float32


def split_into_chunks(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on sentence boundaries, packing sentences under max_chars.

    Never splits mid-sentence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [str(text)]


@dataclass
class TurnText:
    """One item to translate: a caller-defined id, its text, and language."""

    id: object
    text: str
    lang: str


@dataclass
class MTResult:
    """Translation outcome for one TurnText."""

    id: object
    text: str | None
    n_chunks: int
    translate_time_sec: float
    ok: bool
    failure_reason: str | None = None


@dataclass
class MTBundle:
    """A loaded NLLB tokenizer and model."""

    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    device: str

    def translate_batch(self, texts: list[str], src_code: str, tgt_code: str) -> list[str]:
        """Translate one already-chunked batch. No retry here; callers retry per chunk."""
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=MAX_NEW_TOKENS
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def unload(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load(device: str | None = None) -> MTBundle:
    """Load the NLLB tokenizer and model."""
    device = device or get_device()
    dtype = get_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME, dtype=dtype).to(device).eval()
    return MTBundle(tokenizer, model, device)


def _translate_group(
    bundle: MTBundle, items: list[TurnText], src_code: str, tgt_code: str
) -> dict[object, MTResult]:
    """Chunk each item, batch-translate chunks across items, reassemble per item.

    Ported from both notebooks' identical chunk-batch-reassemble pattern:
    chunk each item's text first, then batch chunks across items together so
    short items still translate efficiently in groups.
    """
    chunk_records = []
    expected_chunks: dict[object, int] = {}
    for item in items:
        chunks = split_into_chunks(item.text)
        expected_chunks[item.id] = len(chunks)
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_records.append({"id": item.id, "chunk_idx": chunk_idx, "text": chunk_text})

    translated_chunks: dict[object, dict[int, str]] = {}
    chunk_times: dict[object, float] = {}
    failures: dict[object, str] = {}

    for i in range(0, len(chunk_records), MT_BATCH_SIZE):
        batch = chunk_records[i : i + MT_BATCH_SIZE]
        texts = [r["text"] for r in batch]
        try:
            t0 = time.time()
            translations = bundle.translate_batch(texts, src_code, tgt_code)
            per_item_time = (time.time() - t0) / len(batch)
            for rec, translated in zip(batch, translations):
                translated_chunks.setdefault(rec["id"], {})[rec["chunk_idx"]] = translated
                chunk_times[rec["id"]] = chunk_times.get(rec["id"], 0.0) + per_item_time
        except Exception:
            # whole-batch failure -- fall back to per-chunk so partial progress isn't lost
            for rec in batch:
                try:
                    t0 = time.time()
                    translated = bundle.translate_batch([rec["text"]], src_code, tgt_code)[0]
                    translated_chunks.setdefault(rec["id"], {})[rec["chunk_idx"]] = translated
                    chunk_times[rec["id"]] = chunk_times.get(rec["id"], 0.0) + (time.time() - t0)
                except Exception as ex2:
                    failures[rec["id"]] = str(ex2)[:300]

    results: dict[object, MTResult] = {}
    for item in items:
        if item.id in failures:
            results[item.id] = MTResult(item.id, None, 0, 0.0, False, failures[item.id])
            continue
        chunks_for_item = translated_chunks.get(item.id, {})
        n_expected = expected_chunks[item.id]
        if len(chunks_for_item) != n_expected:
            reason = f"only {len(chunks_for_item)}/{n_expected} chunks translated"
            results[item.id] = MTResult(item.id, None, 0, 0.0, False, reason)
            continue
        joined = " ".join(chunks_for_item[idx] for idx in sorted(chunks_for_item))
        results[item.id] = MTResult(
            item.id, joined, n_expected, chunk_times.get(item.id, 0.0), True
        )
    return results


def _validate_langs(items: list[TurnText]) -> None:
    unknown = {item.lang for item in items} - set(LANG_CODES)
    if unknown:
        raise ValueError(f"Unknown language(s) {unknown}; expected one of {list(LANG_CODES)}")


def translate_to_english(bundle: MTBundle, items: list[TurnText]) -> dict[object, MTResult]:
    """Native -> English (stage 4). `eng` items pass through unchanged, no NLLB call."""
    _validate_langs(items)
    results: dict[object, MTResult] = {}
    for item in items:
        if item.lang == "eng":
            results[item.id] = MTResult(item.id, item.text, 1, 0.0, True)

    for lang in ("ceb", "fil"):
        subset = [item for item in items if item.lang == lang]
        if not subset:
            continue
        results.update(_translate_group(bundle, subset, LANG_CODES[lang], LANG_CODES["eng"]))
    return results


def translate_from_english(bundle: MTBundle, items: list[TurnText]) -> dict[object, MTResult]:
    """English -> native (stage 6). `eng` items pass through unchanged, no NLLB call."""
    _validate_langs(items)
    results: dict[object, MTResult] = {}
    for item in items:
        if item.lang == "eng":
            results[item.id] = MTResult(item.id, item.text, 1, 0.0, True)

    for lang in ("ceb", "fil"):
        subset = [item for item in items if item.lang == lang]
        if not subset:
            continue
        results.update(_translate_group(bundle, subset, LANG_CODES["eng"], LANG_CODES[lang]))
    return results
