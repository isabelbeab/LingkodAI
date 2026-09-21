"""CPU-only smoke test for src/mt.py. Never loads the real NLLB model or
touches the network -- a FakeMTTokenizer/FakeMTModel pair stands in for the
transformers objects MTBundle wraps."""

from __future__ import annotations

import numpy as np
import pytest

from src import mt


class FakeBatch(dict):
    """Stands in for a transformers BatchEncoding: dict-like, with .to()."""

    def to(self, device):
        return self


class FakeMTTokenizer:
    def __init__(self, translate_fn=None):
        self.src_lang = None
        self.tokenize_calls = []
        self.convert_calls = []
        self._translate_fn = translate_fn or (lambda t: f"EN[{t}]")
        self._last_texts = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        self.tokenize_calls.append(
            {"padding": padding, "truncation": truncation, "n_texts": len(texts), "src_lang": self.src_lang}
        )
        self._last_texts = list(texts)
        return FakeBatch(input_ids=np.zeros((len(texts), 3), dtype=np.int64))

    def convert_tokens_to_ids(self, token):
        self.convert_calls.append(token)
        return 7

    def batch_decode(self, outputs, skip_special_tokens=True):
        return [self._translate_fn(t) for t in self._last_texts]


class FakeMTModel:
    """call_plan: list of None (succeed) or an Exception instance to raise,
    consumed in order across .generate() calls. Once exhausted, calls succeed."""

    def __init__(self, call_plan=None):
        self.generate_calls = []
        self._call_plan = list(call_plan) if call_plan is not None else []

    def generate(self, **kwargs):
        self.generate_calls.append({k: v for k, v in kwargs.items() if k != "input_ids"})
        if self._call_plan:
            behavior = self._call_plan.pop(0)
            if behavior is not None:
                raise behavior
        return "OUTPUTS_PLACEHOLDER"


def make_bundle(call_plan=None, translate_fn=None) -> mt.MTBundle:
    return mt.MTBundle(
        tokenizer=FakeMTTokenizer(translate_fn=translate_fn),
        model=FakeMTModel(call_plan=call_plan),
        device="cpu",
    )


# --- split_into_chunks ---------------------------------------------------


def test_split_into_chunks_never_splits_mid_sentence():
    text = "First sentence. Second sentence! Third sentence?"
    chunks = mt.split_into_chunks(text, max_chars=1000)
    assert chunks == [text]


def test_split_into_chunks_packs_short_sentences_together():
    text = "One. Two. This is a much longer sentence that exceeds the budget alone."
    chunks = mt.split_into_chunks(text, max_chars=12)
    # "One." and "Two." both fit together under the 12-char budget and are
    # packed into a single chunk; the long sentence gets its own chunk even
    # though it alone exceeds max_chars -- a chunk is never split mid-sentence.
    assert chunks == [
        "One. Two.",
        "This is a much longer sentence that exceeds the budget alone.",
    ]


def test_split_into_chunks_empty_text_returns_single_chunk():
    assert mt.split_into_chunks("") == [""]


# --- eng passthrough -------------------------------------------------------


def test_eng_passthrough_to_english_no_model_call():
    bundle = make_bundle()
    items = [mt.TurnText(id=1, text="hello there", lang="eng")]
    results = mt.translate_to_english(bundle, items)
    assert results[1].text == "hello there"
    assert results[1].n_chunks == 1
    assert results[1].translate_time_sec == 0.0
    assert results[1].ok is True
    assert bundle.model.generate_calls == []


def test_eng_passthrough_from_english_no_model_call():
    bundle = make_bundle()
    items = [mt.TurnText(id=1, text="hello there", lang="eng")]
    results = mt.translate_from_english(bundle, items)
    assert results[1].text == "hello there"
    assert bundle.model.generate_calls == []


# --- successful batch translate --------------------------------------------


def test_translate_to_english_reassembles_chunks_in_order():
    bundle = make_bundle(translate_fn=lambda t: f"EN[{t}]")
    # Long enough that split_into_chunks (800-char default) actually produces
    # more than one chunk for item "a", so this test exercises reassembly
    # across chunk boundaries, not just a single pass-through chunk.
    long_text = ("Alpha sentence. " * 30 + "Beta sentence. " * 30).strip()
    items = [
        mt.TurnText(id="a", text=long_text, lang="ceb"),
        mt.TurnText(id="b", text="Different text here.", lang="ceb"),
    ]
    results = mt.translate_to_english(bundle, items)

    expected_chunks = mt.split_into_chunks(long_text)
    assert len(expected_chunks) >= 2  # sanity check the fixture text is actually multi-chunk
    expected_text = " ".join(f"EN[{c}]" for c in expected_chunks)

    assert results["a"].ok is True
    assert results["a"].n_chunks == len(expected_chunks)
    assert results["a"].text == expected_text

    assert results["b"].ok is True
    assert results["b"].text == "EN[Different text here.]"
    assert results["b"].n_chunks == 1


# --- whole-batch failure retried per chunk ----------------------------------


def test_whole_batch_failure_falls_back_to_per_chunk():
    bundle = make_bundle(call_plan=[RuntimeError("simulated batch failure")])
    items = [mt.TurnText(id="a", text="Only one sentence here.", lang="fil")]
    results = mt.translate_to_english(bundle, items)

    assert results["a"].ok is True
    assert results["a"].text == "EN[Only one sentence here.]"
    # first call is the failed batch call, second is the successful per-chunk retry
    assert len(bundle.model.generate_calls) == 2


# --- per-chunk retry also fails ---------------------------------------------


def test_chunk_failure_after_retry_marks_item_failed():
    bundle = make_bundle(
        call_plan=[
            RuntimeError("batch boom"),  # the whole-batch attempt
            None,  # per-chunk retry for item "a"'s single chunk succeeds
            RuntimeError("chunk boom"),  # per-chunk retry for item "b"'s single chunk fails
        ]
    )
    items = [
        mt.TurnText(id="a", text="First item text.", lang="ceb"),
        mt.TurnText(id="b", text="Second item text.", lang="ceb"),
    ]
    results = mt.translate_to_english(bundle, items)

    assert results["a"].ok is True
    assert results["b"].ok is False
    assert "chunk boom" in results["b"].failure_reason
    assert results["b"].text is None


# --- language code mapping and direction -----------------------------------


def test_lang_codes_mapping():
    assert mt.LANG_CODES == {"ceb": "ceb_Latn", "fil": "tgl_Latn", "eng": "eng_Latn"}


def test_translate_to_english_uses_native_src_and_eng_tgt():
    bundle = make_bundle()
    items = [mt.TurnText(id="a", text="Usa ka pangutana.", lang="ceb")]
    mt.translate_to_english(bundle, items)

    assert bundle.tokenizer.tokenize_calls[0]["src_lang"] == "ceb_Latn"
    assert bundle.tokenizer.convert_calls[0] == "eng_Latn"


def test_translate_from_english_uses_eng_src_and_native_tgt():
    bundle = make_bundle()
    items = [mt.TurnText(id="a", text="A question.", lang="fil")]
    mt.translate_from_english(bundle, items)

    assert bundle.tokenizer.tokenize_calls[0]["src_lang"] == "eng_Latn"
    assert bundle.tokenizer.convert_calls[0] == "tgl_Latn"


def test_unknown_lang_raises():
    bundle = make_bundle()
    items = [mt.TurnText(id="a", text="text", lang="xyz")]
    with pytest.raises(ValueError):
        mt.translate_to_english(bundle, items)


# --- tokenizer and generate() call shape ------------------------------------


def test_tokenizer_called_with_padding_and_truncation():
    bundle = make_bundle()
    items = [mt.TurnText(id="a", text="Some text.", lang="ceb")]
    mt.translate_to_english(bundle, items)

    call = bundle.tokenizer.tokenize_calls[0]
    assert call["padding"] is True
    assert call["truncation"] is True


def test_generate_receives_only_forced_bos_and_max_new_tokens():
    bundle = make_bundle()
    items = [mt.TurnText(id="a", text="Some text.", lang="ceb")]
    mt.translate_to_english(bundle, items)

    call = bundle.model.generate_calls[0]
    assert set(call.keys()) == {"forced_bos_token_id", "max_new_tokens"}
    assert call["max_new_tokens"] == mt.MAX_NEW_TOKENS
