# qwentts.cpp patches

`scripts/build_qwentts.sh` clones [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)
at `0cbde9b5d21aa2142efa5d62e14b97f197ac6f6d` (the commit used for Track A,
ggml submodule `0af0d7d5f66a6976b259b292cb4e7dc60457aa45`) and applies these
patches with `git apply`, in order. The script refuses to build if the
`*.patch` files here differ from the list inside it, and records each
patch's SHA-256 in `BUILD_INFO.txt` next to the binaries.

Both patches keep upstream behaviour unless their option is set. Upstream
prebuilt images (`ghcr.io/serveurpersocom/qwentts.cpp:*`) do not carry them,
so `tts-server` is always built from source.

## 0001-icl-non-streaming-layout.patch

Upstream `qwen-tts` builds the ICL voice-clone prompt in
`generate_icl_prompt` (`qwen_tts/core/models/modeling_qwen3_tts.py`,
`qwen-tts==0.1.1`), which has two branches:

- `non_streaming_mode=False`: the text stream `[ref_text ; text ; tts_eos]`
  and the codec stream `[codec_bos ; ref_codes]` are aligned to the codec
  length and summed; leftover text rides along during generation.
- `non_streaming_mode=True`: the text stream summed with `codec_pad`, then
  the codec stream summed with `tts_pad`; trailing text is `tts_pad`.

qwentts.cpp ports only the first. LingkodAI's evaluated Stage 7 (`src/tts.py`,
the DS6/DS7 synthesis notebook) uses the second. The patch adds the second
layout behind a handle-level option:

- `qt_init_params.icl_non_streaming` (ABI 6, tail field, default `false`)
- `--icl-non-streaming` on `qwen-tts` and `tts-server`
- `--non-streaming` on `tests/debug-clone-cossim.py`, which switches both
  sides (Python `non_streaming_mode=True`, C++ `--icl-non-streaming`) so the
  project's own cossim harness can validate the layout

The server logs `ICL prompt layout: non_streaming_mode=True` at startup when
the option is on.

## 0002-talker-kv-length.patch

The talker KV cache was hardcoded to 4096 positions. LingkodAI generates up
to `max_new_tokens=4096` frames on top of the ICL prefix (a few hundred
positions, more under the `non_streaming_mode=True` layout), which overflowed
partway through a long generation. The patch adds:

- `qt_init_params.talker_ctx` (ABI 7, tail field, 0 selects 4096)
- `--talker-ctx <n>` on `qwen-tts` and `tts-server`
- an admit-time check: a request whose prompt length plus `max_new_tokens`
  exceeds the KV length fails with `QT_STATUS_INVALID_PARAMS` (HTTP 400 from
  the server) before any compute, with both numbers in the error message

Memory: the KV cache is f32 K and V for 28 layers, 8 KV heads, head dim 128,
so 224 KiB per position per batch slot. 4096 positions take 896 MiB (as
logged by Track A); 6144 take 1,344 MiB.

## Status

Compiled and checked on CPU only (2026-10-11): `tts-server`, `qwen-tts` and
`test-abi-c` build with no warnings, `test-abi-c` passes, and both binaries
list the new flags. Neither patch has run against the real model yet. The
layout is validated on Colab in `notebooks/tts_cpp_ab_colab.ipynb` (greedy
cossim against `qwen-tts` with `non_streaming_mode=True`: prompt ids exact,
logits cosine at least 0.999).
