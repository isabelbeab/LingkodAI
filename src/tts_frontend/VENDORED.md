# Vendored TTS text-frontend files

These nine files are copied verbatim from HK's TTS work on JOJIE (see
`reference/TTS files/scripts/` for the originals). Never edit them; a
workaround belongs in a new file, same as any other "do not edit" module in
this repo. They import each other by top-level name, so callers put
`src/tts_frontend/` on `sys.path` (see `src/tts.py`) rather than rewriting
imports.

No original copy date was recorded when these were first vendored. The
table below records the md5 as of 2026-09-21, the date this file was filled
in, as the current known-good baseline: if any of these hashes ever change
from what a future `md5sum` reports, that file was edited and should be
reverted or the change flagged, not silently kept.

| File | md5 | recorded |
|---|---|---|
| `shared_domain_frontend.py`   | `9cccaaff9b0116ce1d691e0dc9e17554` | 2026-09-21 |
| `ceb_respell_lab.py`          | `427c892a3d0099465dca2c5a5e17e1e7` | 2026-09-21 |
| `ceb_segmentation.py`         | `f15e2c6d79d85d5a8034201720eaaf6c` | 2026-09-21 |
| `ceb_prc_digit_verbalizer.py` | `eecf312b667a16e58319cb7356edee81` | 2026-09-21 |
| `ceb_full_pipeline.py`        | `e69eb564474f89f6c81887c310b0b9f0` | 2026-09-21 |
| `fil_prc_digit_verbalizer.py` | `ae52f9a5199726cbceb7e67dbe23ce25` | 2026-09-21 |
| `eng_prc_digit_verbalizer.py` | `74490e9df871d0fb45bd32b693c9550f` | 2026-09-21 |
| `lingkod_tts_utils.py`        | `63ef5eb1af469ad5f5d242f9f792a862` | 2026-09-21 |
| `lingkod_stt_utils.py`        | `5154c9e2c34090817bdd6159e8276823` | 2026-09-21 |

All nine required, or imports fail: `fil_prc_digit_verbalizer` calls
`tts._require_stt()` at import time (`tts` here is `lingkod_tts_utils`,
imported under that alias).
