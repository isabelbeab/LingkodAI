# input/

Put recordings here for `scripts/run_folder.py`; the results appear in
`output/`. Everything in this folder except this file is gitignored.

- An audio file placed directly here is a one-question conversation:
  `input/renew_id.m4a` gives `output/renew_id/`.
- A subfolder is a multi-turn conversation. Its audio files are the turns,
  in filename order, so number them: `input/fees/01_ask.m4a`,
  `input/fees/02_followup.m4a` give `output/fees/`.

Any format ffmpeg reads works (wav, m4a, mp3, aac, ...). Each recording
should be one question in Cebuano, Filipino or English. Then, from the repo
root:

```bash
python scripts/run_folder.py
```

See the README, "Running your own recordings".
