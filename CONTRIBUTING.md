# Contributing

Thanks for helping make ASL technology more accessible.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install ".[capture]"   # optional: MediaPipe + OpenCV
make data
make test
```

Keep all tunables in `configs/*.yaml`. Add a test for every behavior change, run
`ruff check src tests`, and run `pytest -q -m "not slow"` before opening a PR.
Regenerate the gold-set report with `signstream regress` when changing model or
data behavior; do not silently loosen `configs/thresholds.yaml`.

Please use synthetic/procedural data or properly licensed ASL video only. Do not
commit participant-identifying data or models trained on unlicensed corpora.
