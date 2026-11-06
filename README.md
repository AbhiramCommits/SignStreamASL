# SignStreamASL

**End-to-end American Sign Language recognition that runs on a laptop CPU.**
Sign language is a first-class language, but most speech and vision tooling treats
it as an afterthought. SignStreamASL is a complete, reproducible reference pipeline
for the full product arc: capturing and curating signing video, extracting a compact
61-landmark body-relative representation, training a temporal recognizer, measuring
signer-level fairness, decoding continuous signing into timestamped gloss events,
and serving the model behind a low-latency API with ONNX Runtime.

The repository deliberately ships with a deterministic **procedural ASL corpus**
(60 glosses, 14 signers, 5,040 takes) so every number below can be regenerated from
a clean clone without a multi-gigabyte download. The same keypoint schema accepts
real WLASL/How2Sign clips or webcam recordings through the optional MediaPipe path.

---

## Architecture

```text
 video / webcam
       |
       v
 MediaPipe Holistic  ──>  61 landmarks (21 L hand + 21 R hand + 19 pose)
       |                         |
       |                    normalize: shoulder-midpoint origin,
       |                    shoulder-width scale, resample to 48 frames
       v                         |
 .npz clips  ──>  curation ──> versioned manifest_v1.json + splits
                  │ dedupe / quality / label audit
                  v
        PoseSequenceTransformer (183 -> 128, 4 pre-norm layers)
                  │
        ┌─────────┴──────────┐
        v                    v
 evaluation + fairness    sliding-window decoder
 (top-1/5, ECE, gaps,     (48 f window, stride 8, EMA,
  confusion, gold gate)    threshold + <ABSTAIN>, NMS)
        │                    │
        └─────────┬──────────┘
                  v
       FastAPI + micro-batcher + ONNX Runtime
```

| Module | Role |
|---|---|
| `signstream.data.keypoints` | 61-landmark schema, normalization, resampling, MediaPipe extraction |
| `signstream.data.corpus` | deterministic ASL-structured corpus with injected defects |
| `signstream.data.curate` | perceptual-hash dedupe, quality filter, model-free label audit |
| `signstream.data.manifest` / `splits` | versioned manifest, content hashes, signer-safe splits |
| `signstream.models` | pose transformer, BiGRU ablation, pooled linear probe |
| `signstream.train` | CPU training, cosine schedule, early stopping, MLflow + JSONL |
| `signstream.eval` | metrics, calibration, per-signer fairness, frozen gold gate |
| `signstream.stream` | continuous sentence construction and sliding-window gloss decoder |
| `signstream.serve` | Torch/ONNX backends, async micro-batcher, FastAPI endpoints |
| `signstream.export` / `bench` | dynamic-axis ONNX export, parity check, latency benchmarks |

---

## Dataset

### 61-landmark schema

Every clip is a float16 array shaped `[T, 61, 3]`, flattened to `[T, 183]` for the
model:

- 0–20: left hand (wrist + 20 finger joints, MediaPipe hand topology)
- 21–41: right hand
- 42–60: upper-body pose (nose, eyes, ears, shoulders, elbows, wrists, pinky/index/thumb,
  hips), mapped from MediaPipe Pose indices
- Normalization: subtract the shoulder midpoint and divide by Euclidean shoulder width.
  Body size and camera distance are removed before the model sees a clip.
- Missing hands are NaN-filled then forward/backward filled; all-NaN tracking failures
  remain visible to curation.

### The procedural corpus and why it exists

Real ASL corpora require large manual downloads, licences, and preprocessing before a
single experiment is possible. A clean clone that cannot reproduce a single number is
not a product. SignStreamASL therefore generates a deterministic, physically
structured corpus from a SHA-256 hash of each gloss:

- **60 glosses**: HELLO, THANK-YOU, HELP, WATER, NAME, LEARN, ... WHO.
- **14 signers**, each with persistent body scale, signing speed 0.75–1.30×, spatial
  amplitude, articulation noise, and capture-quality tier.
- **6 takes per (gloss, signer)** → **5,040 clips**. Each gloss has 3–6 keyframe
  waypoints, a movement path (arc / straight / circular / repeated-contact), a
  two-handed flag, and a 25–60-frame prototype.
- **S11 and S12 are LOW capture quality**: 3× landmark noise, 15% random frame
  dropouts, and occasional single-hand tracking loss.
- **S04 and S13 are left-dominant**; their landmark x-axis is mirrored.
- Injected curation defects: ~2% near-duplicate takes, ~1.5% label swaps to a
  confusable gloss, ~1% truncated clips (<12 frames), ~0.5% all-NaN tracking failures.

The curated result is **4,718 kept clips** from 5,040 (see Results). The manifest
records per-gloss and per-signer counts, content hashes, quality tier, handedness,
curation drops, and split definitions.

### Using real signing video

The capture path writes exactly the same schema:

```bash
signstream extract-keypoints \
  --videos-dir /path/to/wlasl_subset_or_webcam_recording \
  --out artifacts/data/keypoints

signstream extract-keypoints --videos-dir ./my_clips --out artifacts/data/keypoints
```

Each clip becomes `<name>.npz` with `keypoints[T,61,3]`, `valid_length`, plus
`ingest_log.json`. Point the extractor at a downloaded WLASL/How2Sign subset or a
webcam recording; the keys have the same body-relative, 61-landmark layout the
trainer consumes. Label metadata still has to be supplied for supervised training;
the procedural corpus is the turnkey path, real clips are the extension point.
`decode --video` reuses the same extractor directly.

---

## How to run

```bash
# Python 3.11 is required
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional capture path (MediaPipe + OpenCV), graceful if unavailable
pip install ".[capture]"

# Full reproducible pipeline
make all
```

Equivalent individual commands:

```bash
signstream build-data --config configs/data.yaml
signstream build-data --coverage-only

signstream train --model pose_transformer --split signer_holdout
signstream train --model pose_transformer --split random
signstream train --model bigru --split signer_holdout
signstream train --model bigru --split random
signstream train --model linear_probe --split signer_holdout
signstream train --model linear_probe --split random
signstream label-budget --model pose_transformer

signstream evaluate --checkpoint artifacts/checkpoints/pose_transformer_signer_holdout.pt --split signer_holdout
signstream evaluate --checkpoint artifacts/checkpoints/pose_transformer_random.pt --split random
signstream evaluate --checkpoint artifacts/checkpoints/bigru_signer_holdout.pt --split signer_holdout
signstream decode --input artifacts/data/continuous/ --threshold 0.6
signstream regress

signstream export-onnx --checkpoint artifacts/checkpoints/pose_transformer_signer_holdout.pt
signstream bench --all
signstream serve --port 8000
```

### API

```bash
SIGNSTREAM_BACKEND=torch signstream serve --port 8000
# or after export-onnx:
SIGNSTREAM_BACKEND=onnx signstream serve --port 8000
```

Real captured responses:

```text
GET /health
{"status":"ok","model":"pose_transformer","backend":"torch","class_count":60,
 "manifest_hash":"af1b378a9531b50d5e83ef489532ad684ee896a83248afa99c86c9941d15224b",
 "uptime_s":3.541,"version":"0.1.0"}
```

```text
POST /predict   {"sequence": [[[x,y,z] × 61] × 48]}
200 OK
{"top5":[
  {"gloss":"HELLO","confidence":0.9195002317428589},
  {"gloss":"PLAY","confidence":0.0231526680290699},
  {"gloss":"EVENING","confidence":0.009803620167076588},
  {"gloss":"SAD","confidence":0.009438215754926205},
  {"gloss":"GO","confidence":0.004693842492997646}],
 "abstain":false,"threshold":0.6,"model":"pose_transformer","backend":"torch"}
```

- `POST /predict/batch` accepts up to 32 sequences and demonstrates the async
  micro-batcher; the captured run returned two predictions with status 200.
- A malformed landmark (`[0.1, 0.2]` instead of three coordinates) returns **422**.
- `POST /predict/video` accepts multipart video, extracts MediaPipe landmarks, and
  returns timestamped gloss events plus `<ABSTAIN>` semantics. It returns 503 with a
  clear message when capture dependencies are absent.
- `GET /metrics` is Prometheus-style and includes request count, p50/p95/p99 latency,
  abstain rate, and a batch-size histogram. The captured health/predict/batch/metrics
  outputs are in `artifacts/reports/api_examples/`.

### Docker

```bash
docker build -t signstream:local .
docker run --rm -p 8000:8000 -e SIGNSTREAM_BACKEND=torch signstream:local
# or
docker compose up --build
```

The image installs the package, copies the frozen checkpoint/ONNX fixtures, and runs
`uvicorn signstream.serve.app:app`. Verified in this build: the image built successfully
and, with `docker run -p 8013:8000`, both `GET /health` and `POST /predict` returned
HTTP 200 (`artifacts/reports/api_examples/docker_*.json`).

---

## Results

All figures below are copied from generated JSON/Markdown artifacts. They were
produced on the CPU hardware listed under **Reproducibility**, not estimated.

### Isolated-sign accuracy

Split definitions: **random** is a seeded 70/15/15 stratified clip split; **signer
holdout** holds out S12, S13, S14 entirely for test, with S11 + S10 for validation.

| Model | Split | Top-1 | Top-5 | Macro F1 |
|---|---|---:|---:|---:|
| PoseSequenceTransformer | random | **0.837** | **0.997** | **0.833** |
| BiGRU | random | 0.816 | 0.984 | 0.812 |
| Pooled linear probe | random | **0.868** | **0.993** | **0.862** |
| PoseSequenceTransformer | signer holdout | **0.699** | **0.943** | **0.707** |
| BiGRU | signer holdout | 0.693 | 0.933 | 0.698 |
| Pooled linear probe | signer holdout | 0.506 | 0.777 | 0.536 |

The random → unseen-signer drop is **13.8 points** for the transformer, **12.3** for
BiGRU, and **36.2** for the linear probe. The transformer beats both baselines on the
signer-holdout test by top-1, top-5, and macro F1, but the margin over BiGRU is only
0.6 points; the much cheaper linear probe actually wins the random split. This is an
honest limitation of a 60-class procedural corpus: models that capture local motion
(BiGRU) are strong, and the transformer's advantage appears mainly under signer shift.

### Signer fairness

From `artifacts/reports/eval_pose_transformer_signer_holdout.md`:

| Held-out signer | Quality | Handedness | N | Top-1 | Top-5 | Abstain |
|---|---|---|---:|---:|---:|---:|
| S12 | LOW | right-dominant | 341 | **0.909** | 1.000 | 0.384 |
| S13 | HIGH | left-dominant | 318 | **0.305** | 0.824 | 0.890 |
| S14 | HIGH | right-dominant | 337 | **0.858** | 0.997 | 0.424 |

- **Fairness gap (best − worst signer top-1): 0.604.**
- HIGH tier pooled top-1: 0.589; LOW tier: 0.909 → **HIGH − LOW gap = −0.320**.
- Left-dominant signer top-1: 0.305; right-dominant: 0.883 → **handedness gap = 0.578**.

The dominant fairness risk in this corpus is **handedness**, not the injected capture
noise: the only left-dominant held-out signer collapses to 30.5% while both
right-dominant signers exceed 85%. The LOW-quality signer S12 is actually more
accurate because its extra noise and dropout act as augmentation and it shares a
right-dominant style with the training pool. For real deployment this says: do not
audit only capture quality; recruit and evaluate left-dominant signers explicitly,
and treat per-signer abstention (S13 abstains on 89% of clips) as a product failure,
not a rounding error.

### Curation and label audit

| Quantity | Real value |
|---|---:|
| Clips generated | 5,040 |
| Clips kept | 4,718 |
| Duplicate drops | 105 |
| Too-few-frames drops | 50 |
| Excess-NaN drops | 21 |
| No-motion drops | 0 |
| Suspected-mislabel drops | 146 |
| Label-audit precision | **0.473** |
| Label-audit recall | **0.908** |
| Injected mislabels recovered | 69 / 76 |

Coverage after curation: 60 glosses, 14 signers, clips per gloss **min 73 / median 79
/ max 84**. The label audit's top-3% disagreement flagging recovers 91% of injected
swaps at 47% precision, which is useful triage rather than an oracle.

### Label budget

Transformer retrained from scratch at each fraction of training labels; test set is
the signer-holdout test split.

| Labels | Training clips | Epochs | Test top-1 | Test top-5 |
|---:|---:|---:|---:|---:|
| 5% | 156 | 4 | 0.053 | 0.127 |
| 10% | 300 | 4 | 0.040 | 0.185 |
| 25% | 759 | 4 | 0.088 | 0.233 |
| 50% | 1,522 | 4 | 0.084 | 0.230 |
| 100% | 3,047 | 5 | **0.193** | **0.539** |

These are deliberately short runs (4–5 epochs) so the whole sweep stays in minutes.
The curve is noisy at very low fractions but shows the expected large gain at 100%;
the full-budget model trained for 20 epochs reaches 0.699.

### Continuous decoding

150 synthetic sentences, 4–8 held-out signs each, transition/rest frames, **908 gold
signs**. Matching requires the correct gloss and a predicted midpoint inside the gold
span.

| Threshold | Precision | Recall | F1 | Gloss error rate | Abstain rate |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.354 | 0.561 | 0.434 | 1.051 | 0.501 |
| 0.5 | 0.549 | 0.248 | 0.341 | 0.763 | 0.886 |
| **0.6** | **0.618** | **0.118** | **0.198** | **0.878** | **0.955** |
| 0.7 | 0.553 | 0.029 | 0.054 | 0.966 | 0.986 |
| 0.8 | 0.800 | 0.009 | 0.017 | 0.989 | 0.997 |

At the requested 0.6 threshold the decoder emits 173 events (107 matches), with
**95.5% of windows abstaining**. Lowering to 0.3 more than doubles recall at the cost
of many insertion errors. This is the honest isolated-to-continuous gap: the model
was trained on centered isolated signs, not segmented continuous signing.

### Latency and real-time viability

50 warmup + 300 timed iterations per configuration; batch shapes `[B,48,183]`.

| Backend | Batch | p50 ms | p95 ms | p99 ms | Clips/s | Per-clip ms |
|---|---:|---:|---:|---:|---:|---:|
| Torch | 1 | 1.938 | 2.324 | 2.554 | 516.0 | 1.938 |
| Torch | 8 | 4.376 | 4.946 | 5.675 | 1,828.2 | 0.547 |
| Torch | 32 | **23.506** | **27.804** | **33.350** | 1,361.3 | 0.735 |
| ONNX Runtime | 1 | 1.118 | 2.982 | 4.932 | 894.7 | 1.118 |
| ONNX Runtime | 8 | 7.502 | 20.336 | 29.463 | 1,066.5 | 0.938 |
| ONNX Runtime | 32 | 24.578 | 63.670 | 86.967 | 1,302.0 | 0.768 |

Streaming path (ONNX, 48-frame window, stride 8): **p50 1.205 ms/window**, p95 6.051
ms/window, 829.9 windows/s, implied **6,639 FPS** ceiling. At 30 FPS only 3.75 windows/s
are required, so the serving path is comfortably **real-time viable**. The ceiling is
model-only; capture and MediaPipe extraction are additional costs. On this run Torch
had lower p50 at batch 1 and 32; ONNX won batch 8. Both are well inside the streaming
budget, and the report records the real spread rather than picking one lucky number.

### ONNX parity

| Metric | Value |
|---|---:|
| Parity clips | 500 |
| Max absolute logit delta | **6.20e-06** |
| Mean absolute logit delta | **6.32e-07** |
| Argmax agreement | **100.0%** |
| ONNX size | 2,416,116 bytes |
| PyTorch checkpoint size | 2,330,911 bytes |

### Tests and CI

- **90 tests total** (88 fast + 2 `slow`); **84% statement coverage** on the CI
  non-slow suite and **87% including slow tests**
  (`pytest -q -m "not slow" --cov=src/signstream --cov-report=term-missing`).
- GitHub Actions runs `ruff check`, the non-slow suite with `--cov-fail-under=80`,
  and a second regression job that rebuilds the deterministic data, restores the
  frozen checkpoint fixture, and runs `signstream regress`.
- Regression gate baseline from the real run: gold top-1 **0.667**, top-5 **0.917**,
  macro F1 **0.677**, max fairness gap **0.604**. Tolerances are ±0.02 absolute for
  accuracy/F1 and +0.05 for the fairness gap (the wrong direction fails the gate).

---

## Limitations & next steps

The procedural corpus is a **stand-in for real signing video**, not a substitute.
Its motions are generated prototypes, so the reported accuracies measure whether the
pipeline can learn a controlled visual language, not ASL competence. A model trained
here will encounter a large domain shift on WLASL/How2Sign/real webcam footage:
backgrounds, occlusions, different body proportions, camera motion, and signing
styles all change.

The 60 glosses are **isolated lexical signs**, not ASL grammar. There are no
non-manual markers (facial expression, mouthing, brow position), classifiers,
spatial verbs, role shift, or syntax. The continuous decoder stitches isolated signs
together; it is not a sign-language translation system. The continuous F1 at the
default threshold (0.198) and the 95.5% abstain rate show exactly where the next
modeling work belongs.

Next steps that would move this from a strong reference implementation toward a
product: fine-tune on real WLASL and How2Sign data; replace isolated classification
plus NMS with CTC or a seq2seq/Transformer decoder trained on continuous signing;
add non-manual features from face/head landmarks; add signer-adaptive calibration and
per-signer abstention dashboards; and recruit a substantially broader signer pool —
especially left-dominant and low-resource signers — before claiming fairness. The
handedness gap (0.578) is actionable: mirror augmentation and left-handed recruitment
should be measured, not assumed.

---

## Reproducibility

- Data seed **42**, model seed **7**, label-budget seed **11**, evaluation seed **99**.
- Dataset version **v1**, manifest hash
  **`af1b378a9531b50d5e83ef489532ad684ee896a83248afa99c86c9941d15224b`**.
- Frozen gold set: 300 clips, content hash
  **`a16960a9a9efbc39f2898a2b17fe9df60b1e72dfa7a242590012edbcde41332f`**.
- Every run writes `artifacts/runs/<run_id>/metrics.jsonl` and local MLflow data under
  `mlruns/`; reports are committed under `artifacts/reports/`.
- Full reproduction: `make all` (data → 6 model runs → label budget → eval → decode →
  ONNX → bench → tests → regression gate).
- **Verified on:** macOS 27.2, Apple arm64, 8 logical CPUs, 16 GB RAM, Python 3.11.14,
  PyTorch 2.15.0, NumPy 1.26.4, ONNX Runtime 1.31.1. Docker build/run was verified in
  this environment (see the Docker section); if Docker is unavailable on another
  machine, all non-container numbers remain reproducible with `make all`.
