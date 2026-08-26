# Phase 0 — Bootstrap and de-risk the stack

Status: **complete, 5/5 gates passed**
Commits: `9512233` (skeleton + face stack), `53e23be` (visual gate)

---

## 1. What was built, and why

### The one-sentence purpose

Phase 0 contains **no pipeline code**. It exists to answer a single question before
any expensive work begins: *do the two models this project depends on actually load
and behave correctly on this machine?* If the answer were no, the plan's whole shape
would change — better to learn that on day one than on day three.

### What was actually created

| Artifact | Why it exists |
|---|---|
| `pyproject.toml` | Renamed `frameseek` → `sceneseek`, added `[build-system]`, declared the `sceneseek` CLI entrypoint. The identity mismatch was fixed *now*, before it leaked into module names and imports. |
| `src/sceneseek/config.py` | Single source of truth. No other module hard-codes a model name, path, or magic number. |
| `src/sceneseek/{acquire,enrich,retrieve,evaluate,web}/` | Empty packages that pre-declare the shape of Phases 1–6, so later work has an obvious home. |
| `scripts/spike_env.py` | The gate. Self-contained: fetches its own fixtures, runs both models, checks five conditions, exits non-zero on failure. |
| `README.md` | Setup, model table, phase checklist. |
| `data/spike/` | Five cached fixture images (gitignored). |
| `.gitignore` | Now also ignores `data/`. |

`hello.py` deleted. Three commits total.

### Why `config.py` declares thresholds as `None`

```python
FACE_SIM_THRESHOLD: float | None = None
TARGET_IDENTITY_PRECISION = 0.98
```

This is deliberate and worth understanding. The face-similarity cutoff is a number
**Phase 2 calibrates from labelled data** — it is one of the project's deliverables,
not a constant someone picks. Declaring it `None` means any code that reaches for it
before calibration crashes loudly.

A plausible-looking hardcoded `0.4` would silently work, produce results, and quietly
poison every number Phase 5 reports. The whole point of the project is to *measure*
whether identity filtering beats appearance-only ranking; a guessed threshold makes
that measurement meaningless while looking perfectly healthy.

### Why the gate uses two actors × two photos

The fixtures are five CC-licensed Wikimedia images: two of Al Pacino, two of Christoph
Waltz, one snowy lake. This specific arrangement is what makes the test meaningful.

Two photos of the *same* person give a **genuine pair**. Photos of *different* people
give **impostor pairs**. You need both. A recognizer that returns confident numbers for
everything looks fine if you only ever test it on one person — the only way to show it
separates identities is to check that same-person scores land clearly above
different-person scores.

### Why fixtures are downloaded rather than committed

The script fetches its own test images and caches them in gitignored `data/spike/`.
Re-runs are free. This keeps binaries out of git while making the gate reproducible on
any machine — the same principle Phase 1 applies at scale, where `sources.jsonl` (the
provenance manifest) is the committed artifact and the JPEGs are not.

---

## 2. Dense concepts explained

### Embeddings

Both models turn an input into a **vector** — a fixed-length list of numbers. The face
recognizer turns a cropped face into 512 numbers; the visual encoder turns a whole image
into 768 numbers. The useful property is that *semantically similar inputs land near each
other* in that space. Two photos of the same person produce nearby vectors; photos of
different people produce distant ones.

Nothing about a single vector is interpretable. Only *comparisons between* vectors mean
anything.

### Cosine similarity, and why L2 normalization matters

To compare two vectors we use **cosine similarity** — the cosine of the angle between
them. It ranges from −1 (opposite) through 0 (unrelated) to +1 (identical direction).
It measures *direction only*, ignoring magnitude, which is what we want: how brightly lit
a face is shouldn't change who it is.

The formula divides by both vectors' lengths:

```
cos(a, b) = (a · b) / (|a| × |b|)
```

If you **L2-normalize** every vector on the way out — rescale it to length exactly 1 —
then `|a| = |b| = 1` and the formula collapses to just `a · b`, a plain dot product.

This is why the plan specifies embeddings are normalized at the boundary. It means every
downstream comparison in the entire project is a single dot product, and a whole-corpus
search becomes one matrix multiply against a numpy array. It is also why ~450 frames need
no vector database.

### Genuine vs. impostor distributions

Standard vocabulary in biometrics:

- **Genuine pair** — two samples of the *same* identity. Should score high.
- **Impostor pair** — two samples of *different* identities. Should score low.

Collect many of each and you get two distributions. Their **separation** is the real
measure of a recognizer's quality, and it is what Phase 2 uses to pick a threshold: slide
a cutoff between the two distributions until you hit the target precision. That is a
calibrated, defensible number, unlike a guess.

### The `buffalo_l` model pack

`buffalo_l` is a bundle of ONNX models from InsightFace (~330 MB). Two of them are used:

- **RetinaFace `det_10g`** — *detection*. Finds faces, returns bounding boxes and a
  confidence score. Answers "where are the faces?"
- **ArcFace `w600k_r50`** — *recognition*. Takes one cropped, aligned face and returns
  a 512-d embedding. Answers "who is this?"

The pack also ships landmark and gender/age models, which this project never uses.
`config.FACE_MODULES = ("detection", "recognition")` skips loading them.

**ArcFace** is trained with an angular-margin loss — it explicitly pushes different
identities apart *in angle*, which is exactly what makes cosine similarity the right
comparison for its output.

### SigLIP vs. CLIP — and why the raw numbers look small

Both learn a shared image/text space where a picture and its caption land near each other.
The difference is the **loss function**:

- **CLIP** uses a softmax contrastive loss. Each image competes against every other image
  in the batch to match its caption. Scores are *relative to the batch*, which requires
  large batches and a lot of cross-device communication.
- **SigLIP** uses a **sigmoid** loss. Every image–text pair is judged independently as
  simply "match" or "no match." No batch-wide normalization, so it scales better and
  trains more efficiently at the same compute.

**The practical consequence for this project:** SigLIP's sigmoid training does not
produce CLIP-style high cosines. In the gate, a *correct* image–caption match scored
only **+0.129**. That is a strong match for this model, not a weak one.

So Phase 4 must **rank by relative order and never threshold on a raw cosine value.**
A rule like "reject anything below 0.3" would discard every correct answer.

### MPS, and why fp32

**MPS** (Metal Performance Shaders) is PyTorch's Apple-silicon GPU backend — the
equivalent of CUDA on this Mac. The gate explicitly verifies the model's parameters *and*
its output tensors are on `mps`, because a device mismatch can silently fall back to CPU
and simply run slower with no error.

The model runs in **fp32** (32-bit floats) rather than fp16. At 375M parameters that is
~1.5 GB, comfortable inside 24 GB, and at ~450 images throughput is irrelevant. fp16 would
halve memory for no benefit here while exposing the project to the dtype bugs that
periodically surface in `transformers` vision towers on MPS. Boring and correct beats
clever and fragile.

### ONNX Runtime and execution providers

The face models are **ONNX** files — a portable model format independent of PyTorch.
ONNX Runtime executes them via **execution providers**, its pluggable backends. On this
machine `CoreMLExecutionProvider`, `AzureExecutionProvider`, and `CPUExecutionProvider`
are available.

The project pins **CPU** deliberately. At ~450 images CPU is already fast (0.06–0.1 s per
image), and CoreML adds conversion quirks and non-determinism for no useful gain.

### `det_score` vs. bounding-box area

**This is the most important concept in Phase 0.**

`det_score` is the detector's confidence that a region *is a face*. It is **not** a
measure of whether that face is what the picture is about. A small, sharp, well-lit
background face can easily out-score the large subject in the foreground.

Bounding-box **area** is a much better proxy for "is this the subject" in a portrait or
a film still, because the camera frames what matters. See §4 for the measurement that
settled this.

### `pooler_output`

A vision transformer emits one vector *per image patch*. To get a single vector for the
whole image, those are **pooled** into one. SigLIP uses an attention-pooling head, and the
result is `pooler_output` — that is the embedding to compare.

In `transformers` 5.x, `get_image_features()` returns the whole model output object, and
`.pooler_output` must be read from it. See §4.

### The Hugging Face cache layout

Understanding this is what made the interrupted download recoverable:

```
models--google--siglip2-base-patch16-384/
  blobs/<sha256>            the real file content, named by its own hash
  snapshots/<revision>/     symlinks pointing into blobs/
  <blob>.<suffix>.incomplete   a partial download in progress
```

Because a blob is **named by the sha256 of its contents**, a partial download can be
finished by any tool and then verified: hash the completed file and check it equals the
filename. That is exactly how the 1,280 MB partial was salvaged rather than re-fetched.

---

## 3. The metrics, explained

### Face detection and recognition

```
  pacino_a  2 face(s)  chose area=11036 px^2   det_score=0.865
  pacino_b  1 face(s)  chose area=112962 px^2  det_score=0.754
  waltz_a   3 face(s)  chose area=272864 px^2  det_score=0.690  (top det_score=0.832 is a smaller face)
  waltz_b   1 face(s)  chose area=167921 px^2  det_score=0.874
```

| Metric | Meaning | Reading |
|---|---|---|
| `det_score` | Detector confidence that the region is a face, 0–1 | All ≥0.69, comfortably over the 0.5 threshold. Detection is not a problem. |
| `area px^2` | Bounding-box area of the chosen face | Ranges 11k → 273k. A **25× spread**, and it correlates with embedding quality. |
| face count | Faces found per image | 1–3. Even simple portraits contain bystanders. |

### The similarity table

```
genuine (same actor):              impostor (different actors):
  pacino_a/pacino_b   +0.3086        pacino_a/waltz_a   +0.0152
  waltz_a/waltz_b     +0.5271        pacino_a/waltz_b   +0.0026
                                     pacino_b/waltz_a   -0.0044
separation margin: +0.2934           pacino_b/waltz_b   -0.0693
```

- **Genuine scores** (+0.31, +0.53) — same person, as expected.
- **Impostor scores** (−0.069 … +0.015) — all clustered tightly around **zero**. This is
  the ideal shape: different identities are *unrelated*, not merely less similar.
- **Separation margin** = (worst genuine) − (best impostor) = 0.3086 − 0.0152 = **+0.2934**.

The margin is the headline. A positive margin means a threshold exists that perfectly
separates genuine from impostor on this data; its *size* says how much room there is
before harder images start to overlap. +0.29 with impostors pinned at zero is healthy.

**The caveat that matters:** this is measured on clean publicity portraits. Godfather
interiors, Rocky IV's bloodied side-on boxing faces, and low-light Basterds scenes will
compress this margin substantially. Phase 0 proves the tools work; it does not prove they
work on the actual corpus.

### Visual encoder

```
params 375M   dtype torch.float32   device mps   embedding dim: 768

image        "a snowy mountain lake"   "a portrait of a man in a suit"
snow                       +0.1293                          -0.0524
pacino_a                   -0.0807                          +0.0667
```

This is a **similarity matrix**: rows are images, columns are captions, cells are cosine
similarities. Correctness means the **diagonal dominates** — each image scores highest on
its own caption. It does, in both rows.

Note again how small the winning values are (+0.129, +0.067). For SigLIP this is a
confident match. **Rank by order, never threshold on the absolute value.**

### Resource and timing figures

| Metric | Value | Meaning |
|---|---|---|
| Peak RSS | **1.36 GB** | Resident Set Size — actual physical RAM held. Against a 12 GB gate and 24 GB machine, there is enormous headroom. |
| Face model load | 0.2 s warm (75 s first run) | First run includes the ~330 MB download. |
| Face detect + embed | 0.06–0.43 s/image | On CPU. At 450 frames ≈ **under 2 minutes** for the whole corpus. |
| Visual model load | 7.5 s | Reading 1.5 GB of weights onto the GPU. |
| Visual encode | 0.2 s for 2 images + 2 captions | Trivial at this corpus size. |
| Embedding dims | 512 (face), 768 (image) | 450 × 768 floats ≈ 1.4 MB — hence no vector DB. |

The timing numbers justify a plan decision: **plain CPU for faces is fast enough**, so
CoreML is unnecessary complexity.

### Versions pinned

```
uv 0.12.6          python 3.12.13 arm64    numpy 2.5.2
torch 2.13.0       torchvision 0.28.0      transformers 5.15.1
insightface 1.0.1  onnxruntime 1.29.0      huggingface_hub 1.28.0
```

---

## 4. Key notes

### 4.1 `det_score` is not subject salience — the finding to remember

In `waltz_a`, RetinaFace found three faces:

| Face | `det_score` | Area | Who |
|---|---|---|---|
| [0] | **0.832** | 2,839 px² | background bystander |
| [1] | 0.720 | 1,637 px² | background bystander |
| [2] | 0.690 | **272,864 px²** | **the actual subject** |

The subject's face is **96× larger** than the bystanders — and the detector was *more
confident* about the bystanders.

Selecting the highest-`det_score` face produced a same-person cosine of **−0.03**: the
code was confidently comparing the wrong human. Selecting the **largest** face gave
**+0.53**. The overall separation margin moved from **−0.031 to +0.293** on that one
change.

**Carry into Phase 2:** wherever a frame is reduced to one face, rank by bbox area. Movie
stills are wall-to-wall extras, so this will bite far harder there than it did on
portraits. Naive `det_score` selection would have produced a broken character gallery that
still looked plausible.

### 4.2 Small faces produce weak embeddings

The quieter half of the same lesson. The one small face in the set (11,036 px²) scored
**+0.31** genuine, where the large faces scored **+0.53**. Face size correlates with
embedding quality.

**Carry into Phase 2:** apply a **minimum face-area floor** before a face is allowed to
seed a character's gallery centroid. A blurry 40×40 background face should never define
what Michael Corleone looks like. Such faces can still be *assigned* to a character —
just not used to build the reference.

### 4.3 `insightface` installed clean — fallbacks unused

The plan's single biggest predicted risk was `insightface` failing to build its
`mesh_core_cython` extension. Version **1.0.1 installed with no build step at all.**

Both documented fallbacks (raw ONNX; `facenet-pytorch`) remain unused. `faces.py` should
**still** be written behind the `FaceBackend` protocol in Phase 2 — the protocol's value
is that it keeps the swap cheap, and that value does not disappear because today's install
happened to succeed.

### 4.4 `transformers` 5.x changed the feature-extraction contract

`get_image_features()` and `get_text_features()` now return the **full model output
object** (`BaseModelOutputWithPooling`), not a pooled tensor. The docstring in the
installed package still shows the old tensor-returning contract, so the code is the only
reliable source.

Symptom: `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'device'`.

Handled by a helper that accepts either shape:

```python
def _pooled(out):
    return out if hasattr(out, "device") else out.pooler_output
```

**Carry into Phase 3:** `enrich/visual.py` makes the identical call and needs the same
handling.

### 4.5 Hugging Face downloads: Xet hangs, and `hf` does not resume

Two separate traps, both costly:

1. **Xet transport hangs.** `huggingface_hub` 1.x routes downloads through Xet by
   default. Transfers sat at *exactly* 0 bytes/sec with the process alive, no error, and
   no timeout — indistinguishable from a dead network, which is what made it slow to
   diagnose. `HF_HUB_DISABLE_XET=1` restored throughput immediately (0 → 3.7 MB/s). It is
   now forced from `config.py` via `os.environ.setdefault`, **before** any transformers
   import.

2. **`hf download` never resumes.** Each invocation creates a *new*
   `<blob>.<suffix>.incomplete` file and restarts from zero, orphaning the previous
   partial. Three partials (1280 MB, 469 MB, 447 MB) accumulated for one 1.5 GB file. The
   instinct — kill it and retry — is precisely what destroys progress.

**The recovery recipe**, which worked and is worth keeping:

```bash
# rename the largest partial to the target blob name (the stem of the .incomplete file)
mv blobs/<sha256>.<suffix>.incomplete blobs/<sha256>
curl -L -C - --retry 20 --retry-all-errors -o blobs/<sha256> <resolve-url>
shasum -a 256 blobs/<sha256>          # must equal the filename
ln -sf ../../blobs/<sha256> snapshots/<rev>/model.safetensors
```

`curl -C -` resumes properly, and the sha256-named blob makes the result self-verifying.
The recovered file hashed correctly on the first try.

### 4.6 Model downgraded: base instead of so400m

`VISUAL_MODEL` is now `google/siglip2-base-patch16-384` (375M params, 1.5 GB) rather than
`siglip2-so400m-patch14-384` (4.54 GB in fp32). so400m is demoted to the head of
`VISUAL_MODEL_FALLBACKS` as the quality upgrade path.

**The risk this introduces, recorded in `config.py`:** Phase 5 measures `filter_rank`
against `appearance_only`, and the *description-critical* query class depends on ranker
quality. If base underperforms there, that must not be misread as evidence about the
identity-filter thesis. **Phase 5 should re-run the eval against so400m before drawing
conclusions about the ranker.**

### 4.7 What Phase 0 does *not* prove

Worth stating plainly, because the gate passing 5/5 can read as more than it is.

These results come from **clean, well-lit publicity portraits** — close to the easiest
input either model will ever see. The actual corpus is Godfather interiors in near
darkness, Rocky IV boxing scenes with bloodied faces at extreme angles, and low-light
Basterds farmhouse dialogue.

Phase 0 proves **the tools are not broken and the environment works**. Phase 2 is where
the thesis is genuinely stressed, and where the +0.293 margin should be expected to shrink.
