# Image-model bring-up spike — DINOv2 + SigLIP2

**Date:** 2026-08-25
**Purpose:** supply the measured numbers [[ADR-005-image-search-is-primary]] needs.
**Status:** ADR-005 is still **Proposed**. This spike changed no project code, no
dependency of the working venv, and nothing in the root contract.

Run in a **separate venv** (`.venv-spike`) on purpose. The brain stem records
that insightface needed `--no-deps` and that `opencv-python` shadows the pinned
headless build; torch drags in its own stack, and breaking the live face
pipeline to answer a storage question would be a self-inflicted wound.

---

## Established

### The platform is not a blocker

| | |
|---|---|
| torch | **2.11.0+cu128** (`--index-url .../whl/cu128`), cp314 wheel |
| torchvision | 0.26.0+cu128 — **required**, `AutoImageProcessor` refuses without it |
| CUDA | build 12.8, `is_available` True |
| GPU | RTX 5070, **sm_120** present in `torch.cuda.get_arch_list()` |
| verified by | a real 2048² GPU matmul, not by trusting `is_available` |

🔴 **The default PyPI wheel is CPU-only on Windows.** Installing plain
`pip install torch` reproduces the project's loudest documented trap — the silent
CUDA→CPU fallback that cost 12× throughput — with no error. The `cu128` index URL
is mandatory, and Blackwell needs ≥12.8 regardless.

### Both models run, and both are 768-d

| model | dim | params | batch=1 | batch=7 |
|---|---|---|---|---|
| `facebook/dinov2-base` | **768** | 87 M | 70 img/s | **179 img/s** |
| `google/siglip2-base-patch16-384` | **768** | 375 M | 60 img/s | **93 img/s** |

768 is measured, not assumed. It is the input to every storage figure below.

### Both actually work

Tested on 7 real corpus images. Six are photographs of people; one is **a fish**
(`archive.fosdem.org/2024/schedule/speaker/QQ3MZP/...png`).

**DINOv2**, image↔image cosine:

```
person <-> person   0.289
person <-> fish    -0.017
separation         +0.306
```

**SigLIP2**, text→image. ⚠️ Score it on the model's own scale —
`sigmoid(cosine * logit_scale + logit_bias)`, measured `logit_scale=112.85`,
`logit_bias=-16.77`. Raw cosine is **not** meaningful for a sigmoid-loss model,
and reading it as if it were made the model look broken on the first pass:

```
query                     person ... person   FISH
"a photo of a fish"        0.000 ...  0.000   0.335
"a photo of a person"      0.016 ...  0.045   0.000
```

Clean discrimination in both directions.

---

## NOT established — do not build on these

### Truncation is uncosted, and the ADR's 512-d option is unfounded

ADR-005 currently offers "512-d truncated → ~206 GB". That assumes truncation is
free, which is only true for Matryoshka-trained models. **Neither model
advertises MRL**, and the spike could not settle it:

```
768-d full                  +0.306  (baseline)
512-d naive truncation      +0.319  (104%)
256-d naive truncation      +0.353  (115%)
512-d PCA                   +0.095  (31%)
```

Both halves of that table are invalid, and it is worth saying why rather than
quoting the flattering half:

- **n=7 with one fish.** Truncation appearing to *improve* separation is noise,
  not evidence. A real test needs hundreds of images across known groups.
- **The PCA rows are structurally meaningless.** 7 samples yield at most 6
  components, so 512/384/256 all collapse to the same 6-dim space — which is
  exactly why all three report an identical 0.095.

**The 512-d option stays uncosted until measured properly.** Treat only the
768-d figures as real.

### Everything else still open

- Throughput measured on 7 cached images, not a corpus-scale run competing with
  fetch. The face tier is fetch-bound at 1 rps; there is no reason to expect
  different here, which makes GPU img/s largely irrelevant to wall-clock.
- DINOv2 was substituted for DINOv3 (below). Not a like-for-like answer.
- HNSW has never been exercised on this machine: `indexed_vectors_count: 0`,
  because the collection is under Qdrant's 10,000 `indexing_threshold`. Every
  query so far has been a full scan, so current latency says nothing about scale.

---

## Storage, recomputed at the measured 768-d

At 30M images, against the current face engine's measured 49.8 GB:

| configuration | DINOv2 | SigLIP2 | + base | **total** |
|---|---|---|---|---|
| originals + int8 *(as `faces` is configured today)* | 115.2 | 115.2 | 49.8 | **280.2 GB** |
| int8 only, `on_disk` originals disabled | 23.0 | 23.0 | 49.8 | **95.9 GB** |

The entire decision is one config flag. `index/vectors.py:114` creates `faces`
with `on_disk=True` float32 originals **plus** the int8 copy, which is what makes
INDEX.md's "20.5 GB against 5.1 GB" the largest line in the current budget. Keep
that policy for two 30M-image collections and it is 280 GB; drop originals and it
is 96 GB.

Dropping originals costs rescoring — `VectorStore.search` currently passes
`rescore=True`, which re-ranks int8 candidates against the float32 originals. For
faces that was deliberate, because the recall it buys lands on exactly the hard
low-similarity matches the engine exists to find. Whether whole-image search
needs the same is **not established** and is a genuine open question, not a
formality.

---

## Two environment findings

⚠️ **C: has 9.8 GB free (96% used).** The spike venv and the HuggingFace cache
both had to go on F: (`HF_HOME=F:/.../data/hf-cache`). torch+CUDA is ~3 GB
installed and SigLIP2 is another ~1.5 GB; installing to the default locations
would have run the system disk out.

⚠️ **`facebook/dinov3-*` is `gated=manual`** — it needs terms accepted on
HuggingFace plus a token. DINOv2 was substituted to keep the spike moving. If
DINOv3 is the intended model, that access has to be granted before any number
here is treated as final; dimensions and throughput will differ.

Symlink warning on F: is cosmetic (`HF_HUB_DISABLE_SYMLINKS_WARNING=1`), but it
does mean the cache stores duplicates rather than links — more disk than the
model sizes suggest.

---

## Reproducing

```
uv venv --python 3.14 .venv-spike
uv pip install --python .venv-spike/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install --python .venv-spike/Scripts/python.exe \
    numpy pillow transformers sentencepiece protobuf
HF_HOME=F:/Josh\ Brannon/arc_search/data/hf-cache .venv-spike/Scripts/python.exe data/spike.py
```

`.venv-spike/` and `data/hf-cache/` are throwaway and should be gitignored, not
committed.
