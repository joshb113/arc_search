# Face model bring-up — measurements

**Date:** 2026-08-24
**Hardware:** RTX 5070, 12 GB, driver 610.74. Windows. Python 3.14.
**Question that started it:** does `insightface` run at all on this machine?

Answer: yes, but nothing about the shipped configuration was correct, and two of
the three defects found were **silent**.

---

## 1. The stack works — via insightface 1.0.1, not 0.7.3

`pyproject.toml` floored `insightface>=0.7.3`, which is sdist-only and needs a C
toolchain. **1.0.1** (released 2026-05-23) is the first release published as a
pure-python `py3-none-any` wheel, and every transitive dependency resolves with
a cp314 wheel:

```
scikit-image 0.26.0  scipy 1.18.1  onnx 1.22.0  onnxruntime 1.29.0
```

Floor raised to `>=1.0.1`. Do not float it back down.

**Packaging trap:** insightface depends on `opencv-python`, arc_search pins
`opencv-python-headless`. Both provide `cv2` and installing both shadows one
arbitrarily. Installed here with `--no-deps` plus explicit deps. A plain
`pip install -e .` will pull the full build.

## 2. 🔴 antelopev2 unzips one level too deep

Exactly as `config.py` warns, and it is a hard failure rather than a silent one:

```
~/.insightface/models/antelopev2/antelopev2/*.onnx    <- as downloaded
AssertionError: assert 'detection' in self.models
```

Flattened by hand. All five models present, including `genderage.onnx` — so the
`min_age_est` filter, described in config as "the highest-value filter in the
system", does work. Verified: real ages 33 and 27 came back on real photos.

⚠️ `config.py` cites `docker/flatten_models.sh` for this. **That file does not
exist** — there is no `docker/` directory. Doc/code drift; the flatten step is
currently a manual thing nobody has written down.

## 3. 🔴 CUDA was loading, silently, as CPU

The largest finding. onnxruntime **does not raise** when the CUDA provider fails
to load — one line to stderr, then every session falls back to CPU.

```
Error loading onnxruntime_providers_cuda.dll which depends on
"cublasLt64_13.dll" which is missing
→ effective provider: CPUExecutionProvider   (on all 5 models)
```

Two separate causes:

1. No CUDA runtime at all — driver only, no toolkit. Fixed by the
   `onnxruntime-gpu[cuda,cudnn]` extras, which ship it as wheels.
2. Even installed, the DLLs were unreachable. The wheels use a **consolidated
   `nvidia/cu13/bin/x86_64` layout**; onnxruntime looks elsewhere.
   `os.add_dll_directory` does **not** fix this — it governs DLLs loaded with
   `LOAD_LIBRARY_SEARCH_USER_DIRS`, not the transitive imports of a DLL loaded
   by absolute path. **Prepending to `PATH` fixed it outright.** Both were tried;
   this is measured, not reasoned.

Now handled by `faces.register_cuda_runtime()`, called before insightface is
imported, globbing for any dir under `nvidia/` that actually contains DLLs.

### What it was worth

| Provider | 720p image | throughput |
|---|---|---|
| CUDA | 20.4 ms | **49.0 img/s** |
| CPU | 242.1 ms | 4.1 img/s |

**11.9x.** Projected over the 30M-image scale target: **~7 days vs ~85 days.**

### The compensating check

`_ensure_app()` logged `providers=self._cfg.providers` — the *requested* list. It
would have reported CUDA while running on CPU.

This is the **same defect class the crawler already shipped and fixed**: the
startup log printed the requested per-host rate while the crawl ran at half of
it (`politeness.override_ignored`). Now:

- `FaceExtractor.effective_providers()` asks the live sessions
- `faces.model_ready` logs `requested=` **and** `effective=`
- `faces.cuda_unavailable` warns when CUDA was asked for and CPU was delivered

---

## 4. 🔴 `min_face_px = 64` discards half of all real faces

Measured over 25 distinct FOSDEM speaker photos carrying `alt="Photo of NAME"`,
fetched at 1 rps. 18 detections from 25 images (7 images are logos/placeholders
with no face at all — correct).

```
face short side px:  min=35  p10=50  MEDIAN=64  p90=99  max=101
image dims present:  128x128, 150x150, 220x180
```

| `min_face_px` | detections kept |
|---|---|
| 40 | 17/18 (94%) |
| 48 | **17/18 (94%)** |
| 56 | 15/18 (83%) |
| **64 (current)** | **9/18 (50%)** |
| 72 | 4/18 (22%) |

**The threshold sits exactly on the median.** A 128x128 FOSDEM avatar yields a
~50–64 px face, so half the corpus is rejected as `too_small`.

This is the **third** instance of this bug class in this project:

1. `min_image_dim=200` → rejected 9 of 9 real photos → now 64
2. `min_image_bytes=8000` → rejected 4 of 9 → now 2000
3. `min_face_px=64` → rejects 9 of 18 ← **open**

### ⚠️ Why the ordering matters more than the number

Running the backfill at 64 does not merely skip those images — it **tombstones**
them. `mark_examined(image_id, 0)` writes `face_count = 0`, which `Deduper` reads
as *examined, barren, never look again*. A later recalibration would not recover
them without an explicit reset pass.

So this must be settled **before** the first full backfill, or the provisional
pass must not write barren verdicts.

**Not changed unilaterally.** Non-negotiable #5 says thresholds are derived from
`arc_search.eval.calibrate`, not picked — and there is a real argument on the
other side, since `faces.py` documents small faces as the dominant false-positive
source at 10M scale. The tension is genuine: this corpus is mostly 128px avatars,
and the number that maximises recall here may be wrong at scale. It needs a
decision, and probably an ADR.

---

## 5. The 2013–2014 alt-text claim is wrong

`00_Brain/CLAUDE.md` and [[plan-003-precision]] both state that
`alt="Photo of NAME"` exists **2015–2025 but not 2013–2014**, measured one page
per year.

The corpus disagrees. Every labeled URL sampled here is under
`archive.fosdem.org/2013/`:

```
.../2013/schedule/speaker/andrew_dinn/...    Photo of Andrew Dinn
.../2013/schedule/speaker/dave_neary/...     Photo of Dave Neary
```

One page per year was too small a sample. The label is more available than
believed, which is good news for calibration — but the claim should be
re-measured properly rather than simply inverted.

---

## Reproducing

```
docker compose up -d postgres qdrant
.venv/Scripts/python.exe -m pip install --no-deps insightface
.venv/Scripts/python.exe -m pip install scipy scikit-image onnx tqdm requests
.venv/Scripts/python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]"
# then flatten ~/.insightface/models/antelopev2/antelopev2/* up one level
```

Probe scripts and raw output are in `data/probe/`.
