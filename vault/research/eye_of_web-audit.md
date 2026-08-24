# Audit: MehmetYukselSekeroglu/eye_of_web

**Date:** 2026-08-23
**Commit:** `943644f` (2026-01-18) · 316★ · MIT · 34k LOC Python / 106 files
**Question asked:** is this forkable as the base for arc_search?
**Answer:** No. See [[ADR-002-greenfield-not-fork]].

This is the closest existing project to arc_search — InsightFace + Milvus +
Postgres + a crawler. Auditing it was cheaper than rediscovering its failures
ourselves, and every non-negotiable in the root `CLAUDE.md` traces to a finding
here.

---

## Repo health

| Signal | Finding |
|---|---|
| License | MIT — forking was legally fine |
| Tests | **Zero.** The three `*test*.py` files are manual scripts |
| CI | **None** |
| Dependency pinning | **None.** Every dep `>=`, no lockfile |
| Largest file | `src/app/routes/web.py` — **5,269 LOC** |
| HTTP stacks | **Three** — Selenium + Playwright + aiohttp |
| Vector store | Mid-migration: `pgvector/pgvector:pg16` *and* Milvus 2.4.0 both live |
| Deployment | 6 containers; dev-default creds; 5432/19530 published to host |

## Storage — the decisive finding

Per crawled image it stores a **half-scale WebP q90 copy of the entire scene**,
zlib-wrapped, as a Postgres `BYTEA`. **~40 KB/image ⇒ ~1.2 TB at 30M.**

No face crops are stored on the crawl path at all.

It is not configurable. `src/lib/database_tools.py:868`:

```python
save_image = True  # [TR] this line may be for dev/test, needs a decision
```

...unconditionally overwrites the `save_image: bool = False` parameter declared
four lines above. The no-pixels path exists in the signature and is dead code.

Also `proccess_image.py:61` re-encodes the decoded image to **PNG** purely as a
transport buffer to the DB layer — a 60 KB JPEG becomes a multi-MB lossless blob
in RAM before being resized back down.

**Scaling defect:** `database_tools.py:971` does
`SELECT "ID" FROM "ImageID" WHERE "BinaryImage" = %s` against a column with **no
index** (`schema.sql:170`). Every insert is a full sequential scan doing
multi-KB byte-equality comparisons. Quadratic; dominates wall-clock before 1M
rows. And the cheap SHA1 check at `:1006` runs *after* the blob insert at `:962`.

→ [[ADR-001-crop-only-storage]]

## Dedup — cryptographic only

Zero repo-wide matches for `imagehash`, `phash`, `dhash`, `pdq`, `perceptual`.
SHA1 exact-byte only. One re-encode or a single-pixel crop defeats it entirely,
so the same photo across 50 sites becomes 50 blobs and 50 near-identical index
rows — which then inflates match counts in a way that reads like corroborating
evidence but is one image counted 50 times.

No face-level canonicalization either: no search-before-insert, no clustering.
One row per detection, forever.

## Model and quality gating

- **`buffalo_l` (R50), not the antelopev2 the README claims.**
  `init_insightface.py:44`. The Turkish comment explains the fallback: antelopev2
  unzips to a nested `~/.insightface/models/antelopev2/antelopev2` path that
  breaks their Docker flow. *(We flatten it in the Dockerfile and keep R100.)*
- `init_insightface.py:38` hardcodes `["CPUExecutionProvider"]`;
  `onnxruntime-gpu` is commented out in requirements. **~20× throughput loss.**
- Quality gate is a single global `det_thresh=0.75`. No min size, no blur, no
  pose. A 12×12 background face is indexed at the same weight as a portrait.

## Ranking — none

`database_tools.py:3085`: `similarity_score = distance` — raw ANN score, sorted
descending. That is the entire ranking.

Landmarks, bbox, and `det_score` are stored, indexed, and returned in
`output_fields` — then used **only to draw overlays**. Nothing feeds back.

`lib/similarity_utils.py` (Numba/CUDA cosine/euclidean/manhattan) is plumbed
through the UI as an `algorithm` + `use_cuda` selector and is **entirely
vestigial**. Its own docstring admits it. The picker has no effect on results.

## Correctness bugs

1. **Inverted similarity comparison.** `database_tools.py:3723-3729` treats
   `hit.distance` as a distance (`1.0 - distance`, `distance <= threshold`) while
   `:3085` correctly treats it as a similarity. Milvus COSINE returns similarity,
   so `find_similar_face_ids_in_milvus` — the entry point for "find all images of
   this person" — **accepts hits with cosine ≤ 0.55 and rejects genuine
   matches.** It logs confident `✓ ACCEPTED / ✗ REJECTED` labels while doing it.

2. **Two of three search sources silently dead.**
   - `findSimilarWhiteListFaces` (`:2862`) is a leftover pgvector query selecting
     columns that do not exist in `schema.sql`. Raises `UndefinedColumn`, caught
     at `:2975`, returns `[]`.
   - `findSimilarEgmFaces` is called at `search_controller.py:183` and **is never
     defined anywhere in the repo.** Raises `AttributeError`, swallowed at `:197`.

   Both absorbed by broad `except`. The operator sees "no match", not an error.
   For a watchlist tool, a silent false negative is the worst failure mode
   available.

3. **Thresholds are uncalibrated literals.** `0.6` repeated in ~10 sites, `0.45`
   in five more. No labeled set, no ROC, no tuning script. The only justification
   in the repo is an inline comment reading *"for balanced precision/recall"*.

## Scale claim — refuted

README claims "billions of face data". No benchmark exists anywhere. Against it:

- Milvus **standalone**, single node
- No sharding, no partition key, no mmap (zero grep hits)
- No quantization at all — raw `FLOAT_VECTOR`
- **All three** vector indexes force-loaded to RAM at startup
  (`MILVUS_SCHEMA_GENERATOR.py:283-291`) ⇒ ~2.9 KB/face ⇒ **3+ TB at 1B**
- HNSW built on `landmarks_2d` (212-d) and `face_box` (**4-d**) — never searched
- `consistency_level="Strong"` on every query
- **top_k hardcoded to 100**, no caller override
- N+1 Postgres enrichment with `= ANY(BIGINT[])` that cannot use the btree index

## Crawler — the weakest layer

`single_domain_playwright_crawler.py` **is not a crawler.** It fetches a fixed
list and never enqueues anything.

The one genuine crawler:

| Issue | Detail |
|---|---|
| robots.txt | **Entirely absent.** Zero matches for `robots`, `Crawl-delay` |
| Rate limiting | **None.** Only a 5s pause *between domains* |
| Frontier | In-memory only. **A crash loses the crawl.** `is_crawled` written to PG, never read back |
| URL dedup | Raw string, fragment stripped only. `/a`, `/a/`, `?b=1&c=2`, `?c=2&b=1` all separate |
| Retry | **None.** A transient 503 permanently drops the page |
| Worker exit | `len(active_threads) == 1` — unsatisfiable with >1 thread. Threads spin forever, leak across targets |
| Pre-download filter | Extension blacklist only. 1×1 tracking pixels fully downloaded and face-detected |
| srcset / JSON-LD / `<picture>` | **Absent.** og:image only in platform scrapers |
| TLS | `verify=False` throughout, warnings suppressed |
| Depth bug | Over-depth URLs never marked visited ⇒ unbounded repeat work at the boundary |

**Identity posture — deliberate, not sloppy:** rotating 2010-era Opera 8.51–9.99
UA strings, a forged `Referer: google.com/search` on every request,
`--disable-blink-features=AutomationControlled` plus `navigator.webdriver`
masking, and `GOOGLE_BOT_IMAGE = "Googlebot-Image/1.0"` sitting ready in
`user_agent_tools.py:99`. **No honest bot identity exists anywhere in the repo.**

That last item matters beyond taste — Googlebot impersonation converts a
robots.txt argument into a fraud argument. Not something to inherit in a git
history.

---

## What we took

1. **The string-interning schema** (`BaseDomainID`, `UrlPathID`, `ImageTitleID`
   as ID→string tables). Genuinely good, and why their non-blob metadata is only
   ~200 B/row. Adopted in `sql/schema.sql`.
2. **The antelopev2 Docker gotcha, pre-solved.**
3. **A concrete anti-spec.** Every gap arc_search designs against is now
   empirically confirmed as the thing that breaks a real system at this scale.
4. **The transferable lesson:** the absence of a labeled eval set is *why* the
   thresholds are unjustified and *why* the inverted comparison survived. Build
   the eval set in week 1.
