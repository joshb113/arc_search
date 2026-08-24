# ADR-003 — Store the image URL, inline rather than interned

**Status:** accepted, 2026-08-24
**Supersedes nothing. Amends the storage budget in ADR-001 by +1.57 GB.**

## Context

`image` held `sha1`, `pdq`, dimensions, `byte_size` and `face_count` — and no
URL. Provenance stopped at the *page* an image appeared on, via `image_source`.

Three things that costs us:

1. **An image cannot be re-fetched.** ADR-001's whole argument for crop-only
   storage is that a better model in 2027 should not require re-crawling. But
   re-deriving a crop needs either the original bytes (which we correctly do not
   keep) or its URL (which we were not keeping either). The re-derivation story
   had a hole in it.
2. **Corpus analysis cannot produce scope rules.** When the ccc-media validation
   turned out to be 70% conference logos, working out the logo URL shape meant
   re-fetching a page by hand, because the database could not answer "what were
   the URLs of the images you called logos".
3. **Debugging stops one level short.** "Where did this row come from" reaches
   the page, not the file.

## Decision

Add to `image`:

```sql
domain_id BIGINT NOT NULL REFERENCES domain(id),
url_path  TEXT   NOT NULL,
```

`domain_id` is **interned**; `url_path` is stored **inline**.

Both `NOT NULL`. The ~2,400 images crawled before this ADR were deleted and
their frontier entries reset to PENDING so the running crawl re-covered them.
A nullable column would have bought an hour of crawl time and cost a permanent
ambiguity in the table.

Note the host comes from the *image's own URL*, not the page's — images
routinely live on a different host from the page that references them
(`static.media.ccc.de` serves every thumbnail on `media.ccc.de`).

## Why inline, when the rest of the schema interns

Measured on 100k realistic paths (mean 84 chars, sampled from both live
verticals), Postgres 16, including indexes and `VACUUM ANALYZE`:

| design | per image | at 10M images |
|---|---|---|
| **inline `TEXT` on `image`** | **157 B** | **1.57 GB** |
| interned via a `url_path` row | 295 B | 2.95 GB |
| *interning overhead* | *+138 B* | *+1.38 GB* |

Interning costs **88% more** and returns nothing. The interning tables exist to
collapse values that *repeat* — a few dozen hosts across 30M images, page paths
shared by many images, alt text like "conference logo" appearing hundreds of
times. Image paths are effectively unique per image, so an interning table adds
one row and one UNIQUE index entry per image for zero deduplication.

So the rule is: **intern what repeats, store inline what does not.** `domain_id`
repeats heavily and stays interned. `url_path` does not and goes inline.

(This also raises a question about the existing `url_path` table, which is 1:1
with `page` and therefore interning something that never repeats either. Out of
scope here; noted in `vault/plans/INDEX.md`.)

## Cost

+1.57 GB at the 10M-face target, against ADR-001's 48 GB. The revised total is
**~49.6 GB**, a 3.3% increase. Still one machine, still one disk.

No index on `url_path`. A `(domain_id, url_path)` btree would roughly double the
column's cost, and nothing in the query path needs it: dedup is by `sha1`,
provenance is by `image_source`, and corpus analysis is an occasional scan. Add
one if a real query demands it, not before.

## Alternatives rejected

**Nullable column, keep the existing rows.** Saves an hour of recrawl, costs a
permanent "NULL means it predates ADR-003" caveat that every future reader of
this column has to know about. Not worth it at 0.02% of the eventual corpus.

**Store every URL an image was seen at**, rather than the first. The same bytes
do appear at multiple URLs. But `image` is keyed by content hash, so this would
need another many-to-many table — near-doubling the cost — to answer a question
nobody has asked yet. `image_source` already records every *page*, which is the
provenance question the engine actually exists to answer. One URL is enough to
re-fetch.

**Store the full URL including scheme and host as one string.** Simpler, but
duplicates the host on every row: ~20 bytes × 10M = 200 MB to store
"archive.fosdem.org" ten million times, when it is the textbook case for
interning.
