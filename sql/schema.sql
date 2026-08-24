-- arc_search schema
--
-- Design borrowed from eye_of_web: string interning via ID->string tables.
-- That part of their design was genuinely good and keeps metadata at ~200 B/row.
--
-- What is NOT borrowed: image blobs. eye_of_web stored a half-scale WebP of every
-- crawled scene as a Postgres BYTEA (~40 KB/image, ~1.2 TB at 30M) and did an
-- unindexed `WHERE "BinaryImage" = %s` full-blob equality scan on every insert.
-- Here, pixels live on the filesystem as 128px face crops and Postgres holds only
-- the path. See vault/decisions/ADR-001-crop-only-storage.md

BEGIN;

-- ---------------------------------------------------------------- interning

CREATE TABLE IF NOT EXISTS domain (
    id      BIGSERIAL PRIMARY KEY,
    host    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS url_path (
    id        BIGSERIAL PRIMARY KEY,
    path      TEXT NOT NULL,
    path_hash BYTEA NOT NULL UNIQUE          -- sha1(path); index the hash, not the text
);

CREATE TABLE IF NOT EXISTS text_blob (
    id        BIGSERIAL PRIMARY KEY,
    body      TEXT NOT NULL,
    body_hash BYTEA NOT NULL UNIQUE
);

-- ---------------------------------------------------------------- crawl

CREATE TABLE IF NOT EXISTS page (
    id           BIGSERIAL PRIMARY KEY,
    domain_id    BIGINT NOT NULL REFERENCES domain(id),
    url_path_id  BIGINT NOT NULL REFERENCES url_path(id),
    title_id     BIGINT REFERENCES text_blob(id),
    http_status  SMALLINT,
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_checked TIMESTAMPTZ,                -- reaper: TTL sweep for dead links
    UNIQUE (domain_id, url_path_id)
);
CREATE INDEX IF NOT EXISTS page_last_checked_idx ON page (last_checked NULLS FIRST);

-- One row per distinct image, keyed by content hash. No pixels.
CREATE TABLE IF NOT EXISTS image (
    id         BIGSERIAL PRIMARY KEY,
    sha1       BYTEA NOT NULL UNIQUE,        -- exact-byte identity
    pdq        BIT(256),                     -- perceptual hash, NULL until computed
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    byte_size  INTEGER NOT NULL,

    -- Where the file lives, so it can be re-fetched. See ADR-003.
    --
    -- The host is INTERNED and the path is INLINE, which breaks the pattern
    -- used elsewhere in this file on purpose: interning exists to collapse
    -- values that repeat, and image paths are effectively unique per image.
    -- Measured on 100k realistic paths, interning the path cost 295 B/image
    -- against 157 B/image inline -- 88% more, for zero deduplication.
    --
    -- domain_id comes from the IMAGE's URL, not the page's. Images routinely
    -- live on another host: static.media.ccc.de serves every thumbnail that
    -- appears on media.ccc.de.
    --
    -- No index. Dedup goes through sha1, provenance through image_source, and
    -- corpus analysis is an occasional scan; a btree here would roughly double
    -- the column's cost for no query that currently exists.
    domain_id  BIGINT NOT NULL REFERENCES domain(id),
    url_path   TEXT NOT NULL,

    -- Tri-state, and the distinction is load-bearing:
    --   -1  never examined by a detector   (the crawl tier writes this)
    --    0  examined, no qualifying face   => barren, skip on recrawl
    --   >0  this many faces indexed
    --
    -- This defaulted to 0 and that was a bug. The crawl tier runs with no model
    -- in the loop, so every row it wrote claimed "examined, no faces". On the
    -- next startup Deduper.load() read those as barren and week 2 would have
    -- skipped the entire corpus -- a silent, total loss of face extraction that
    -- would have presented as a model failure. Caught by
    -- test_resume_does_not_mark_unexamined_images_barren.
    face_count SMALLINT NOT NULL DEFAULT -1,
    CONSTRAINT face_count_valid CHECK (face_count >= -1),

    first_seen TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Work queue for the indexing pass: everything the detector has not seen.
CREATE INDEX IF NOT EXISTS image_unexamined_idx ON image (id) WHERE face_count < 0;
-- PDQ near-duplicate lookup is done in the app via a BK-tree seeded from this
-- column; the btree here is only for the exact-PDQ fast path.
CREATE INDEX IF NOT EXISTS image_pdq_idx ON image (pdq) WHERE pdq IS NOT NULL;

-- Many-to-many: the same image appears on many pages. This is the table that
-- makes "where else does this face appear" cheap.
CREATE TABLE IF NOT EXISTS image_source (
    image_id     BIGINT NOT NULL REFERENCES image(id) ON DELETE CASCADE,
    page_id      BIGINT NOT NULL REFERENCES page(id) ON DELETE CASCADE,
    alt_text_id  BIGINT REFERENCES text_blob(id),
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (image_id, page_id)
);
CREATE INDEX IF NOT EXISTS image_source_page_idx ON image_source (page_id);

-- ---------------------------------------------------------------- faces

CREATE TABLE IF NOT EXISTS face (
    id            BIGSERIAL PRIMARY KEY,
    image_id      BIGINT NOT NULL REFERENCES image(id) ON DELETE CASCADE,

    -- Qdrant point id. The 512-d embedding lives ONLY in Qdrant.
    qdrant_id     UUID NOT NULL UNIQUE,

    -- Geometry at ORIGINAL image resolution. Never pre-scaled to match a
    -- derived artifact -- eye_of_web multiplied landmarks by 0.5 to line up
    -- with a lossy thumbnail, using a constant duplicated across two files.
    bbox          REAL[4] NOT NULL,
    landmarks     REAL[] NOT NULL,           -- 5-point kps, flattened [x,y]*5
    src_width     INTEGER NOT NULL,          -- so crops stay re-derivable
    src_height    INTEGER NOT NULL,

    det_score     REAL NOT NULL,
    blur_var      REAL NOT NULL,             -- Laplacian variance
    yaw           REAL,
    quality       REAL NOT NULL,             -- composite, used for re-rank weighting
    age_est       SMALLINT,

    crop_path     TEXT,                      -- relative path under ARC_CROP_DIR

    -- Face-level canonicalization. NULL => this row IS canonical.
    canonical_id  BIGINT REFERENCES face(id) ON DELETE SET NULL,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS face_image_idx     ON face (image_id);
CREATE INDEX IF NOT EXISTS face_canonical_idx ON face (canonical_id);
CREATE INDEX IF NOT EXISTS face_quality_idx   ON face (quality DESC);

-- ---------------------------------------------------------------- exclusion

-- Hash-based opt-out, consulted on every search. Exists from day 1 so that
-- honouring a removal request is a row insert, not an engineering project.
CREATE TABLE IF NOT EXISTS exclusion (
    id         BIGSERIAL PRIMARY KEY,
    qdrant_id  UUID NOT NULL,                -- vector to suppress
    reason     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS exclusion_qdrant_idx ON exclusion (qdrant_id);

-- ---------------------------------------------------------------- eval

-- Labeled pairs backing threshold calibration. Thresholds are DERIVED from a
-- run of arc_search.eval.calibrate against this table, never hardcoded.
CREATE TABLE IF NOT EXISTS eval_pair (
    id         BIGSERIAL PRIMARY KEY,
    face_a     BIGINT NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    face_b     BIGINT NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    same_person BOOLEAN NOT NULL,
    labeled_by TEXT,
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (face_a, face_b)
);

COMMIT;
