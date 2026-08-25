"""Search by uploaded photo. FastAPI, bound to 127.0.0.1.

LEGAL POSTURE
-------------
This binds to loopback and the default must not change without reading the
README section of the same name. A local index and a reachable service are
different legal objects; ``docker-compose.yml`` publishes every port on
127.0.0.1 for the same reason, and eye_of_web published Postgres and Milvus on
0.0.0.0 with default credentials.

THE UPLOADED PHOTO IS NEVER WRITTEN TO DISK
-------------------------------------------
It is read into memory, decoded, embedded, and dropped. Nothing persists it,
nothing logs its bytes, and there is no upload directory to forget about. The
whole premise is that the corpus AND the query stay on your hardware -- a query
image quietly accumulating in a temp folder would break that for no benefit.

NO VERDICTS
-----------
The UI reports RAW COSINE and says so. ``SearchSettings.calibrated`` is False
and t_plausible/t_strong/t_near_certain are placeholders; one measured impostor
pair already scores 0.651, above t_near_certain. Rendering "strong match" from
an uncalibrated number is exactly the assertion-instead-of-measurement that
non-negotiable #5 exists to forbid. When calibration has run, the banner and the
score column change together -- not one of them.
"""

from __future__ import annotations

import html
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from arc_search.config import FaceSettings, IndexSettings, SearchSettings
from arc_search.index.faces import FaceExtractor
from arc_search.index.vectors import VectorStore
from arc_search.serve.repo import SearchRepo

log = structlog.get_logger(__name__)

# A query photo is one image. This is a denial-of-service bound on a loopback
# service, not a quality gate -- quality is decided by the same FaceSettings
# gate the index tier uses, so an uploaded photo is judged exactly as a crawled
# one was.
MAX_UPLOAD_BYTES = 20_000_000


@dataclass
class Deps:
    """Everything the app needs, injected so tests need no live services."""

    repo: SearchRepo
    vectors: VectorStore
    extractor: FaceExtractor
    face_cfg: FaceSettings
    search_cfg: SearchSettings
    crop_root: Path
    # Whole-image search, ADR-005. Optional: absent means face-only, which is
    # what the UI falls back to rather than erroring.
    image_vectors: object | None = None
    embedder: object | None = None
    thumbs: ThumbnailCache | None = None
    # Injected so tests need no network. Signature mirrors Fetcher.get_image.
    fetch_image: object | None = None


class ThumbnailCache:
    """Bounded in-memory cache of re-fetched result images.

    🔴 WHY THE SERVER FETCHES THESE AND THE BROWSER DOES NOT

    The obvious implementation is ``<img src="https://theirsite/foo.jpg">`` and
    let the browser do it: free, instant, no politeness budget. It is also the
    one thing this project cannot do. The source host would then receive a
    request per result, from the user's own address, revealing exactly which
    images this person is looking at -- and for a face search engine that is the
    entire premise inverted. "The corpus and the query both stay on your
    hardware" has to include *which results you looked at*.

    So the server fetches, through the same politeness layer as everything else,
    and caches. Measured: the corpus spans 2 hostnames that resolve to ONE
    server, so a 20-result grid is 20 requests at one box -- 4 s at 5 rps. The
    cache is what makes the second look at a result free.

    ⚠️ Bounded and memory-only, deliberately. ADR-001 forbids persisting scene
    images; an unbounded or on-disk cache would become exactly the store that
    non-negotiable #1 exists to prevent. Eviction is LRU on a byte budget, so
    the worst case is a slow page, never a full disk.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._max = max_bytes
        self._data: OrderedDict[int, tuple[bytes, str]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, image_id: int) -> tuple[bytes, str] | None:
        item = self._data.get(image_id)
        if item is None:
            self.misses += 1
            return None
        self._data.move_to_end(image_id)
        self.hits += 1
        return item

    def put(self, image_id: int, body: bytes, content_type: str) -> None:
        if len(body) > self._max:
            return  # one oversized image must not evict everything else
        if image_id in self._data:
            self._bytes -= len(self._data[image_id][0])
        self._data[image_id] = (body, content_type)
        self._bytes += len(body)
        while self._bytes > self._max and self._data:
            _, (evicted, _) = self._data.popitem(last=False)
            self._bytes -= len(evicted)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._data),
            "bytes": self._bytes,
            "hits": self.hits,
            "misses": self.misses,
        }


def create_app(deps: Deps) -> FastAPI:
    app = FastAPI(title="arc_search", docs_url=None, redoc_url=None)

    # -- helpers ----------------------------------------------------------

    def _decode(raw: bytes) -> np.ndarray:
        import cv2

        if not raw:
            raise HTTPException(400, "empty upload")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode that file as an image")
        return img

    def _search(raw: bytes, limit: int):
        """Detect in the upload, search on the best face, hydrate the results."""
        img = _decode(raw)
        faces, rejects = deps.extractor.extract(img)

        if not faces:
            # The reject breakdown is the whole diagnostic. "No faces found" with
            # no reason is what makes a search engine feel broken; "the face was
            # 40px and the floor is 48" is something a person can act on.
            return None, rejects, []

        # Best available face by the same composite quality the index ranks on.
        query = max(faces, key=lambda f: f.quality)
        hits = deps.vectors.search(
            query.embedding,
            limit=limit,
            exclude=deps.repo.exclusions(),
        )
        return query, rejects, deps.repo.hydrate(hits)

    # -- routes -----------------------------------------------------------

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "calibrated": deps.search_cfg.calibrated,
                "vectors": deps.vectors.count(),
                **deps.repo.stats(),
            }
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_page(_home(deps)))

    @app.get("/crop/{face_id}")
    def crop(face_id: int) -> FileResponse:
        """Serve a stored face crop.

        The path comes from our own database, but it is still resolved and
        checked against the crop root before opening. A DB value is not a
        trusted path -- that assumption is how a traversal becomes possible
        later, when something else starts writing the column.
        """
        row = deps.repo._exec("SELECT crop_path FROM face WHERE id = %s", (face_id,)).fetchone()
        if row is None or not row[0]:
            raise HTTPException(404, "no crop for that face")

        root = deps.crop_root.resolve()
        path = (root / row[0]).resolve()
        if not path.is_relative_to(root):
            log.warning("serve.crop_path_escaped", face_id=face_id, crop_path=row[0])
            raise HTTPException(404, "no crop for that face")
        if not path.is_file():
            raise HTTPException(404, "crop file is missing from disk")
        return FileResponse(path, media_type="image/webp")

    @app.get("/thumb/{image_id}")
    async def thumb(image_id: int):
        """Re-fetch a result image through politeness, cached.

        The server does this rather than the browser -- see ThumbnailCache for
        why. A miss costs one politeness token; a hit costs nothing.
        """
        if deps.thumbs is None:
            raise HTTPException(503, "thumbnails are not configured")

        cached = deps.thumbs.get(image_id)
        if cached is not None:
            body, ctype = cached
            return Response(content=body, media_type=ctype, headers={"X-Cache": "hit"})

        url = deps.repo.image_url(image_id)
        if url is None:
            raise HTTPException(404, "no such image")
        try:
            fetched = await deps.fetch_image(url)
        except Exception as exc:
            log.info("serve.thumb_unavailable", image_id=image_id, error=type(exc).__name__)
            raise HTTPException(404, "source image is unavailable") from exc

        ctype = fetched.content_type or "image/jpeg"
        deps.thumbs.put(image_id, fetched.body, ctype)
        return Response(content=fetched.body, media_type=ctype, headers={"X-Cache": "miss"})

    @app.get("/text", response_class=HTMLResponse)
    async def text_search(
        q: str = Query(default="", max_length=500),
        limit: int = Query(default=24, ge=1, le=100),
        format: str = Query(default="html", pattern="^(html|json)$"),
    ):
        """Text -> image. The mode ADR-005 exists to add."""
        if deps.image_vectors is None or deps.embedder is None:
            raise HTTPException(503, "whole-image search is not configured")
        if not q.strip():
            return HTMLResponse(_page(_home(deps)))

        vec = deps.embedder.embed_text([q])[0]
        # Over-fetch, then collapse: a near-dup must not consume one of the
        # k slots it is about to be folded into.
        hits = deps.image_vectors.search_named("text", vec, limit=limit * 3)
        results = deps.repo.collapse_near_duplicates(deps.repo.hydrate_images(hits))[:limit]

        if format == "json":
            return JSONResponse(
                {
                    "query": q,
                    "mode": "text",
                    "calibrated": deps.search_cfg.calibrated,
                    # SigLIP is sigmoid-loss: raw cosine is not even the model's
                    # own scale, let alone a probability. Named so no client can
                    # mistake it.
                    "score_type": "raw_cosine",
                    "results": [_image_json(r) for r in results],
                }
            )
        return HTMLResponse(_page(_image_results(deps, "text", q, results)))

    @app.post("/similar")
    async def similar(
        photo: UploadFile = File(...),
        limit: int = Query(default=24, ge=1, le=100),
        format: str = Query(default="html", pattern="^(html|json)$"),
    ):
        """Image -> image, whole scene. Not a face search; see the mode tabs."""
        if deps.image_vectors is None or deps.embedder is None:
            raise HTTPException(503, "whole-image search is not configured")

        raw = await photo.read()
        img = _decode(raw)
        from PIL import Image

        pil = Image.fromarray(img[:, :, ::-1])  # cv2 gives BGR; PIL wants RGB
        (vec,) = deps.embedder.embed_images([pil])
        hits = deps.image_vectors.search_named("scene", vec.scene, limit=limit * 3)
        results = deps.repo.collapse_near_duplicates(deps.repo.hydrate_images(hits))[:limit]

        if format == "json":
            return JSONResponse(
                {
                    "mode": "scene",
                    "calibrated": deps.search_cfg.calibrated,
                    "score_type": "raw_cosine",
                    "results": [_image_json(r) for r in results],
                }
            )
        return HTMLResponse(_page(_image_results(deps, "scene", "uploaded image", results)))

    @app.get("/similar/{image_id}", response_class=HTMLResponse)
    async def more_like_this(image_id: int, limit: int = Query(default=24, ge=1, le=100)):
        """'More like this' from a result already in the index.

        Reuses the stored vector instead of re-fetching and re-embedding -- the
        point id is the image id, so it is a retrieve, not a round trip to the
        source host.
        """
        if deps.image_vectors is None:
            raise HTTPException(503, "whole-image search is not configured")
        points = deps.image_vectors.client.retrieve(
            deps.image_vectors.name, ids=[image_id], with_vectors=True
        )
        if not points:
            raise HTTPException(404, "that image has no scene vector")
        vec = points[0].vector["scene"]
        hits = deps.image_vectors.search_named(
            "scene", np.asarray(vec, dtype=np.float32), limit=limit * 3 + 1
        )
        # Drop the query image itself; it is trivially its own nearest neighbour.
        hits = [h for h in hits if h[0] != image_id]
        results = deps.repo.collapse_near_duplicates(deps.repo.hydrate_images(hits))[:limit]
        return HTMLResponse(_page(_image_results(deps, "scene", f"image #{image_id}", results)))

    @app.post("/search")
    async def search(
        photo: UploadFile = File(...),
        limit: int = Query(default=0, ge=0, le=500),
        format: str = Query(default="html", pattern="^(html|json)$"),
    ):
        raw = await photo.read()
        k = limit or deps.search_cfg.return_k
        query, rejects, results = _search(raw, k)

        if format == "json":
            return JSONResponse(
                {
                    # Named to be unmistakable in a client. These are cosine
                    # similarities, and until `calibrated` is true they carry no
                    # calibrated meaning at all.
                    "calibrated": deps.search_cfg.calibrated,
                    "score_type": "raw_cosine",
                    "detected": query is not None,
                    "rejected": {k2: v for k2, v in rejects.__dict__.items() if v},
                    "results": [
                        {
                            "face_id": r.face_id,
                            "cosine": round(r.score, 4),
                            "label": r.label,
                            "quality": round(r.quality, 3),
                            "image_url": r.image_url,
                            "pages": r.pages,
                            "crop": f"/crop/{r.face_id}",
                        }
                        for r in results
                    ],
                }
            )

        return HTMLResponse(_page(_results(query, rejects, results, deps.search_cfg)))

    return app


# --------------------------------------------------------------------------
# HTML. Deliberately one file with no template engine and no CDN: this is a
# loopback tool, and a build step or an external asset fetch would both be
# larger than the thing they serve.
# --------------------------------------------------------------------------

_CSS = """
:root{--bg:#0f1115;--fg:#e6e6e6;--dim:#8b93a1;--line:#232733;--accent:#7aa2f7;
--warn:#e0af68;--warnbg:#2a2318}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:22px;margin:0 0 4px} h1 span{color:var(--dim);font-weight:400}
.sub{color:var(--dim);font-size:13px;margin-bottom:24px}
.banner{background:var(--warnbg);border:1px solid var(--warn);color:var(--warn);
padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:24px}
.banner b{color:#f5d08c}
form{border:1px solid var(--line);border-radius:10px;padding:20px;margin-bottom:28px}
input[type=file]{color:var(--dim);font:inherit;width:100%;margin-bottom:12px}
button{background:var(--accent);color:#0f1115;border:0;border-radius:7px;
padding:9px 18px;font:600 14px inherit;cursor:pointer}
button:hover{filter:brightness(1.1)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:var(--dim);font-weight:500;font-size:12px;
text-transform:uppercase;letter-spacing:.04em;padding:0 10px 8px;
border-bottom:1px solid var(--line)}
td{padding:10px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:hover td{background:#151822}
img.crop{width:56px;height:56px;border-radius:6px;display:block;background:#1a1d26}
.score{font-variant-numeric:tabular-nums;font-weight:600}
.label{font-weight:500} .nolabel{color:var(--dim);font-style:italic}
a{color:var(--accent);text-decoration:none;font-size:12px}
a:hover{text-decoration:underline}
.meta{color:var(--dim);font-size:12px}
.stats{color:var(--dim);font-size:12px;margin-top:28px;
border-top:1px solid var(--line);padding-top:14px}
.empty{border:1px solid var(--line);border-radius:10px;padding:24px;color:var(--dim)}
.tabs{display:flex;gap:6px;margin-bottom:18px}
.tabs a,.tabs span{padding:7px 14px;border-radius:7px;font-size:13px;
border:1px solid var(--line);color:var(--dim);text-decoration:none}
.tabs .on{background:var(--accent);color:#0f1115;border-color:var(--accent);font-weight:600}
input[type=text]{width:100%;background:#151822;border:1px solid var(--line);color:var(--fg);
border-radius:7px;padding:10px 12px;font:inherit;margin-bottom:12px}
input[type=text]:focus{outline:0;border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#141720}
.card img{width:100%;height:132px;object-fit:cover;display:block;background:#1a1d26}
.card .body{padding:8px 10px}
.card .s{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px}
.card .n{font-size:11px;color:var(--dim);margin-top:2px;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.card .lk{font-size:11px;margin-top:4px;display:block}
.note{color:var(--dim);font-size:12px;margin:14px 0}

code{background:#1a1d26;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def _page(body: str) -> str:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>arc_search</title><style>{_CSS}</style></head>"
        f"<body><div class=wrap>{body}</div></body></html>"
    )


def _banner(cfg: SearchSettings) -> str:
    if cfg.calibrated:
        return ""
    # This is not decoration. Until eval.calibrate has run, a number here has no
    # calibrated meaning, and saying so is the difference between a tool and a
    # machine that produces confident nonsense about people.
    return (
        "<div class=banner><b>UNCALIBRATED.</b> Scores are raw cosine similarity, "
        "not confidence. Thresholds have not been derived from a labelled set, so "
        "this page shows no match/no-match verdict. A measured impostor pair "
        "already scores 0.651.</div>"
    )


def _header() -> str:
    return (
        "<h1>arc_search <span>— face search</span></h1>"
        "<div class=sub>Self-hosted. The corpus and the query both stay on this "
        "machine; the uploaded photo is never written to disk.</div>"
    )


def _tabs(active: str, enabled: bool) -> str:
    """Mode switcher. Face search is one mode of an image engine, not the product."""

    def tab(key, label, href):
        if key == active:
            return f"<span class=on>{label}</span>"
        if not enabled and key != "face":
            return f"<span title='whole-image search is not configured'>{label}</span>"
        return f"<a href='{href}'>{label}</a>"

    return (
        "<div class=tabs>"
        + tab("text", "Text → image", "/?mode=text")
        + tab("scene", "Similar images", "/?mode=scene")
        + tab("face", "Find this face", "/?mode=face")
        + "</div>"
    )


def _home(deps) -> str:
    """One page, three modes. Text first: it is the primary mode now (ADR-005)."""
    enabled = deps.image_vectors is not None and deps.embedder is not None
    out = [_header(), _banner(deps.search_cfg), _tabs("text", enabled)]
    if enabled:
        out.append(
            "<form action=/text method=get>"
            "<input type=text name=q placeholder='describe an image — "
            "e.g. a person speaking at a lectern' autofocus>"
            "<button type=submit>Search</button></form>"
            "<form action=/similar method=post enctype=multipart/form-data>"
            "<input type=file name=photo accept=image/* required>"
            "<button type=submit>Find visually similar images</button></form>"
        )
    else:
        out.append(
            "<div class=empty><b>Whole-image search is not configured.</b><br>"
            "<span class=meta>The scene/text collection or the embedding models "
            "are unavailable, so only face search is offered.</span></div>"
        )
    out.append(
        "<form action=/search method=post enctype=multipart/form-data>"
        "<input type=file name=photo accept=image/* required>"
        "<button type=submit>Find this face</button></form>"
    )
    out.append(_stats(deps.repo.stats()))
    return "".join(out)


def _image_json(r) -> dict:
    return {
        "image_id": r.image_id,
        "cosine": round(r.score, 4),
        "alt": r.alt,
        "url": r.url,
        "pages": r.pages,
        "faces": r.face_count if r.face_count > 0 else 0,
        "thumb": f"/thumb/{r.image_id}",
    }


def _image_results(deps, mode: str, query: str, results) -> str:
    enabled = deps.image_vectors is not None and deps.embedder is not None
    out = [_header(), _banner(deps.search_cfg), _tabs(mode, enabled)]
    out.append(
        f"<div class=sub>{html.escape(mode)} search for "
        f"<b>{html.escape(query)}</b> — {len(results)} result(s)</div>"
    )
    if not results:
        out.append("<div class=empty>Nothing matched.</div>")
        return "".join(out) + "<p><a href=/>&larr; search again</a></p>"

    cards = []
    for r in results:
        alt = f"<div class=n>{html.escape(r.alt)}</div>" if r.alt else "<div class=n>&nbsp;</div>"
        page = r.pages[0] if r.pages else r.url
        cards.append(
            "<div class=card>"
            f"<a href='/similar/{r.image_id}' title='more like this'>"
            f"<img src='/thumb/{r.image_id}' loading=lazy alt=''></a>"
            "<div class=body>"
            f"<div class=s>{r.score:.4f}"
            + (
                f" <span class=n style='font-weight:400'>+{r.duplicates} dup</span>"
                if r.duplicates
                else ""
            )
            + f"</div>{alt}"
            f"<div class=n>{r.width}&times;{r.height}"
            + (f" &middot; {r.face_count} face(s)" if r.face_count > 0 else "")
            + "</div>"
            f"<a class=lk href='{html.escape(page)}' target=_blank rel=noreferrer>source page</a>"
            "</div></div>"
        )

    out.append("<div class=grid>" + "".join(cards) + "</div>")
    out.append(
        "<div class=note>Thumbnails are re-fetched by <b>this machine</b>, not by "
        "your browser, so the source host never learns which results you looked "
        "at. First view of each is rate-limited; afterwards it is cached.<br>"
        "Near-duplicate copies are folded together by PDQ perceptual hash and "
        "marked <b>+n dup</b>; the threshold is PDQ's conventional 31 and is not "
        "yet calibrated against this corpus.</div>"
    )
    out.append("<p><a href=/>&larr; search again</a></p>")
    return "".join(out)


def _upload_form(stats: dict, cfg: SearchSettings) -> str:
    return (
        _header() + _banner(cfg) + "<form action=/search method=post enctype=multipart/form-data>"
        "<input type=file name=photo accept=image/* required>"
        "<button type=submit>Search</button></form>" + _stats(stats)
    )


def _stats(stats: dict) -> str:
    return (
        "<div class=stats>"
        f"{stats['faces']:,} faces indexed · "
        f"{stats['images_with_faces']:,} of {stats['images']:,} images · "
        f"{stats['pages']:,} pages · "
        f"{stats['exclusions']:,} suppressed"
        "</div>"
    )


def _results(query, rejects, results, cfg: SearchSettings) -> str:
    out = [_header(), _banner(cfg)]

    if query is None:
        rejected = {k: v for k, v in rejects.__dict__.items() if v}
        why = (
            "".join(f"<code>{html.escape(k)}</code> &times;{v} " for k, v in rejected.items())
            if rejected
            else "the detector found no face at all."
        )
        out.append(
            "<div class=empty><b>No usable face in that photo.</b><br><br>"
            f"Rejected by the quality gate: {why}<br><br>"
            "<span class=meta>The same gate is applied to crawled images, so an "
            "upload is judged exactly as the corpus was.</span></div>"
            "<p><a href=/>&larr; try another photo</a></p>"
        )
        return "".join(out)

    out.append(
        "<div class=sub>Query face: "
        f"det_score {query.det_score:.3f} · quality {query.quality:.3f}"
        + (f" · apparent age ~{query.age_est}" if query.age_est is not None else "")
        + f" · {len(results)} result(s)</div>"
    )

    if not results:
        out.append("<div class=empty>A face was detected, but the index returned nothing.</div>")
        return "".join(out) + "<p><a href=/>&larr; search again</a></p>"

    rows = []
    for r in results:
        label = (
            f"<span class=label>{html.escape(r.label)}</span>"
            if r.label
            else "<span class=nolabel>no weak label</span>"
        )
        pages = "".join(
            f"<div><a href='{html.escape(p)}' target=_blank rel=noreferrer>"
            f"{html.escape(p[:64])}</a></div>"
            for p in r.pages[:3]
        )
        rows.append(
            "<tr>"
            f"<td><img class=crop src='/crop/{r.face_id}' alt='face {r.face_id}' loading=lazy></td>"
            f"<td class=score>{r.score:.4f}</td>"
            f"<td>{label}<div class=meta>quality {r.quality:.3f} · "
            f"{r.src_width}&times;{r.src_height}</div></td>"
            f"<td>{pages}</td>"
            "</tr>"
        )

    out.append(
        "<table><thead><tr><th>face</th><th>cosine</th><th>weak label</th>"
        "<th>seen on</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p><a href=/>&larr; search again</a></p>"
    )
    return "".join(out)


# --------------------------------------------------------------------------


def build_default_app() -> FastAPI:
    """Wire the app from settings. Used by ``__main__`` and by uvicorn.

    Whole-image search is optional: if the collection or the models are
    unavailable the app still starts and offers face search, rather than
    refusing to run. A UI that will not start is worse than one mode short.
    """
    import httpx

    from arc_search.config import CrawlSettings, EmbedSettings
    from arc_search.crawler.fetch import Fetcher
    from arc_search.crawler.politeness import Politeness

    index_cfg, face_cfg, search_cfg = IndexSettings(), FaceSettings(), SearchSettings()
    repo = SearchRepo(index_cfg.pg_dsn)

    image_vectors = embedder = None
    try:
        from arc_search.index.embed import ImageEmbedder

        iv = VectorStore(index_cfg, spec=index_cfg.image_spec())
        if iv.verify():
            raise RuntimeError(f"images collection unavailable: {iv.verify()}")
        embedder = ImageEmbedder(EmbedSettings())
        scene_dim, text_dim = embedder.dims()
        want = dict(iv.spec.named)
        if (scene_dim, text_dim) != (want["scene"], want["text"]):
            raise RuntimeError(f"model dims {(scene_dim, text_dim)} != collection {want}")
        image_vectors = iv
        log.info("serve.whole_image_ready", device=embedder.effective_device())
    except Exception as exc:
        log.warning(
            "serve.whole_image_unavailable",
            error=f"{type(exc).__name__}: {exc}",
            detail="serving face search only",
        )
        image_vectors = embedder = None

    # Thumbnails go through the SAME politeness layer as the crawler. The server
    # fetches them so the source host never sees the user's address -- see
    # ThumbnailCache.
    crawl_cfg = CrawlSettings()
    client = httpx.AsyncClient(http2=True, follow_redirects=True, timeout=crawl_cfg.timeout_s)
    fetcher = Fetcher(crawl_cfg, client, Politeness(crawl_cfg, client))

    return create_app(
        Deps(
            repo=repo,
            vectors=VectorStore(index_cfg),
            extractor=FaceExtractor(face_cfg),
            face_cfg=face_cfg,
            search_cfg=search_cfg,
            crop_root=face_cfg.crop_dir,
            image_vectors=image_vectors,
            embedder=embedder,
            thumbs=ThumbnailCache(),
            fetch_image=fetcher.get_image,
        )
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    cfg = SearchSettings()
    ap = argparse.ArgumentParser(prog="arc_search.serve.app")
    ap.add_argument("--host", default=cfg.bind_host)
    ap.add_argument("--port", type=int, default=cfg.bind_port)
    args = ap.parse_args(argv)

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # Not blocked -- there are legitimate reasons to bind elsewhere behind a
        # tunnel -- but it is a decision with legal weight, so it is stated out
        # loud rather than happening quietly because of a flag in a script.
        log.warning(
            "serve.non_loopback_bind",
            host=args.host,
            detail="a local index and a reachable service are different legal "
            "objects. See README 'Legal posture'.",
        )

    uvicorn.run(build_default_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
