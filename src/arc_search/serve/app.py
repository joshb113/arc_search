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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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
        return HTMLResponse(_page(_upload_form(deps.repo.stats(), deps.search_cfg)))

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
    """Wire the app from settings. Used by ``__main__`` and by uvicorn."""
    index_cfg, face_cfg, search_cfg = IndexSettings(), FaceSettings(), SearchSettings()
    return create_app(
        Deps(
            repo=SearchRepo(index_cfg.pg_dsn),
            vectors=VectorStore(index_cfg),
            extractor=FaceExtractor(face_cfg),
            face_cfg=face_cfg,
            search_cfg=search_cfg,
            crop_root=face_cfg.crop_dir,
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
