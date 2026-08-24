"""Link and image extraction.

eye_of_web delegated extraction to an unvendored third-party package
(``HiveWebCrawler``) and added CSS-background scraping on top. It had zero
repo-wide matches for ``srcset``, ``ld+json``, or ``<picture>``, and used
``og:image`` only inside its platform scrapers -- so on a general crawl it
systematically missed the highest-resolution and most semantically-central
image on most modern pages.

This module extracts, in priority order:

  1. ``og:image`` / ``twitter:image``  -- the page's own declared hero image
  2. JSON-LD ``image`` fields          -- structured, often full-resolution
  3. ``<picture>/<source srcset>``     -- largest candidate by descriptor
  4. ``<img srcset>``                  -- ditto
  5. ``<img src>`` / ``data-src``      -- the fallback everyone else stops at
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from selectolax.parser import HTMLParser

_SKIP_EXT = re.compile(r"\.(svg|gif|ico|webmanifest)(\?|$)", re.I)
_SRCSET_ENTRY = re.compile(r"\s*(\S+)(?:\s+([\d.]+)([wx]))?\s*")


@dataclass(frozen=True)
class FoundImage:
    url: str
    alt: str | None
    source: str  # which extractor found it -- useful for tuning
    width_hint: int | None = None


def _pick_srcset(srcset: str) -> tuple[str, int | None] | None:
    """Return the largest candidate in a srcset attribute."""
    best: tuple[str, int | None] | None = None
    best_score = -1.0
    for chunk in srcset.split(","):
        m = _SRCSET_ENTRY.fullmatch(chunk)
        if not m or not m.group(1):
            continue
        url, num, unit = m.group(1), m.group(2), m.group(3)
        # 'w' descriptors are pixel widths; 'x' are density multipliers.
        score = float(num) if num else 1.0
        if unit == "x":
            score *= 1000.0  # a 2x is usually bigger than a 1000w
        hint = int(float(num)) if (num and unit == "w") else None
        if score > best_score:
            best_score, best = score, (url, hint)
    return best


def _walk_jsonld(node: object, out: list[str]) -> None:
    if isinstance(node, dict):
        img = node.get("image") or node.get("thumbnailUrl")
        if isinstance(img, str):
            out.append(img)
        elif isinstance(img, list):
            out.extend(x for x in img if isinstance(x, str))
        elif isinstance(img, dict) and isinstance(img.get("url"), str):
            out.append(img["url"])
        for v in node.values():
            _walk_jsonld(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_jsonld(v, out)


def extract_images(html: str, base_url: str) -> list[FoundImage]:
    tree = HTMLParser(html)
    found: list[FoundImage] = []
    seen: set[str] = set()

    def push(raw: str | None, alt: str | None, source: str, hint: int | None = None) -> None:
        if not raw:
            return
        raw = raw.strip()
        if not raw or raw.startswith("data:"):
            return
        url = urljoin(base_url, raw)
        if not urlsplit(url).scheme.startswith("http"):
            return
        if _SKIP_EXT.search(url):
            return
        if url in seen:
            return
        seen.add(url)
        found.append(FoundImage(url=url, alt=alt, source=source, width_hint=hint))

    # 1. Declared hero images.
    for prop in ("og:image", "og:image:secure_url", "twitter:image"):
        for n in tree.css(f'meta[property="{prop}"], meta[name="{prop}"]'):
            push(n.attributes.get("content"), None, "meta")

    # 2. JSON-LD.
    for n in tree.css('script[type="application/ld+json"]'):
        text = n.text(strip=True)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            continue
        urls: list[str] = []
        _walk_jsonld(payload, urls)
        for u in urls:
            push(u, None, "jsonld")

    # 3. <picture><source srcset>
    for pic in tree.css("picture"):
        for src in pic.css("source"):
            ss = src.attributes.get("srcset")
            if ss and (best := _pick_srcset(ss)):
                push(best[0], None, "picture", best[1])

    # 4 + 5. <img>
    for img in tree.css("img"):
        alt = (img.attributes.get("alt") or "").strip() or None
        ss = img.attributes.get("srcset")
        if ss and (best := _pick_srcset(ss)):
            push(best[0], alt, "img-srcset", best[1])
        for attr in ("src", "data-src", "data-original", "data-lazy-src"):
            push(img.attributes.get(attr), alt, f"img-{attr}")

    return found


def extract_links(html: str, base_url: str) -> list[str]:
    tree = HTMLParser(html)
    out: list[str] = []
    seen: set[str] = set()

    # rel=nofollow is advisory, but on a targeted crawl there is no reason to
    # ignore a publisher telling you a link is not theirs.
    for a in tree.css("a[href]"):
        rel = (a.attributes.get("rel") or "").lower()
        if "nofollow" in rel:
            continue
        href = (a.attributes.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = urljoin(base_url, href)
        if not urlsplit(url).scheme.startswith("http"):
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def page_title(html: str) -> str | None:
    tree = HTMLParser(html)
    if node := tree.css_first("title"):
        return node.text(strip=True) or None
    return None
