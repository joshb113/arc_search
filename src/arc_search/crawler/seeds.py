"""Seed configuration: which hosts we crawl, and the scope test for every URL.

This module is deliberately pure -- YAML in, dataclasses out, one boolean
function. It imports no httpx, no protego, no database driver. Scope decisions
are the single easiest thing in a crawler to get quietly wrong, and quietly
wrong here means either a crawl that escapes onto the open web or one that
silently drops the pages you wanted. Both are only catchable by unit tests, so
this has to be unit-testable without a network.

eye_of_web had no scope model at all: it followed whatever it found, discovered
it was crawling social platforms, and bolted on a hardcoded blocklist of ~15
domains inside the fetch function.

Precedence, highest first:

  1. ``global_deny_hosts``   -- suffix match, wins over everything
  2. ``allow_hosts``         -- suffix match; empty means "the seed hosts"
  3. ``deny_patterns``       -- regex against the full URL

Host matching is suffix-based: an entry of ``example.com`` also admits
``www.example.com`` and ``static.example.com``. That is what you want for a
targeted crawl (image CDNs are almost always subdomains) but it does mean an
allow entry is a subtree, not a single host. List the narrowest host that still
covers your images.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml


def host_matches(host: str, entry: str) -> bool:
    """True if ``host`` is ``entry`` or a subdomain of it.

    Case-insensitive, tolerant of a trailing dot on the FQDN. Crucially this is
    a LABEL-wise test: ``notexample.com`` does not match ``example.com``, which
    a naive ``host.endswith(entry)`` would wrongly admit.
    """
    host = host.lower().rstrip(".")
    entry = entry.lower().rstrip(".").lstrip(".")
    if not host or not entry:
        return False
    return host == entry or host.endswith("." + entry)


@dataclass
class Vertical:
    name: str
    seeds: list[str]
    enabled: bool = False
    allow_hosts: list[str] = field(default_factory=list)
    # Subtrees to carve back OUT of allow_hosts. Suffix-matched, same as
    # allow_hosts, and checked after it.
    #
    # allow_hosts is a subtree allowlist, so `fosdem.org` also admits
    # video.fosdem.org and lists.fosdem.org -- a mailing-list archive with tens
    # of thousands of pages and no faces in any of them, which would quietly
    # consume the entire crawl budget. Without this the only workaround was a
    # regex over the full URL, which is both harder to read and easy to get
    # subtly wrong.
    deny_hosts: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)
    max_depth: int = 6
    max_pages: int = 50_000
    per_host_rps: float | None = None

    _deny_re: list[re.Pattern[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Compile once. A crawl evaluates these millions of times.
        self._deny_re = [re.compile(p) for p in self.deny_patterns]
        if not self.allow_hosts:
            # Empty allow list means "stay on the seed hosts" -- never "go
            # anywhere". A scope config that fails open is not a scope config.
            derived = {urlsplit(s).hostname for s in self.seeds}
            self.allow_hosts = sorted(h for h in derived if h)

    def host_allowed(self, host: str) -> bool:
        if any(host_matches(host, entry) for entry in self.deny_hosts):
            return False
        return any(host_matches(host, entry) for entry in self.allow_hosts)

    def denied_by_pattern(self, url: str) -> str | None:
        """Return the pattern that rejected this URL, or None. Returning the
        pattern rather than a bool is what makes a surprising drop debuggable."""
        for pat in self._deny_re:
            if pat.search(url):
                return pat.pattern
        return None


@dataclass
class SeedConfig:
    verticals: list[Vertical] = field(default_factory=list)
    global_deny_hosts: list[str] = field(default_factory=list)

    @property
    def active(self) -> list[Vertical]:
        return [v for v in self.verticals if v.enabled]

    def host_rate_limits(self) -> dict[str, float]:
        """``allow_host -> requests/sec`` for every active vertical that sets one.

        Handed to ``Politeness``, which is what finally makes
        ``Vertical.per_host_rps`` do something. If two verticals claim the same
        host at different rates, the slower one wins -- a promise to be gentle
        with a host is not cancelled by a second vertical being in a hurry.
        """
        out: dict[str, float] = {}
        for v in self.active:
            if not v.per_host_rps or v.per_host_rps <= 0:
                continue
            for host in v.allow_hosts:
                key = host.lower()
                out[key] = min(out.get(key, v.per_host_rps), v.per_host_rps)
        return out

    def globally_denied(self, host: str) -> bool:
        return any(host_matches(host, entry) for entry in self.global_deny_hosts)

    def in_scope(self, url: str, vertical: Vertical) -> tuple[bool, str]:
        """``(allowed, reason)``. The reason is always populated, including on
        success, so the crawl log can account for every URL it ever saw."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False, "bad_scheme"
        host = (parts.hostname or "").lower()
        if not host:
            return False, "no_host"
        if self.globally_denied(host):
            return False, "global_deny"
        if not vertical.host_allowed(host):
            return False, "off_vertical"
        if pat := vertical.denied_by_pattern(url):
            return False, f"deny_pattern:{pat}"
        return True, "ok"


def _coerce(raw: dict[str, Any]) -> Vertical:
    known = {
        "name",
        "seeds",
        "enabled",
        "allow_hosts",
        "deny_hosts",
        "deny_patterns",
        "max_depth",
        "max_pages",
        "per_host_rps",
    }
    if unknown := set(raw) - known:
        # Typos in a scope config fail silently and dangerously -- a misspelled
        # `deny_pattern` key means no denies at all. Refuse to load instead.
        raise ValueError(f"vertical {raw.get('name', '?')!r}: unknown keys {sorted(unknown)}")
    if not raw.get("name"):
        raise ValueError("every vertical needs a name")
    if not raw.get("seeds"):
        raise ValueError(f"vertical {raw['name']!r} has no seeds")
    return Vertical(**raw)


def load_seeds(path: Path) -> SeedConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy seeds.example.yaml to seeds.yaml and edit it."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = SeedConfig(
        verticals=[_coerce(v) for v in data.get("verticals", [])],
        global_deny_hosts=list(data.get("global_deny_hosts", [])),
    )

    # A seed that its own vertical would reject is always a config bug, and it
    # is invisible at runtime: the crawl just reports zero pages.
    for v in cfg.active:
        for seed in v.seeds:
            ok, reason = cfg.in_scope(seed, v)
            if not ok:
                raise ValueError(
                    f"vertical {v.name!r}: seed {seed} is out of its own scope ({reason})"
                )
    return cfg
