"""Scope tests.

Scope bugs are silent in both directions: too tight and the crawl reports zero
pages, too loose and it wanders onto the open web. Neither shows up as an
exception. These tests are the only thing standing between the two.
"""

from __future__ import annotations

import pytest

from arc_search.crawler.seeds import SeedConfig, Vertical, host_matches, load_seeds

# --- host matching ---------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "entry", "expected"),
    [
        ("example.com", "example.com", True),
        ("www.example.com", "example.com", True),
        ("static.cdn.example.com", "example.com", True),
        ("EXAMPLE.COM", "example.com", True),
        ("example.com.", "example.com", True),  # trailing dot FQDN
        ("example.com", ".example.com", True),  # leading dot entry
        # The one that a naive endswith() gets wrong, and the reason this is a
        # named function instead of an inline expression.
        ("notexample.com", "example.com", False),
        ("example.com.evil.net", "example.com", False),
        ("example.org", "example.com", False),
        ("", "example.com", False),
        ("example.com", "", False),
    ],
)
def test_host_matching_is_label_wise(host, entry, expected):
    assert host_matches(host, entry) is expected


# --- vertical --------------------------------------------------------------


def test_empty_allow_hosts_derives_from_seeds_and_fails_closed():
    """An empty allow list must mean 'the seed hosts', never 'anywhere'."""
    v = Vertical(name="v", seeds=["https://a.example/x", "https://b.example/y"])
    assert v.allow_hosts == ["a.example", "b.example"]
    assert v.host_allowed("a.example")
    assert not v.host_allowed("c.example")


def test_deny_pattern_returns_the_pattern_not_just_false():
    v = Vertical(
        name="v",
        seeds=["https://a.example/"],
        deny_patterns=[r"\.pdf$", r"/login(/|$)"],
    )
    assert v.denied_by_pattern("https://a.example/paper.pdf") == r"\.pdf$"
    assert v.denied_by_pattern("https://a.example/login") == r"/login(/|$)"
    assert v.denied_by_pattern("https://a.example/photo.jpg") is None


# --- precedence ------------------------------------------------------------


def _cfg() -> SeedConfig:
    return SeedConfig(
        verticals=[
            Vertical(
                name="v",
                enabled=True,
                seeds=["https://conf.example/"],
                allow_hosts=["conf.example", "img.example"],
                deny_patterns=[r"\?", r"\.pdf$"],
            )
        ],
        global_deny_hosts=["facebook.com", "conf.example.tracker.net"],
    )


@pytest.mark.parametrize(
    ("url", "ok", "reason"),
    [
        ("https://conf.example/speakers", True, "ok"),
        ("https://img.example/a/b.jpg", True, "ok"),
        ("http://conf.example/speakers", True, "ok"),
        # off-vertical
        ("https://other.example/x", False, "off_vertical"),
        # deny patterns
        ("https://conf.example/x?page=2", False, "deny_pattern:\\?"),
        ("https://conf.example/paper.pdf", False, "deny_pattern:\\.pdf$"),
        # scheme + host guards
        ("ftp://conf.example/x", False, "bad_scheme"),
        ("javascript:alert(1)", False, "bad_scheme"),
        ("https:///nohost", False, "no_host"),
    ],
)
def test_in_scope(url, ok, reason):
    allowed, why = _cfg().in_scope(url, _cfg().verticals[0])
    assert (allowed, why) == (ok, reason)


def test_global_deny_beats_allow_hosts():
    """A globally denied host must lose even when a vertical allows its parent."""
    cfg = _cfg()
    v = cfg.verticals[0]
    # m.facebook.com is a subdomain of a globally denied host.
    assert cfg.in_scope("https://m.facebook.com/x", v) == (False, "global_deny")
    # And the pathological case: allowed by the vertical, denied globally.
    assert v.host_allowed("conf.example.tracker.net") is False
    cfg.verticals[0].allow_hosts.append("tracker.net")
    assert cfg.in_scope("https://conf.example.tracker.net/x", v) == (False, "global_deny")


# --- loading ---------------------------------------------------------------


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    """A misspelled `deny_pattern` silently means no denies at all."""
    p = tmp_path / "s.yaml"
    p.write_text(
        "verticals:\n"
        "  - name: v\n"
        "    seeds: [https://a.example/]\n"
        "    deny_pattern: ['x']\n",  # note: singular, a typo
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_seeds(p)


def test_seed_outside_its_own_scope_is_a_load_error(tmp_path):
    """Otherwise the crawl just quietly reports zero pages."""
    p = tmp_path / "s.yaml"
    p.write_text(
        "verticals:\n"
        "  - name: v\n"
        "    enabled: true\n"
        "    seeds: ['https://a.example/list?page=1']\n"
        "    allow_hosts: [a.example]\n"
        "    deny_patterns: ['\\?']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out of its own scope"):
        load_seeds(p)


def test_disabled_vertical_is_not_scope_checked(tmp_path):
    """Tier-2 entries are parked, half-tuned. They must not block a load."""
    p = tmp_path / "s.yaml"
    p.write_text(
        "verticals:\n"
        "  - name: parked\n"
        "    enabled: false\n"
        "    seeds: ['https://a.example/list?page=1']\n"
        "    deny_patterns: ['\\?']\n",
        encoding="utf-8",
    )
    cfg = load_seeds(p)
    assert cfg.active == []


def test_missing_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"seeds\.example\.yaml"):
        load_seeds(tmp_path / "nope.yaml")


# --- the real config -------------------------------------------------------


def test_shipped_example_config_loads():
    """seeds.example.yaml is documentation people copy. It must parse."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "seeds.example.yaml"
    cfg = load_seeds(example)
    assert cfg.verticals
    assert cfg.active == []  # the example ships with everything disabled
    assert "facebook.com" in cfg.global_deny_hosts


# --- deny_hosts ------------------------------------------------------------


def test_deny_hosts_carves_a_subtree_out_of_allow_hosts():
    """allow_hosts matches subtrees, which admits far more than intended.

    Observed live: `allow_hosts: [fosdem.org]` queued video.fosdem.org,
    lists.fosdem.org and ksp.fosdem.org. A mailing-list archive is tens of
    thousands of crawlable pages with no faces in any of them -- enough to
    consume a whole crawl budget and produce nothing.
    """
    v = Vertical(
        name="v",
        seeds=["https://example.com/"],
        allow_hosts=["example.com"],
        deny_hosts=["lists.example.com"],
    )
    assert v.host_allowed("example.com")
    assert v.host_allowed("static.example.com")
    assert not v.host_allowed("lists.example.com")
    # ...and the deny is itself a subtree.
    assert not v.host_allowed("archive.lists.example.com")


def test_deny_hosts_reports_off_vertical_at_the_scope_check():
    cfg = SeedConfig(
        verticals=[
            Vertical(
                name="v",
                enabled=True,
                seeds=["https://example.com/"],
                allow_hosts=["example.com"],
                deny_hosts=["lists.example.com"],
            )
        ]
    )
    v = cfg.verticals[0]
    assert cfg.in_scope("https://example.com/a", v) == (True, "ok")
    assert cfg.in_scope("https://lists.example.com/a", v) == (False, "off_vertical")


def test_deny_hosts_is_a_recognised_key():
    """Unknown keys are rejected outright, so a new field must be registered."""
    import textwrap
    from pathlib import Path
    from tempfile import mkdtemp

    p = Path(mkdtemp()) / "s.yaml"
    p.write_text(
        textwrap.dedent("""
            verticals:
              - name: v
                enabled: true
                seeds: ['https://example.com/']
                allow_hosts: [example.com]
                deny_hosts: [lists.example.com]
        """),
        encoding="utf-8",
    )
    cfg = load_seeds(p)
    assert cfg.verticals[0].deny_hosts == ["lists.example.com"]


def test_the_real_seeds_file_excludes_the_faceless_subdomains():
    """Regression on the actual config, not just the model."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "seeds.yaml"
    if not path.exists():
        import pytest as _pytest

        _pytest.skip("seeds.yaml is gitignored; only present on a configured checkout")
    cfg = load_seeds(path)
    fosdem = next(v for v in cfg.verticals if v.name == "fosdem")
    for host in ("lists.fosdem.org", "video.fosdem.org", "ksp.fosdem.org"):
        assert not fosdem.host_allowed(host), host
    assert fosdem.host_allowed("archive.fosdem.org")
