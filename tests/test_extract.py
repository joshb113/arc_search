"""Image and link extraction.

eye_of_web had zero repo-wide matches for srcset, ld+json, or <picture>, and used
og:image only inside platform scrapers. On a general crawl it systematically took
the lowest-resolution candidate on every modern responsive page.
"""

from __future__ import annotations

from arc_search.crawler.extract import extract_images, extract_links, page_title

BASE = "https://example.com/gallery/page.html"


def _urls(html: str) -> list[str]:
    return [f.url for f in extract_images(html, BASE)]


class TestSrcset:
    def test_picks_largest_w_descriptor(self):
        html = """<img srcset="/s.jpg 320w, /m.jpg 800w, /l.jpg 1600w" src="/fallback.jpg">"""
        urls = _urls(html)
        assert "https://example.com/l.jpg" in urls
        assert urls.index("https://example.com/l.jpg") < urls.index(
            "https://example.com/fallback.jpg"
        )

    def test_picks_largest_x_descriptor(self):
        html = """<img srcset="/1x.jpg 1x, /3x.jpg 3x">"""
        assert "https://example.com/3x.jpg" in _urls(html)

    def test_bare_srcset_entry(self):
        html = """<img srcset="/only.jpg">"""
        assert "https://example.com/only.jpg" in _urls(html)

    def test_picture_source_beats_img_src(self):
        html = """
        <picture>
          <source srcset="/big.jpg 2000w" type="image/webp">
          <img src="/small.jpg">
        </picture>"""
        urls = _urls(html)
        assert urls.index("https://example.com/big.jpg") < urls.index(
            "https://example.com/small.jpg"
        )


class TestStructured:
    def test_og_image_extracted_first(self):
        html = """
        <meta property="og:image" content="https://cdn.example.com/hero.jpg">
        <img src="/thumb.jpg">"""
        urls = _urls(html)
        assert urls[0] == "https://cdn.example.com/hero.jpg"

    def test_jsonld_string_image(self):
        html = """<script type="application/ld+json">
        {"@type":"Article","image":"https://cdn.example.com/a.jpg"}</script>"""
        assert "https://cdn.example.com/a.jpg" in _urls(html)

    def test_jsonld_list_and_nested_object(self):
        html = """<script type="application/ld+json">
        {"@graph":[{"image":["/one.jpg","/two.jpg"]},
                   {"image":{"@type":"ImageObject","url":"/three.jpg"}}]}</script>"""
        urls = _urls(html)
        for name in ("one", "two", "three"):
            assert f"https://example.com/{name}.jpg" in urls

    def test_malformed_jsonld_does_not_raise(self):
        html = """<script type="application/ld+json">{not valid json</script>
                  <img src="/ok.jpg">"""
        assert "https://example.com/ok.jpg" in _urls(html)


class TestFiltering:
    def test_lazy_load_attributes(self):
        html = """<img data-src="/lazy.jpg"><img data-original="/orig.jpg">"""
        urls = _urls(html)
        assert "https://example.com/lazy.jpg" in urls
        assert "https://example.com/orig.jpg" in urls

    def test_skips_svg_gif_ico(self):
        html = """<img src="/a.svg"><img src="/b.gif"><img src="/c.ico"><img src="/d.jpg">"""
        assert _urls(html) == ["https://example.com/d.jpg"]

    def test_skips_data_uri(self):
        html = """<img src="data:image/png;base64,iVBORw0KGgo="><img src="/real.jpg">"""
        assert _urls(html) == ["https://example.com/real.jpg"]

    def test_deduplicates_within_page(self):
        html = """<img src="/same.jpg"><img src="/same.jpg">"""
        assert _urls(html) == ["https://example.com/same.jpg"]

    def test_relative_urls_resolved_against_base(self):
        html = """<img src="../up.jpg">"""
        assert _urls(html) == ["https://example.com/up.jpg"]

    def test_alt_text_captured(self):
        html = """<img src="/a.jpg" alt="A person at an event">"""
        found = extract_images(html, BASE)
        assert found[0].alt == "A person at an event"


class TestLinks:
    def test_basic_extraction_and_resolution(self):
        html = """<a href="/x">x</a><a href="https://other.com/y">y</a>"""
        assert extract_links(html, BASE) == ["https://example.com/x", "https://other.com/y"]

    def test_skips_nofollow(self):
        html = """<a href="/keep">k</a><a href="/skip" rel="nofollow">s</a>"""
        assert extract_links(html, BASE) == ["https://example.com/keep"]

    def test_skips_non_http_schemes(self):
        html = """<a href="mailto:a@b.c">m</a><a href="javascript:void(0)">j</a>
                  <a href="/ok">o</a>"""
        assert extract_links(html, BASE) == ["https://example.com/ok"]

    def test_title(self):
        assert page_title("<title> Hello </title>") == "Hello"
        assert page_title("<html></html>") is None
