"""Crawl tier: frontier, politeness, extraction.

Deliberately empty of re-exports. Import concrete modules:

    from arc_search.crawler.frontier import Frontier
    from arc_search.crawler.extract import extract_images

Eagerly re-exporting here would make ``arc_search.crawler.extract`` -- pure
HTML parsing with no I/O -- transitively require httpx and protego, so the
fast unit tests could not run without the whole network stack installed.
Keep the tiers decoupled.
"""
