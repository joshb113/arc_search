# /focus [tier]
Deep-load a tier's active files into context.

Accepted values for [tier]: crawler, index, serve, eval

- **crawler** → src/arc_search/crawler/ + tests/test_{frontier,extract,fetch,seeds,politeness,run}.py + seeds.yaml
- **index**   → src/arc_search/index/ + sql/schema.sql + tests/test_{dedup,store}.py
- **serve**   → src/arc_search/serve/ + the query-path plan steps
- **eval**    → src/arc_search/eval/ + the UNCALIBRATED block in config.py + any labeled-set research

After loading, report a one-paragraph summary of the tier's current state and
the next unblocked action.

Note: the tiers are deliberately decoupled — `arc_search.crawler.extract` must
not require opencv or onnxruntime. If a focus reveals a cross-tier import that
breaks that, say so.
