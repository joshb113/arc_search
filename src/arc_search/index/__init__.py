"""Index tier: dedup, face extraction, storage.

Deliberately empty of re-exports. Import concrete modules:

    from arc_search.index.dedup import Deduper
    from arc_search.index.faces import FaceExtractor

``dedup`` is pure (numpy only) and must stay testable without onnxruntime,
opencv, or a CUDA device present.
"""
