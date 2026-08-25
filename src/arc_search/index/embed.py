"""Whole-image embedding: scene and text. ADR-005, plan-005 Phase 2.

TWO VECTORS PER IMAGE, AND THE SECOND ONE IS NOT WHAT ITS NAME SUGGESTS
----------------------------------------------------------------------
``scene``  DINOv2-class image embedding. Image -> image, whole-scene similarity.
``text``   SigLIP-class **image** embedding. This is the model's IMAGE tower, so
           what is stored is a picture's position in a joint text/image space,
           which a *text* embedding can then be searched against.

           It is NOT an embedding of the alt text. That is a different and also
           useful thing which this module does not do -- worth saying out loud,
           because "the text vector" reads like it should be.

🔴 ORDERING: torch must load its CUDA DLLs before faces.register_cuda_runtime()
prepends the onnxruntime cu13 directories to PATH. Measured: insightface-then-
torch dies with WinError 127 on cudnn_cnn64_9.dll; torch-then-insightface works
and both keep working. ``register_cuda_runtime`` imports torch itself to make
that self-enforcing, so nothing here has to remember it -- but importing this
module also gets torch loaded early, which is the belt to that braces.

🔴 THE DEVICE IS REPORTED, NOT ASSUMED. ``effective_device()`` asks the loaded
parameters where they actually are. This project has already shipped the other
version twice: the crawler logged its requested rate while running at half of
it, and onnxruntime silently fell back to CPU at 1/12th speed while the config
said CUDA. A config value is a request, not an observation.

NO QUALITY GATE
---------------
Unlike faces, every image gets a vector. There is no ``too_small`` /
``bad_pose`` equivalent, because "is this image worth looking at" is not a
question the scene model can answer -- and the one gate that exists,
``min_image_dim``, is applied at crawl time and had its derivation voided by
ADR-005. See plan-005 Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from arc_search.config import EmbedSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ImageVectors:
    """The two whole-image vectors for one image, both L2-normalized."""

    scene: np.ndarray  # DINOv2-class, scene_dim
    text: np.ndarray  # SigLIP-class image tower, text_dim


class ImageEmbedder:
    """Scene + text embedding. Models are lazy-loaded and swappable by config."""

    def __init__(self, cfg: EmbedSettings | None = None) -> None:
        self._cfg = cfg or EmbedSettings()
        self._scene = None
        self._scene_proc = None
        self._text = None
        self._text_proc = None

    # -- loading -----------------------------------------------------------

    def _ensure(self):
        if self._scene is not None:
            return

        # Before transformers, and therefore before anything else grabs CUDA.
        # See the module docstring: this ordering is load-bearing on Windows.
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        cfg = self._cfg
        device = cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            # Loud, and it does not silently continue on CPU pretending
            # otherwise. At 30M images the difference is not a slow afternoon.
            log.warning(
                "embed.cuda_unavailable",
                detail="CUDA was requested but torch.cuda.is_available() is False; "
                "falling back to CPU. Expect a ~20x throughput loss. On Windows "
                "check that torch came from the cu128 index -- the default PyPI "
                "wheel is CPU-only and installs without complaint.",
            )
            device = "cpu"

        self._scene_proc = AutoImageProcessor.from_pretrained(cfg.scene_model)
        self._scene = AutoModel.from_pretrained(cfg.scene_model).to(device).eval()
        self._text_proc = AutoProcessor.from_pretrained(cfg.text_model)
        self._text = AutoModel.from_pretrained(cfg.text_model).to(device).eval()
        self._device = device

        dims = self.dims()
        log.info(
            "embed.models_ready",
            scene_model=cfg.scene_model,
            text_model=cfg.text_model,
            requested_device=cfg.device,
            effective_device=self.effective_device(),
            scene_dim=dims[0],
            text_dim=dims[1],
        )
        return self._scene

    def effective_device(self) -> str:
        """Where the weights ACTUALLY are. Not what was asked for."""
        if self._scene is None:
            return "unloaded"
        return str(next(self._scene.parameters()).device)

    def dims(self) -> tuple[int, int]:
        """(scene_dim, text_dim), read off the loaded models.

        Measured rather than configured, so a model swap cannot silently
        disagree with the Qdrant collection it is being written into. Both are
        768 for dinov2-base / siglip2-base; DINOv3 ViT-L would be 1024 and that
        moves the scale target, so it must not pass unnoticed.
        """
        self._ensure()
        return (
            int(self._scene.config.hidden_size),
            int(getattr(self._text.config, "vision_config", self._text.config).hidden_size),
        )

    # -- embedding ---------------------------------------------------------

    @staticmethod
    def _pooled(out):
        """transformers 5.x returns an output object, not a tensor.

        Measured on siglip2: BaseModelOutputWithPooling with
        {last_hidden_state (B,576,768), pooler_output (B,768)}. `or` on tensors
        raises -- ambiguous truthiness -- so this tests for None explicitly.
        """
        import torch

        if torch.is_tensor(out):
            return out
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state[:, 0]

    def _norm(self, v) -> np.ndarray:
        import torch

        return torch.nn.functional.normalize(self._pooled(v).float(), dim=-1).cpu().numpy()

    def embed_images(self, images: list) -> list[ImageVectors]:
        """Embed a batch of PIL images. Returns one ImageVectors per input.

        Batched because the GPU wants it: measured 70 img/s at batch 1 against
        179 img/s at batch 7 for the scene model. The crawl loop produces images
        one at a time, so whatever calls this has to accumulate.
        """
        if not images:
            return []
        import torch

        self._ensure()
        with torch.inference_mode():
            s = self._norm(
                self._scene(**self._scene_proc(images=images, return_tensors="pt").to(self._device))
            )
            t = self._norm(
                self._text.get_image_features(
                    **self._text_proc(images=images, return_tensors="pt").to(self._device)
                )
            )
        return [ImageVectors(scene=s[i], text=t[i]) for i in range(len(images))]

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed text queries into the SAME space as the ``text`` image vector.

        This is the query side of text->image search. The vectors returned here
        are compared against the ``text`` vectors stored per image.
        """
        import torch

        self._ensure()
        with torch.inference_mode():
            return self._norm(
                self._text.get_text_features(
                    **self._text_proc(text=texts, padding="max_length", return_tensors="pt").to(
                        self._device
                    )
                )
            )

    def text_logit_params(self) -> tuple[float, float]:
        """(logit_scale, logit_bias) for turning cosine into a probability.

        SigLIP is trained with a sigmoid loss, so raw cosine is not its scale:
        score = sigmoid(cosine * logit_scale + logit_bias). Measured on
        siglip2-base: scale 112.85, bias -16.77. Read raw, every score sits
        around 0.1 and a working model looks undiscriminating -- that cost a
        false alarm during bring-up, which is why this is exposed rather than
        left for a caller to rediscover.
        """
        import torch

        self._ensure()
        scale = getattr(self._text, "logit_scale", None)
        bias = getattr(self._text, "logit_bias", None)
        if scale is None:
            return (1.0, 0.0)
        return (
            float(torch.as_tensor(scale).exp().item()),
            float(torch.as_tensor(bias).item()) if bias is not None else 0.0,
        )
