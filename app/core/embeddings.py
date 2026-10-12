from __future__ import annotations

from collections.abc import Sequence

from qdrant_client.models import SparseVector

from app.config import get_settings
from app.core.vector_utils import lexical_weights_to_sparse


class BGEM3Embedder:
    def __init__(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "FlagEmbedding is required: pip install -e '.[heavy]'"
            ) from exc
        settings = get_settings()
        kwargs: dict = {}
        if settings.use_fp16:
            kwargs["use_fp16"] = True
        if settings.embed_device:
            kwargs["device"] = settings.embed_device
        self._model = BGEM3FlagModel(settings.embedding_model_name, **kwargs)
        self.dim = int(settings.embed_dim)

    def encode(self, texts: Sequence[str]) -> tuple[list[list[float]], list[SparseVector]]:
        cleaned = [t if t and t.strip() else "empty" for t in texts]
        output = self._model.encode(
            cleaned,
            batch_size=8,
            max_length=2048,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"].tolist()
        sparse = [lexical_weights_to_sparse(lw) for lw in output["lexical_weights"]]
        return dense, sparse

    def encode_one(self, text: str) -> tuple[list[float], SparseVector]:
        dense, sparse = self.encode([text])
        return dense[0], sparse[0]
