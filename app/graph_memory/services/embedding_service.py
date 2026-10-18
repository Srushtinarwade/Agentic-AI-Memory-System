"""CPU-only sentence embeddings loaded from disk; no runtime network download."""
from __future__ import annotations

import asyncio
from pathlib import Path


class EmbeddingService:
    def __init__(self, path: Path, dimensions: int = 384):
        self.path = Path(path)
        self.dimensions = dimensions
        self._model = None
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        if not self.path.is_dir():
            raise RuntimeError(
                f"Local embedding model not found: {self.path}. "
                "Run: python scripts/download_model.py"
            )

        def load():
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(self.path.resolve()), device="cpu", local_files_only=True)
            if model.get_sentence_embedding_dimension() != self.dimensions:
                raise RuntimeError("Incorrect embedding model dimensions; expected 384")
            return model

        self._model = await asyncio.to_thread(load)

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            raise RuntimeError("Embedding service is not started")
        async with self._lock:  # one CPU inference job at a time on laptops
            arr = await asyncio.to_thread(
                self._model.encode,
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
            )
        output = [[float(x) for x in row] for row in arr]
        if any(len(x) != self.dimensions for x in output):
            raise RuntimeError("Embedding dimension mismatch")
        return output

    async def shutdown(self) -> None:
        self._model = None
