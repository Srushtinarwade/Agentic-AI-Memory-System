"""Run ONCE while online. The FastAPI app only loads the local CPU model."""
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
    target.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading sentence-transformers/all-MiniLM-L6-v2 (one-time setup)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    actual = model.get_sentence_embedding_dimension()
    if actual != 384:
        raise RuntimeError(f"Unexpected embedding dimensions {actual}; expected 384")
    model.save_pretrained(str(target))
    print(f"Model saved to {target}")
    print("Embedding dimensions: 384; runtime device: CPU")


if __name__ == "__main__":
    main()
