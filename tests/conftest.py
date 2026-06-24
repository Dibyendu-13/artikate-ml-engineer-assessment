from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_assets() -> None:
    from src.rag_pipeline import INDEX_DIR, RAGPipeline, load_env_file
    from src.classifier.train import main as train_classifier

    load_env_file()
    if not (INDEX_DIR / "chunks.faiss").exists() or not (INDEX_DIR / "chunks.json").exists():
        pipeline = RAGPipeline.build()
        pipeline.save()
    train_classifier()


_ensure_assets()
