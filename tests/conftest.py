from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_assets() -> None:
    from src.rag_pipeline import RAGPipeline
    from src.classifier.train import main as train_classifier
    from src.utils import DATA_DIR

    if not (DATA_DIR / "index" / "chunks.faiss").exists():
        pipeline = RAGPipeline.build()
        pipeline.save()
    if not (DATA_DIR / "classifier" / "model.joblib").exists():
        train_classifier()


_ensure_assets()
