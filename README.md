# Artikate ML Engineer Tasks

This repository contains a complete submission for the four required assessment sections.

## Contents

- `ANSWERS.md`: written diagnosis and post-mortem for the LLM pipeline failures.
- `src/rag_pipeline.py`: production-style RAG pipeline for legal PDFs.
- `data/sample_pdfs/`: three sample contract PDFs generated locally.
- `data/rag_eval.json`: 10 manual QA pairs for evaluation.
- `src/classifier/`: CPU-friendly ticket classifier and benchmark.
- `DESIGN.md`: architecture decisions for the RAG system.

## Setup

The project is self-contained and uses local libraries only.
The RAG sample PDFs, vector index, and classifier artifacts are generated locally during the build/train steps.

### Recommended environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run

Generate sample PDFs, build the RAG index, and run the evaluation harness:

```bash
python -m src.rag_pipeline --build
python -m src.rag_pipeline --eval
```

Train and evaluate the ticket classifier:

```bash
python -m src.classifier.train
python -m src.classifier.evaluate
python -m src.classifier.latency_test
```

Or run the common flow with:

```bash
make all
```

## Notes

- No API keys are required.
- The RAG pipeline uses local TF-IDF embeddings plus FAISS and runs fully offline.
- The classifier uses a CPU-friendly scikit-learn model to satisfy the 500ms constraint.
- Section 5 is optional and not included.
