#  ML Engineer Tasks

This repository is a complete submission for the four required assessment sections.

## What’s Included

- `ANSWERS.md`: written diagnosis, post-mortem, and systems-design answers.
- `DESIGN.md`: architecture notes for the legal RAG pipeline.
- `src/rag_pipeline.py`: offline legal-question answering pipeline with citations.
- `data/sample_pdfs/`: uploaded legal PDFs organized by document type in subfolders.
- `data/rag_eval.json`: manual QA pairs for evaluating retrieval quality.
- `src/classifier/`: CPU-friendly ticket classifier and benchmark.
- `tests/`: automated checks for the RAG and classifier flows.

## Setup

The project is self-contained and uses local libraries only.
The RAG corpus is sourced from the uploaded PDFs in `data/sample_pdfs/`, and the vector index and classifier artifacts are generated locally during the build/train steps.

Recommended environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run

Use the project venv explicitly so the commands run in the same environment that has PyMuPDF, OpenAI, and pytest installed.

Build the RAG index and run the retrieval evaluation:

```bash
.venv/bin/python -m src.rag_pipeline --build
.venv/bin/python -m src.rag_pipeline --eval
```

Train and evaluate the ticket classifier:

```bash
.venv/bin/python -m src.classifier.train
.venv/bin/python -m src.classifier.evaluate
.venv/bin/python -m src.classifier.latency_test
```

Run the full verification suite:

```bash
.venv/bin/python -m pytest -q
```

Or run the common flow with:

```bash
make all
```

## Demo Video

Optional walkthrough recording:

- Loom: [Recording](https://www.loom.com/share/9e6d6c1269e94ab387b223212254bbd4)

## Notes

- The RAG pipeline now uses PyMuPDF for PDF extraction and OpenAI embeddings for retrieval.
- The default embedding model is `text-embedding-3-small`, which I chose as a faster, cheaper starter option while keeping good retrieval quality.
- Chunking is recursive and structure-aware: it prefers paragraph and sentence boundaries first, then falls back to word windows with overlap. That preserves clause context better than a fixed-length splitter.
- A small overlap is kept between neighboring chunks so legal exceptions, definitions, and cross-references are less likely to be split apart.
- No API keys are required for the classifier, but the RAG build step does require `OPENAI_API_KEY`.
- The classifier uses a CPU-friendly scikit-learn model to satisfy the 500ms constraint.
- Section 5 is optional and not included.
