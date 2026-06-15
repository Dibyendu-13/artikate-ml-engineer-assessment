# RAG Design

## Chunking strategy

I use page-aware chunking with sentence-window overlap. Each PDF page is extracted independently from the uploaded subfolder corpus, then split into chunks of roughly 180 to 250 words with a 1-sentence overlap.

Why this fits legal documents:

- Contracts are clause-dense and page citations matter, so page boundaries must be preserved.
- Legal questions often depend on nearby definitions or exceptions, so pure fixed-size chunks are too brittle.
- Overlap reduces the risk that a notice period or liability limit is split across chunk boundaries.

I keep chunk metadata for `document`, `page`, and `chunk_id` so the answer can cite the exact source page.

## Embedding model

I use TF-IDF features as the embedding layer, with word and character n-grams.

Why:

- The repository must run locally without downloads or API keys.
- For contract language, lexical overlap is strong for clause lookup questions such as "notice period" or "limitation of liability".
- Character n-grams improve robustness to formatting differences, numbering, and slight OCR noise.

For a larger production corpus, I would replace TF-IDF with a sentence-transformer or domain-tuned embedding model, but that would add external download and inference overhead that is unnecessary for this assessment.

## Vector store choice

I use FAISS with inner-product search over normalized dense vectors produced from a truncated SVD projection of TF-IDF features.

Why FAISS:

- Fast local approximate or exact similarity search.
- Simple file-free in-memory indexing for a clean demo.
- Easier to scale than a pure Python nearest-neighbor loop.

Why not Chroma or Pinecone here:

- Chroma is convenient but adds more moving parts than needed for a local assessment.
- Pinecone is a managed service, which is unnecessary because the task explicitly rewards a self-contained setup.

## Retrieval strategy

I use hybrid retrieval over the uploaded document set:

- First-pass lexical similarity with TF-IDF.
- Dense projection into FAISS for ranking.
- A lightweight second-pass reranker that boosts chunks containing exact legal cue phrases such as "notice period", "limitation of liability", "termination", or matched monetary amounts.

Why:

- Legal questions are precise and clause-oriented.
- Pure top-k semantic retrieval can miss exact clause text.
- A small reranking heuristic is cheap and improves recall for clause lookup questions without requiring a second model.

## Hallucination mitigation

I implement answer refusal when the top retrieved context is too weak.

Mechanism:

- compute retrieval confidence from the top similarity scores and score gap
- if the best chunk score is low or the gap between top candidates is too small, return a refusal instead of an answer
- require the generated answer to quote or paraphrase only from the retrieved chunk set

Why:

- In legal QA, wrong confident answers are worse than refusal.
- Confidence gating is deterministic, testable, and easy to explain.

## Scaling to 50,000 documents

If the corpus grows to 50,000 documents:

- Ingestion becomes the first bottleneck. I would parallelize PDF extraction, cache per-page text, and store structured metadata in a database.
- Embedding generation becomes expensive. I would switch from TF-IDF to batched embedding inference with a smaller transformer, GPU if available, and incremental indexing.
- Retrieval quality would need better recall. I would use a two-stage system: BM25 + vector retrieval, then a cross-encoder reranker.
- FAISS index size would grow. I would move to IVF or HNSW indexing, persist the index to disk, and partition by document type or business unit.
- Evaluation would need to expand. I would maintain a stratified benchmark of legal questions by clause type and measure recall@k plus citation accuracy.
