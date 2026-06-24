# Legal-RAG-System
## Chunking strategy

I use page-aware recursive chunking with overlap. Each PDF page is extracted independently from the uploaded subfolder corpus, then split by paragraph boundaries first, sentence boundaries next, and finally word windows if a span is still too large.

Why this fits legal documents:

- Contracts are clause-dense and page citations matter, so page boundaries must be preserved.
- Legal questions often depend on nearby definitions or exceptions, so pure fixed-size chunks are too brittle.
- Overlap reduces the risk that a notice period or liability limit is split across chunk boundaries.

I keep chunk metadata for `document`, `page`, and `chunk_id` so the answer can cite the exact source page.

## Embedding model

I use OpenAI embeddings as the embedding layer, with `text-embedding-3-small` as the default model.

Why:

- For contract language, semantic similarity is important because user questions are often paraphrased.
- `text-embedding-3-small` gives a strong retrieval baseline for legal documents while keeping build cost and latency lower.
- I keep a local TF-IDF fallback for offline failure handling and tests, but it is not the primary retrieval path.

For a larger production corpus, I would keep the same embedding approach but add batching, caching, and possibly a second-stage reranker for harder legal queries.

## Vector store choice

I use FAISS with inner-product search over normalized dense vectors produced from OpenAI embeddings.

Why FAISS:

- Fast local approximate or exact similarity search.
- Simple file-free in-memory indexing for a clean demo.
- Easier to scale than a pure Python nearest-neighbor loop.

Why not Chroma or Pinecone here:

- Chroma is convenient but adds more moving parts than needed for a local assessment.
- Pinecone is a managed service, which is unnecessary because the task explicitly rewards a self-contained setup.

## Retrieval strategy

I use hybrid retrieval over the uploaded document set:

- Dense retrieval with FAISS over OpenAI embeddings.
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
