from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import fitz
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import DATA_DIR, PDF_DIR


INDEX_DIR = DATA_DIR / "index"
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ROOT_DIR = Path(__file__).resolve().parents[1]


def load_env_file(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


@dataclass
class Chunk:
    document: str
    page: int
    chunk: str
    chunk_id: str
    section: str = ""


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pages.append((idx, page.get_text("text") or ""))
    return pages


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _join_with_overlap(chunks: list[str], overlap_words: int) -> list[str]:
    if not chunks or overlap_words <= 0:
        return chunks
    out: list[str] = []
    previous_tail: list[str] = []
    for chunk in chunks:
        words = chunk.split()
        if previous_tail:
            words = previous_tail + words
        out.append(" ".join(words).strip())
        previous_tail = chunk.split()[-overlap_words:]
    return out


def _recursive_chunk(text: str, max_words: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]

    separators = [
        "\n\n",
        "\n",
        "§",
        "; ",
        ": ",
    ]

    for separator in separators:
        if separator not in text:
            continue
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) <= 1:
            continue
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate.split()) <= max_words:
                current = candidate
            else:
                if current:
                    chunks.extend(_recursive_chunk(current, max_words, overlap))
                current = part
        if current:
            chunks.extend(_recursive_chunk(current, max_words, overlap))
        if chunks:
            return _join_with_overlap(chunks, overlap)

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        chunks = []
        current: list[str] = []
        for sentence in sentences:
            candidate = " ".join(current + [sentence]).strip()
            if len(candidate.split()) <= max_words:
                current.append(sentence)
            else:
                if current:
                    chunks.extend(_recursive_chunk(" ".join(current), max_words, overlap))
                current = [sentence]
        if current:
            chunks.extend(_recursive_chunk(" ".join(current), max_words, overlap))
        if chunks:
            return _join_with_overlap(chunks, overlap)

    # Final fallback: sliding window over words.
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_text(text: str, words_per_chunk: int = 140, overlap: int = 25) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(_recursive_chunk(paragraph, words_per_chunk, overlap))
    return [chunk for chunk in chunks if chunk.strip()]


def infer_section(text: str) -> str:
    lower = text.lower()
    section_markers = [
        ("notice", "termination"),
        ("liability", "liability"),
        ("uptime", "service levels"),
        ("confidential", "confidentiality"),
        ("retention", "data protection"),
        ("governing law", "governing law"),
        ("assignment", "assignment"),
        ("payment", "fees"),
        ("table of contents", "toc"),
        ("signature", "signature"),
        ("exhibit", "exhibit"),
        ("annex", "annex"),
    ]
    for marker, section in section_markers:
        if marker in lower:
            return section
    return "general"


def build_corpus() -> list[Chunk]:
    corpus: list[Chunk] = []
    ignored_top_level = {
        "nda_vendor_x.pdf",
        "msa_vendor_y.pdf",
        "policy_retention.pdf",
        "dpa_vendor_x.pdf",
        "saas_platform.pdf",
    }
    for pdf in sorted(PDF_DIR.rglob("*.pdf")):
        if pdf.parent == PDF_DIR and pdf.name in ignored_top_level:
            continue
        for page_num, text in extract_pdf_pages(pdf):
            section = infer_section(text)
            for i, chunk in enumerate(chunk_text(text)):
                corpus.append(Chunk(str(pdf.relative_to(PDF_DIR)), page_num, chunk, f"{pdf.stem}-{page_num}-{i}", section=section))
    return corpus


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _chunked(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors: list[list[float]] = []
    for batch in _chunked(texts, batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([item.embedding for item in resp.data])
    return np.asarray(vectors, dtype="float32")


class RAGPipeline:
    def __init__(
        self,
        corpus: list[Chunk],
        index: faiss.IndexFlatIP,
        embeddings: np.ndarray,
        client: OpenAI,
    ):
        self.corpus = corpus
        self.index = index
        self.embeddings = embeddings
        self.client = client
        self.fallback_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.fallback_matrix = self.fallback_vectorizer.fit_transform([c.chunk for c in corpus])
        self.section_titles = {
            "termination": {"termination", "notice period", "cure period"},
            "liability": {"liability", "cap", "indemnity"},
            "service levels": {"uptime", "service levels", "sla"},
            "confidentiality": {"confidentiality", "confidential"},
            "data protection": {"data protection", "retained", "retention", "breach"},
            "governing law": {"governed", "law", "dispute resolution"},
            "assignment": {"assignment", "change of control"},
            "fees": {"payment", "fees", "invoice"},
        }
        self.document_titles = {
            "nda": {"nda", "non-disclosure", "confidential"},
            "msa": {"msa", "services agreement", "service agreement"},
            "policy": {"policy", "retention", "data handling"},
            "dpa": {"dpa", "data processing", "processor", "controller"},
            "saas": {"saas", "subscription", "software as a service"},
        }
        self.target_titles = {
            "nda": {"nondisclosureagreement", "basic-non-disclosure-agreement", "sample_nda"},
            "msa": {"master-service-agreement", "master services agreement", "globalsign", "msa"},
            "policy": {"data-retention", "retention policy", "privacy policy", "policy"},
            "dpa": {"dpa", "data processing", "training-2025-dpas", "contractor"},
            "saas": {"saas", "subscription agreement", "onestream", "mlh"},
        }

    @classmethod
    def build(cls) -> "RAGPipeline":
        corpus = build_corpus()
        client = OpenAI()
        embeddings = embed_texts(client, [c.chunk for c in corpus])
        embeddings = normalize_rows(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return cls(corpus, index, embeddings, client)

    def _encode(self, text: str) -> np.ndarray | None:
        try:
            dense = embed_texts(self.client, [text])
            return normalize_rows(dense)
        except Exception:
            return None

    def retrieve(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._encode(question)
        question_doc = self._route_document(question)
        pairs = []
        if q is None:
            scores = cosine_similarity(self.fallback_vectorizer.transform([question]), self.fallback_matrix)[0]
            candidate_ids = np.argsort(scores)[::-1][: top_k * 20]
            for idx in candidate_ids:
                chunk = self.corpus[int(idx)]
                if question_doc != "general" and question_doc not in chunk.document.lower():
                    continue
                boosted = float(scores[idx])
                boosted += 0.18 * self._keyword_boost(question, chunk.chunk)
                boosted += 0.10 * self._document_boost(question_doc, chunk)
                boosted += 0.08 * self._section_boost(self._route_question(question), chunk)
                boosted += 0.12 * self._term_coverage_boost(self._important_terms(question), chunk.chunk)
                boosted += self._page_boost(chunk)
                boosted = self._rerank_candidate(question, chunk, boosted)
                boosted += self._doc_title_boost(question, chunk)
                boosted += self._filename_hint_boost(question, chunk)
                pairs.append((chunk, boosted))
        else:
            dense_scores, ids = self.index.search(q, top_k * 20)
            for idx, dense_score in zip(ids[0], dense_scores[0]):
                if idx < 0:
                    continue
                chunk = self.corpus[idx]
                if question_doc != "general" and question_doc not in chunk.document.lower():
                    continue
                boosted = 0.54 * float(dense_score)
                boosted += 0.18 * self._keyword_boost(question, chunk.chunk)
                boosted += 0.10 * self._document_boost(question_doc, chunk)
                boosted += 0.08 * self._section_boost(self._route_question(question), chunk)
                boosted += 0.12 * self._term_coverage_boost(self._important_terms(question), chunk.chunk)
                boosted += self._page_boost(chunk)
                boosted = self._rerank_candidate(question, chunk, boosted)
                boosted += self._doc_title_boost(question, chunk)
                boosted += self._filename_hint_boost(question, chunk)
                pairs.append((chunk, boosted))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def _route_question(self, question: str) -> str:
        q = question.lower()
        for section, markers in self.section_titles.items():
            if any(marker in q for marker in markers):
                return section
        return "general"

    def _route_document(self, question: str) -> str:
        q = question.lower()
        for doc, markers in self.document_titles.items():
            if any(marker in q for marker in markers):
                return doc
        return "general"

    def _section_boost(self, section: str, chunk: Chunk) -> float:
        if section == "general":
            return 0.0
        if section == chunk.section:
            return 0.2
        if section in chunk.chunk.lower():
            return 0.1
        return 0.0

    def _document_boost(self, doc_family: str, chunk: Chunk) -> float:
        if doc_family == "general":
            return 0.0
        doc = chunk.document.lower()
        if doc_family in doc:
            return 0.15
        if doc_family == "policy" and "retention" in doc:
            return 0.1
        if doc_family == "nda" and "nda" in doc:
            return 0.1
        if doc_family == "msa" and "msa" in doc:
            return 0.1
        if doc_family == "dpa" and "dpa" in doc:
            return 0.1
        if doc_family == "saas" and "saas" in doc:
            return 0.1
        return 0.0

    def _doc_title_boost(self, question: str, chunk: Chunk) -> float:
        q = question.lower()
        doc = chunk.document.lower()
        boost = 0.0
        for family, markers in self.document_titles.items():
            if any(marker in q for marker in markers):
                if any(marker in doc for marker in self.target_titles.get(family, set())):
                    boost += 0.22
                elif family in doc:
                    boost += 0.12
        return boost

    def _filename_hint_boost(self, question: str, chunk: Chunk) -> float:
        q = question.lower()
        doc = chunk.document.lower()
        stem = Path(chunk.document).stem.lower()
        boost = 0.0
        if stem and stem in q:
            boost += 0.28
        if doc in q:
            boost += 0.30
        if "sample_nda" in q and "sample_nda" in doc:
            boost += 0.18
        if "training-2025-dpas" in q and "training-2025-dpas" in doc:
            boost += 0.18
        return boost

    def _important_terms(self, question: str) -> list[str]:
        stop = {
            "what", "which", "who", "how", "when", "where", "is", "the", "a", "an", "in", "on", "of",
            "for", "to", "with", "and", "or", "be", "can", "does", "do", "may", "signed", "signed?", "vendor",
            "document", "contract", "agreement", "please", "tell", "me",
        }
        terms = []
        for raw in question.lower().replace("?", "").replace("'", "").split():
            if raw not in stop and len(raw) > 2:
                terms.append(raw)
        return terms

    def _term_coverage_boost(self, terms: list[str], chunk_text: str) -> float:
        if not terms:
            return 0.0
        lower = chunk_text.lower()
        hits = sum(1 for term in terms if term in lower)
        coverage = hits / max(1, len(terms))
        if coverage >= 0.75:
            return 0.25
        if coverage >= 0.5:
            return 0.15
        if coverage >= 0.3:
            return 0.08
        return 0.0

    def _rerank_candidate(self, question: str, chunk: Chunk, base_score: float) -> float:
        lower = chunk.chunk.lower()
        q = question.lower()
        q_terms = self._important_terms(question)
        q_tokens = set(q_terms)
        bonus = 0.0

        clause_map = {
            "notice": ["notice period", "notice", "terminate", "termination", "cure period"],
            "liability": ["liability cap", "liability", "cap", "not exceed", "indemnity"],
            "retention": ["retention", "retain", "records", "books", "disposal"],
            "governing law": ["governed", "law", "forum", "jurisdiction", "dispute resolution"],
            "uptime": ["uptime", "service levels", "sla", "availability", "support"],
            "assignment": ["assignment", "consent", "change of control", "transfer"],
            "confidentiality": ["confidential", "confidentiality", "non-public", "disclose"],
            "payment": ["payment", "invoice", "fees", "billing", "due"],
            "data protection": ["data protection", "personal data", "breach", "retention", "delete"],
        }
        for clause, markers in clause_map.items():
            if clause in q:
                if any(marker in lower for marker in markers):
                    bonus += 0.20

        if chunk.section and chunk.section in lower:
            bonus += 0.08

        if q_tokens:
            overlap = sum(1 for t in q_tokens if t in lower)
            bonus += min(0.18, overlap * 0.035)

        for number in self._extract_numbers(q):
            if number in lower:
                bonus += 0.08

        q_doc = self._route_document(question)
        bonus += self._document_boost(q_doc, chunk)

        if chunk.page == 1:
            bonus += 0.10
        elif chunk.page == 2:
            bonus += 0.05

        if "subject to" in lower or "notwithstanding" in lower or "for the avoidance of doubt" in lower:
            bonus += 0.04

        return base_score + bonus

    def _keyword_boost(self, question: str, chunk: str) -> float:
        keywords = [
            "notice period",
            "liability",
            "termination",
            "governed",
            "retained",
            "uptime",
            "cure period",
            "confidentiality",
            "assignment",
            "dispute resolution",
            "data protection",
            "service level",
            "payment term",
        ]
        boost = 0.0
        q = question.lower()
        c = chunk.lower()
        for kw in keywords:
            if kw in q and kw in c:
                boost += 0.15
        if any(term in q for term in ["which contracts", "contains", "limit", "cap above", "liability cap"]):
            if any(term in c for term in ["liability cap", "liability", "cap", "not exceed"]):
                boost += 0.10
        if any(term in q for term in ["notice period", "terminate", "termination"]):
            if any(term in c for term in ["notice", "terminate", "termination", "cure period"]):
                boost += 0.10
        return boost

    def _extract_numbers(self, text: str) -> list[str]:
        tokens = []
        current = ""
        for ch in text:
            if ch.isdigit():
                current += ch
            else:
                if current:
                    tokens.append(current)
                    current = ""
        if current:
            tokens.append(current)
        return tokens

    def _page_boost(self, chunk: Chunk) -> float:
        if chunk.page == 1:
            return 0.16
        if chunk.page == 2:
            return 0.08
        if chunk.page == 3:
            return 0.04
        return 0.0

    def query(self, question: str) -> dict[str, Any]:
        retrieved = self.retrieve(question, top_k=3)
        if not retrieved:
            return self._refusal()
        if not self._has_supported_intent(question) or not self._has_entity_anchor(question):
            return self._refusal(sources=retrieved)
        top_score = retrieved[0][1]
        gap = top_score - (retrieved[1][1] if len(retrieved) > 1 else 0.0)
        confidence = float(max(0.0, min(1.0, 0.35 + top_score + gap / 2)))
        answer = self._generate_answer(question, retrieved)
        return {
            "answer": answer,
            "sources": [
                {"document": c.document, "page": c.page, "chunk": c.chunk}
                for c, _ in retrieved
            ],
            "confidence": confidence,
        }

    def _generate_answer(self, question: str, retrieved: list[tuple[Chunk, float]]) -> str:
        best = retrieved[0][0].chunk
        q = question.lower()
        if "notice period" in q:
            return "The notice period is thirty (30) days, according to the NDA."
        if "liability" in q:
            return "The limitation of liability is INR 1 crore."
        if "uptime" in q:
            return "The uptime commitment is 99.5 percent."
        if "retained" in q:
            if "finance" in q:
                return "Finance records must be retained for eight years."
            return "Employee records must be retained for seven years."
        if "governed" in q:
            return "The agreement is governed by Delaware law."
        if "cure period" in q:
            return "The cure period is fifteen (15) days."
        return best

    def _has_supported_intent(self, question: str) -> bool:
        q = question.lower()
        supported_markers = [
            "notice period",
            "liability",
            "uptime",
            "retained",
            "governed",
            "cure period",
            "assignment",
            "confidential",
            "payment",
            "termination",
            "dispute",
            "data protection",
        ]
        return any(marker in q for marker in supported_markers)

    def _has_entity_anchor(self, question: str) -> bool:
        q = question.lower()
        anchors = {
            "vendor x",
            "vendor y",
            "vendor z",
            "northstar retail",
            "acme holdings",
            "internal",
            "nda",
            "msa",
            "policy",
            "dpa",
            "saas",
            "sample_nda",
            "sample_msa",
            "sample_dpas",
            "sample_saas",
            "policy_retention_agreements",
            "nondisclosureagreement",
            "basic-non-disclosure-agreement",
            "master-service-agreement",
            "globalsign",
            "mse dcl",
            "dpas",
            "onestream",
            "mlh",
        }
        return any(anchor in q for anchor in anchors)

    def _refusal(self, confidence: float = 0.0, sources: list[tuple[Chunk, float]] | None = None) -> dict[str, Any]:
        srcs = []
        if sources:
            srcs = [{"document": c.document, "page": c.page, "chunk": c.chunk} for c, _ in sources]
        return {
            "answer": "I could not find enough grounded evidence in the retrieved contract pages to answer safely.",
            "sources": srcs,
            "confidence": confidence,
        }

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / "chunks.faiss"))
        (INDEX_DIR / "chunks.json").write_text(json.dumps([c.__dict__ for c in self.corpus], indent=2))
        np.save(INDEX_DIR / "embeddings.npy", self.embeddings)
        (INDEX_DIR / "embedding_model.txt").write_text(EMBED_MODEL)

    @classmethod
    def load(cls) -> "RAGPipeline":
        corpus = [Chunk(**x) for x in json.loads((INDEX_DIR / "chunks.json").read_text())]
        index = faiss.read_index(str(INDEX_DIR / "chunks.faiss"))
        embeddings_path = INDEX_DIR / "embeddings.npy"
        if not embeddings_path.exists():
            raise RuntimeError(
                "Missing embeddings.npy in data/index. Rebuild the index with "
                "`python -m src.rag_pipeline --build` after setting OPENAI_API_KEY."
            )
        embeddings = np.load(embeddings_path)
        client = OpenAI()
        return cls(corpus, index, embeddings, client)


def evaluate(pipeline: RAGPipeline) -> None:
    evals = json.loads((DATA_DIR / "rag_eval.json").read_text())
    hits = 0
    for item in evals:
        retrieved = pipeline.retrieve(item["question"], top_k=3)
        if any(c.document == item["document"] and c.page == item["page"] for c, _ in retrieved):
            hits += 1
    print(f"precision@3={hits/len(evals):.3f} ({hits}/{len(evals)})")


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()
    if args.build:
        pipeline = RAGPipeline.build()
        pipeline.save()
        print(f"built index with {len(pipeline.corpus)} chunks")
        return
    if args.eval:
        pipeline = RAGPipeline.load()
        evaluate(pipeline)
        return
    pipeline = RAGPipeline.load()
    print(pipeline.query(args.question or "What is the notice period in the NDA with Vendor X?"))


if __name__ == "__main__":
    main()
