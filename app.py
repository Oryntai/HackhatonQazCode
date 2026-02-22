import json
import logging
import math
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from rag_query import RAGRetriever
except ImportError:
    from src.rag.rag_query import RAGRetriever


INDEX_DIR = os.getenv("INDEX_DIR", "./index")
RAG_CHUNK_K = int(os.getenv("RAG_CHUNK_K", "96"))
RAG_DOC_K = int(os.getenv("RAG_DOC_K", "14"))
RAG_DOC_CONTEXT_CHARS = int(os.getenv("RAG_DOC_CONTEXT_CHARS", "12000"))
RESULT_TOP_K = 3

ICD10_REGEX = re.compile(r"^[A-Z]\d{2}(?:\.\d+)?$")
CODE_WITH_NAME_REGEX = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b\s+([^\n\r]{3,180})")
TOKEN_REGEX = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{2,}")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("medical-assistant")


class DiagnoseRequest(BaseModel):
    symptoms: Any = Field(default="", description="Free-text patient symptoms")


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


@dataclass
class DocumentCandidate:
    doc_id: str
    title: str
    source: str
    icd_codes: list[str]
    score: float
    best_text: str


def _normalize_icd(code: Any) -> str:
    text = str(code or "").strip().upper()
    return text if ICD10_REGEX.match(text) else ""


def _normalize_symptoms(raw_symptoms: Any) -> str:
    if raw_symptoms is None:
        return ""
    if isinstance(raw_symptoms, str):
        text = raw_symptoms.strip()
    else:
        text = json.dumps(raw_symptoms, ensure_ascii=False).strip()
    return _repair_mojibake_ru(text)


def _repair_mojibake_ru(text: str) -> str:
    """Repairs common UTF-8/CP1251 mojibake (e.g. 'РџСЂРёРІРµС‚' -> 'Привет')."""
    raw = (text or "").strip()
    if not raw:
        return ""
    # Heuristic: try recovery only for typical broken Cyrillic traces.
    if "Р" not in raw and "С" not in raw:
        return raw
    try:
        repaired = raw.encode("latin1").decode("utf-8")
    except Exception:
        return raw
    # Keep repaired text only if it looks more Cyrillic than the source.
    raw_cyr = sum("А" <= ch <= "я" or ch in "Ёё" for ch in raw)
    repaired_cyr = sum("А" <= ch <= "я" or ch in "Ёё" for ch in repaired)
    return repaired if repaired_cyr > raw_cyr else raw


def _snippet(text: str, limit: int = 320) -> str:
    value = (text or "").strip().replace("\n", " ")
    if not value:
        return "Недостаточно данных"
    return f"{value[:limit]}..."


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_REGEX.findall(text or "")]


def _lexical_overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(text))
    if not doc_tokens:
        return 0.0
    inter = len(query_tokens.intersection(doc_tokens))
    denom = math.sqrt(len(query_tokens) * len(doc_tokens))
    if denom == 0:
        return 0.0
    return inter / denom


class HybridRAGEngine:
    """Dense+lexical+code rerank engine for ICD-10 retrieval."""

    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.doc_to_chunk_indices: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, meta in enumerate(self.retriever.metas):
            doc_id = str(meta.get("doc_id") or "")
            chunk_id = int(meta.get("chunk_id") or 0)
            self.doc_to_chunk_indices[doc_id].append((chunk_id, idx))
        for doc_id in self.doc_to_chunk_indices:
            self.doc_to_chunk_indices[doc_id].sort(key=lambda x: x[0])

        self.doc_code_name_cache: dict[str, dict[str, str]] = {}
        self.doc_code_vectors_cache: dict[str, tuple[list[str], np.ndarray, list[str]]] = {}

    def _protocol_text(self, doc_id: str, max_chars: int = RAG_DOC_CONTEXT_CHARS) -> str:
        pairs = self.doc_to_chunk_indices.get(doc_id) or []
        if not pairs:
            return ""
        parts: list[str] = []
        total = 0
        for _, chunk_idx in pairs:
            chunk = str(self.retriever.texts[chunk_idx] or "").strip()
            if not chunk:
                continue
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
        return "\n".join(parts)

    def _extract_code_names(self, doc_id: str) -> dict[str, str]:
        cached = self.doc_code_name_cache.get(doc_id)
        if cached is not None:
            return cached

        text = self._protocol_text(doc_id)
        mapping: dict[str, str] = {}
        for match in CODE_WITH_NAME_REGEX.finditer(text):
            code = _normalize_icd(match.group(1))
            if not code or code in mapping:
                continue
            name = match.group(2).strip(" -:;,.")
            if not name:
                continue
            name = re.split(r"[.;]{1,2}\s", name)[0].strip()
            mapping[code] = name[:140]

        self.doc_code_name_cache[doc_id] = mapping
        return mapping

    def _get_doc_code_vectors(
        self,
        doc_id: str,
        codes: list[str],
        title: str,
        evidence: str,
    ) -> tuple[list[str], np.ndarray, list[str]]:
        cached = self.doc_code_vectors_cache.get(doc_id)
        if cached is not None:
            return cached

        code_names = self._extract_code_names(doc_id)
        unique_codes = [_normalize_icd(code) for code in codes]
        unique_codes = [code for code in unique_codes if code]
        unique_codes = list(dict.fromkeys(unique_codes))

        descriptor_texts: list[str] = []
        kept_codes: list[str] = []
        for code in unique_codes:
            code_name = code_names.get(code, "")
            descriptor = f"Код {code}. {code_name}. Протокол: {title}."
            if evidence:
                descriptor += f" Контекст: {_snippet(evidence, 220)}"
            descriptor_texts.append(descriptor)
            kept_codes.append(code)

        if not descriptor_texts:
            empty = ([], np.zeros((0, 384), dtype=np.float32), [])
            self.doc_code_vectors_cache[doc_id] = empty
            return empty

        model = self.retriever._get_model()
        vectors = model.encode(
            descriptor_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        result = (kept_codes, vectors, descriptor_texts)
        self.doc_code_vectors_cache[doc_id] = result
        return result

    def _aggregate_documents(
        self,
        rag_chunks: list[dict[str, Any]],
        query_tokens: set[str],
        top_k: int,
    ) -> list[DocumentCandidate]:
        grouped: dict[str, dict[str, Any]] = {}

        for chunk in rag_chunks:
            doc_id = str(chunk.get("doc_id") or "")
            if not doc_id:
                continue

            title = str(chunk.get("title") or "").strip()
            source = str(chunk.get("source") or "").strip()
            text = str(chunk.get("text") or "")
            score = float(chunk.get("score") or 0.0)

            raw_codes = chunk.get("icd_codes") or []
            if isinstance(raw_codes, str):
                raw_codes = [raw_codes]
            codes = [_normalize_icd(code) for code in raw_codes]
            codes = [code for code in codes if code]

            lex = _lexical_overlap(query_tokens, f"{title} {text[:800]}")
            blended = 0.78 * score + 0.22 * lex

            prev = grouped.get(doc_id)
            if prev is None:
                grouped[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "source": source,
                    "codes": codes,
                    "score": blended,
                    "best_text": text,
                    "best_score": score,
                }
                continue

            if score > prev["best_score"]:
                prev["best_score"] = score
                prev["best_text"] = text
                prev["title"] = title or prev["title"]
                prev["source"] = source or prev["source"]

            prev["score"] = max(prev["score"], blended)
            if codes:
                merged = list(dict.fromkeys(prev["codes"] + codes))
                prev["codes"] = merged

        ranked = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
        output: list[DocumentCandidate] = []
        for row in ranked[:top_k]:
            output.append(
                DocumentCandidate(
                    doc_id=row["doc_id"],
                    title=row["title"] or "Без названия",
                    source=row["source"] or "unknown",
                    icd_codes=row["codes"],
                    score=float(row["score"]),
                    best_text=str(row["best_text"]),
                )
            )
        return output

    def predict(self, symptoms: str, top_k: int = RESULT_TOP_K) -> list[dict[str, str]]:
        rag_chunks = self.retriever.search(symptoms, top_k=RAG_CHUNK_K)
        if not rag_chunks:
            return []

        query_tokens = set(_tokenize(symptoms))
        doc_candidates = self._aggregate_documents(rag_chunks, query_tokens, RAG_DOC_K)
        if not doc_candidates:
            return []

        model = self.retriever._get_model()
        query_vec = model.encode(
            [symptoms],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]

        score_by_code: dict[str, float] = defaultdict(float)
        support_by_code: dict[str, dict[str, str]] = {}

        for doc_rank, doc in enumerate(doc_candidates, start=1):
            doc_rank_penalty = 1.0 / (1.0 + 0.06 * (doc_rank - 1))
            dense_component = doc.score * doc_rank_penalty

            # Base scoring from document relevance.
            for code in doc.icd_codes:
                score_by_code[code] += 0.45 * dense_component
                if code not in support_by_code:
                    support_by_code[code] = {
                        "diagnosis": "",
                        "explanation": _snippet(doc.best_text, 320),
                        "source": doc.source,
                    }

            # Fine-grained code rerank using code descriptors from the protocol.
            codes, vectors, _ = self._get_doc_code_vectors(
                doc_id=doc.doc_id,
                codes=doc.icd_codes,
                title=doc.title,
                evidence=doc.best_text,
            )
            if vectors.size == 0:
                continue

            sims = (vectors @ query_vec).tolist()
            code_names = self._extract_code_names(doc.doc_id)
            for code, sim in zip(codes, sims):
                semantic_component = max(0.0, float(sim))
                score_by_code[code] += 0.55 * semantic_component + 0.20 * dense_component

                if code not in support_by_code:
                    support_by_code[code] = {
                        "diagnosis": code_names.get(code, ""),
                        "explanation": _snippet(doc.best_text, 320),
                        "source": doc.source,
                    }
                elif not support_by_code[code]["diagnosis"]:
                    support_by_code[code]["diagnosis"] = code_names.get(code, "")

        if not score_by_code:
            return []

        ranked_codes = sorted(score_by_code.items(), key=lambda x: x[1], reverse=True)

        diagnoses: list[dict[str, str]] = []
        for rank, (code, _) in enumerate(ranked_codes[:top_k], start=1):
            support = support_by_code.get(code, {})
            diagnosis_name = (support.get("diagnosis") or "").strip()
            if not diagnosis_name:
                diagnosis_name = "Диагноз по клиническому протоколу"

            diagnoses.append(
                {
                    "rank": rank,
                    "diagnosis": diagnosis_name,
                    "icd10_code": code,
                    "explanation": support.get("explanation") or "Нет обоснования",
                }
            )

        while len(diagnoses) < top_k:
            diagnoses.append(
                {
                    "rank": len(diagnoses) + 1,
                    "diagnosis": "Неуточненный диагноз",
                    "icd10_code": "R69",
                    "explanation": "Недостаточно данных",
                }
            )

        for idx, item in enumerate(diagnoses, start=1):
            item["rank"] = idx

        return diagnoses


def _hard_fallback(reason: str) -> DiagnoseResponse:
    return DiagnoseResponse(
        diagnoses=[
            Diagnosis(rank=1, diagnosis="Неуточненный диагноз", icd10_code="R69", explanation=reason),
            Diagnosis(rank=2, diagnosis="Неуточненный диагноз", icd10_code="R69", explanation=reason),
            Diagnosis(rank=3, diagnosis="Неуточненный диагноз", icd10_code="R69", explanation=reason),
        ]
    )


app = FastAPI(title="QazCode Medical Diagnosis Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RETRIEVER: RAGRetriever | None = None
ENGINE: HybridRAGEngine | None = None

try:
    RETRIEVER = RAGRetriever(INDEX_DIR)
    ENGINE = HybridRAGEngine(RETRIEVER)
except Exception as exc:
    LOGGER.exception("Retriever initialization failed: %s", exc)

if ENGINE is None:
    raise RuntimeError(
        "RAG engine is unavailable. Install faiss-cpu and ensure INDEX_DIR points to a built index."
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "backend": "hybrid-rag-v2",
        "retriever": "faiss+dense-lexical-code-rerank",
        "index_dir": INDEX_DIR,
        "retriever_available": str(ENGINE is not None).lower(),
        "rag_chunk_k": str(RAG_CHUNK_K),
        "rag_doc_k": str(RAG_DOC_K),
    }


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(payload: DiagnoseRequest) -> DiagnoseResponse:
    request_id = uuid.uuid4().hex[:8]
    symptoms = _normalize_symptoms(payload.symptoms)

    if not symptoms:
        return _hard_fallback("Пустое описание симптомов")

    start = time.perf_counter()
    try:
        assert ENGINE is not None
        predicted = ENGINE.predict(symptoms=symptoms, top_k=RESULT_TOP_K)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not predicted:
            LOGGER.warning("request_id=%s output_source=hard_fallback reason=empty_prediction", request_id)
            return _hard_fallback("Недостаточно данных")

        diagnoses = [Diagnosis(**item) for item in predicted]
        LOGGER.info(
            "request_id=%s output_source=hybrid_rag latency_ms=%.2f top_code=%s",
            request_id,
            elapsed_ms,
            diagnoses[0].icd10_code if diagnoses else "R69",
        )
        return DiagnoseResponse(diagnoses=diagnoses)
    except Exception as exc:
        LOGGER.exception("request_id=%s diagnose failed: %s", request_id, exc)
        return _hard_fallback(f"Ошибка RAG: {exc.__class__.__name__}")


try:
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse("dist/index.html")
except Exception:
    LOGGER.info("Frontend static files are not available (dist is missing).")
