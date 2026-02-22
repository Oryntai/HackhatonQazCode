from pathlib import Path
from typing import Any


class MedicalRetriever:
    def __init__(
        self,
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "clinical_protocols",
        embedding_model: str = "intfloat/multilingual-e5-base",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vectorstore = None

        self._connect()

    def _connect(self) -> None:
        # If index directory is missing, keep retriever disabled.
        if not self.persist_dir.exists():
            self.vectorstore = None
            return

        try:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            self.vectorstore = None
            return

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_dir),
            )
        except Exception:
            self.vectorstore = None

    def _flatten_query(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, dict):
            parts: list[str] = []
            for v in value.values():
                parts.extend(self._flatten_query(v))
            return parts
        if isinstance(value, (list, tuple, set)):
            parts = []
            for v in value:
                parts.extend(self._flatten_query(v))
            return parts
        text = str(value).strip()
        return [text] if text else []

    def get_relevant_context(self, query: Any, k: int = 5) -> str:
        query_text = f"query: {query}"
        query_parts = self._flatten_query(query)
        raw_query = " ".join(query_parts).strip()
        if not raw_query:
            return ""
        query_text = f"query: {raw_query}"

        if not self.persist_dir.exists():
            return ""

        if self.vectorstore is None:
            self._connect()
            if self.vectorstore is None:
                return ""

        try:
            search_k = max(1, int(k))
            docs = self.vectorstore.similarity_search(query_text, k=search_k)
        except Exception:
            return ""

        chunks: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue

            metadata: dict[str, Any] = doc.metadata or {}
            source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
            page = metadata.get("page")
            header = f"[chunk {idx}] source={Path(source).name}"
            if page is not None:
                header = f"{header}, page={page}"

            chunks.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(chunks)
