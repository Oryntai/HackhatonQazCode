import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from tqdm import tqdm

if TYPE_CHECKING:
    from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".json", ".jsonl"}


def discover_files(corpus_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                yield obj
            else:
                raise ValueError(
                    f"{path.name}:{line_num} contains non-object JSON entry"
                )


def _iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for idx, obj in enumerate(data, start=1):
            if isinstance(obj, dict):
                yield obj
            else:
                raise ValueError(f"{path.name}[{idx}] is not a JSON object")
        return

    if isinstance(data, dict):
        yield data
        return

    raise ValueError(f"{path.name} must contain a JSON object or array of objects")


def iter_protocol_objects(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_objects(path)
        return

    if suffix == ".json":
        try:
            yield from _iter_json_objects(path)
        except json.JSONDecodeError:
            # Some corpora are .json but actually JSONL formatted.
            yield from _iter_jsonl_objects(path)
        return

    return


def protocol_to_document(obj: dict[str, Any], path: Path) -> "Document | None":
    from langchain_core.documents import Document

    text = obj.get("text")
    if text is None:
        return None

    content = str(text).strip()
    if not content:
        return None

    metadata = {
        "source": str(obj.get("source_file") or path.name),
        "id": str(obj.get("protocol_id") or ""),
        "title": str(obj.get("title") or ""),
        "corpus_file": str(path),
        "file_type": path.suffix.lower().lstrip("."),
    }
    return Document(page_content=content, metadata=metadata)


def load_file(path: Path) -> list["Document"]:
    docs: list["Document"] = []
    for obj in iter_protocol_objects(path):
        doc = protocol_to_document(obj, path)
        if doc is not None:
            docs.append(doc)
    return docs


def load_documents(paths: Sequence[Path]) -> tuple[list["Document"], list[tuple[Path, str]]]:
    documents: list["Document"] = []
    failed: list[tuple[Path, str]] = []

    for path in tqdm(paths, desc="Loading files", unit="file"):
        try:
            documents.extend(load_file(path))
        except Exception as exc:
            failed.append((path, str(exc)))

    return documents, failed


def split_documents(
    documents: Sequence["Document"],
    chunk_size: int = 2500,
    chunk_overlap: int = 400,
) -> list["Document"]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    for idx, chunk in enumerate(chunks):
        if not (chunk.page_content or "").startswith("passage: "):
            chunk.page_content = f"passage: {chunk.page_content}"
        chunk.metadata["chunk_id"] = idx
    return chunks


def create_embeddings() -> Any:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain-huggingface'. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_chroma_index(
    chunks: Sequence["Document"],
    persist_dir: Path,
    collection_name: str,
    batch_size: int = 128,
) -> None:
    from langchain_community.vectorstores import Chroma

    embeddings = create_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Indexing chunks",
        unit="batch",
    ):
        batch = list(chunks[start : start + batch_size])
        vectorstore.add_documents(batch)

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()


def run_ingest(
    corpus_dir: Path,
    persist_dir: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    reset: bool,
) -> int:
    print("RAG ingestion started")
    print(f"Corpus directory : {corpus_dir}")
    print(f"Chroma directory : {persist_dir}")
    print(f"Collection       : {collection_name}")

    if not corpus_dir.exists() or not corpus_dir.is_dir():
        print(f"ERROR: corpus directory does not exist: {corpus_dir}")
        return 1

    files = discover_files(corpus_dir)
    if not files:
        print(
            "ERROR: no JSON/JSONL files found in corpus directory "
            f"({corpus_dir}). Expected files in data/corpus/."
        )
        return 1

    print(f"Discovered files : {len(files)}")

    if persist_dir.exists() and reset:
        print(f"Reset enabled. Removing existing index at {persist_dir}")
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    documents, failed = load_documents(files)
    if failed:
        print(f"WARNING: failed to load {len(failed)} file(s)")
        for path, error in failed[:10]:
            print(f" - {path}: {error}")
        if len(failed) > 10:
            print(f" - ... and {len(failed) - 10} more")

    if not documents:
        print("ERROR: no protocol documents were loaded successfully")
        return 1

    print(f"Loaded protocol docs: {len(documents)}")
    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(
        f"Chunk settings   : chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )
    print(f"Total chunks     : {len(chunks)}")

    if not chunks:
        print("ERROR: splitter produced no chunks")
        return 1

    build_chroma_index(
        chunks=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        batch_size=batch_size,
    )

    print("Ingestion complete")
    print(f"Chroma DB saved to: {persist_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB index from JSON/JSONL clinical protocol corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus"),
        help="Directory with protocol files (.json/.jsonl). Default: data/corpus",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Directory to store Chroma DB. Default: chroma_db",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="clinical_protocols",
        help="Chroma collection name. Default: clinical_protocols",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2500,
        help="Chunk size for RecursiveCharacterTextSplitter. Default: 2500",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Chunk overlap for RecursiveCharacterTextSplitter. Default: 400",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for writing chunks to Chroma. Default: 128",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Chroma directory before indexing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_ingest(
        corpus_dir=args.corpus_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reset=args.reset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
