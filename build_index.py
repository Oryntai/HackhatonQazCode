"""
build_index.py — Парсинг клинических протоколов РК и построение FAISS-индекса.

Поддерживаемые источники:
  1. corpus.zip  — архив с JSON-файлами (формат Datasaur/Qazcode challenge).
     Каждый JSON содержит поля: id, text (или content), icd_codes, source.
  2. Директория с PDF (.pdf) и/или Word (.docx) файлами.

Запуск:
  python build_index.py --source corpus.zip
  python build_index.py --source ./protocols_dir
  python build_index.py --source corpus.zip --chunk-size 512 --overlap 64
"""

import argparse
import json
import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# ─── Константы по умолчанию ─────────────────────────────────────────────────
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INDEX_DIR = "./index"
DEFAULT_CHUNK_SIZE = 512   # символов
DEFAULT_OVERLAP = 64       # символов


# ════════════════════════════════════════════════════════════════════════════
# 1. ПАРСЕРЫ ДОКУМЕНТОВ
# ════════════════════════════════════════════════════════════════════════════

def _record_from_dict(data: dict, fallback_name: str) -> dict | None:
    """Нормализует одну запись из JSON/JSONL в единый формат."""
    text = (
        data.get("text")
        or data.get("content")
        or data.get("full_text")
        or data.get("body")
        or ""
    )
    if isinstance(text, list):
        text = "\n".join(text)
    text = text.strip()
    if not text:
        return None

    icd_codes = data.get("icd_codes") or data.get("icd") or []
    if isinstance(icd_codes, str):
        icd_codes = [icd_codes]

    return {
        "id": data.get("id") or data.get("protocol_id", fallback_name),
        "text": text,
        "icd_codes": icd_codes,
        "source": data.get("source") or data.get("source_file", fallback_name),
        "title": data.get("title", data.get("name", "")),
    }


def parse_json_corpus(zip_path: str) -> list[dict]:
    """
    Читает corpus.zip.
    Поддерживает два формата:
      - отдельные .json файлы (по одному протоколу)
      - один .jsonl файл (все протоколы построчно, формат реального corpus.zip)
    """
    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        jsonl_files = [n for n in all_names if n.endswith(".jsonl")]
        json_files  = [n for n in all_names if n.endswith(".json")]

        if jsonl_files:
            # Формат реального corpus.zip: один protocols_corpus.jsonl
            for name in jsonl_files:
                print(f"[corpus.zip] читаем JSONL: {name}")
                with zf.open(name) as f:
                    lines = f.read().decode("utf-8").splitlines()
                print(f"  строк: {len(lines)}")
                for line in tqdm(lines, desc="Парсинг JSONL"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec = _record_from_dict(data, name)
                    if rec:
                        records.append(rec)
        elif json_files:
            # Формат: отдельные JSON-файлы
            print(f"[corpus.zip] найдено {len(json_files)} JSON-файлов")
            for name in tqdm(json_files, desc="Парсинг JSON"):
                with zf.open(name) as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                rec = _record_from_dict(data, name)
                if rec:
                    records.append(rec)
        else:
            print("[corpus.zip] не найдено ни .json ни .jsonl файлов")

    return records


def parse_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF через PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Установите pymupdf: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def parse_docx(docx_path: str) -> str:
    """Извлекает текст из Word (.docx)."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Установите python-docx: pip install python-docx")

    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_directory(dir_path: str) -> list[dict]:
    """
    Сканирует директорию на PDF и DOCX файлы.
    Возвращает список записей в том же формате, что parse_json_corpus.
    """
    records = []
    path = Path(dir_path)
    files = list(path.rglob("*.pdf")) + list(path.rglob("*.docx"))
    print(f"[directory] найдено {len(files)} файлов (PDF + DOCX)")

    for fp in tqdm(files, desc="Парсинг файлов"):
        try:
            if fp.suffix.lower() == ".pdf":
                text = parse_pdf(str(fp))
            else:
                text = parse_docx(str(fp))
        except Exception as e:
            print(f"  [!] Ошибка при чтении {fp.name}: {e}")
            continue

        text = text.strip()
        if not text:
            continue

        # Пытаемся извлечь ICD-коды из текста (формат: J18.0, K29.1, ...)
        icd_codes = re.findall(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", text)
        icd_codes = list(set(icd_codes))

        records.append({
            "id": fp.stem,
            "text": text,
            "icd_codes": icd_codes,
            "source": fp.name,
            "title": fp.stem.replace("_", " ").replace("-", " "),
        })
    return records


# ════════════════════════════════════════════════════════════════════════════
# 2. РАЗБИВКА НА ЧАНКИ
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Базовая очистка: убираем лишние пробелы и пустые строки."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Разбивает текст на перекрывающиеся чанки по символам.
    Старается не резать слова посередине.
    """
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Ищем ближайший пробел или \n, чтобы не резать слово
            cut = text.rfind(" ", start, end)
            if cut == -1 or cut <= start:
                cut = end
            else:
                cut += 1  # включаем пробел в предыдущий чанк
        else:
            cut = len(text)

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        next_start = cut - overlap  # следующий чанк начинается с перекрытием
        if next_start <= start:
            break  # предотвращаем бесконечный цикл в конце текста
        start = next_start

    return chunks


def records_to_chunks(
    records: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[list[str], list[dict]]:
    """
    Для каждой записи разбивает текст на чанки.
    Возвращает:
      - texts: список текстов чанков
      - metas: список метаданных (icd_codes, source, chunk_id, ...)
    """
    texts = []
    metas = []

    for rec in tqdm(records, desc="Разбивка на чанки"):
        chunks = split_into_chunks(rec["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metas.append({
                "doc_id": rec["id"],
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source": rec["source"],
                "title": rec["title"],
                "icd_codes": rec["icd_codes"],
            })

    print(f"  Итого чанков: {len(texts)} из {len(records)} документов")
    return texts, metas


# ════════════════════════════════════════════════════════════════════════════
# 3. ЭМБЕДДИНГИ + FAISS
# ════════════════════════════════════════════════════════════════════════════

def load_embedder(model_name: str = DEFAULT_MODEL):
    """Загружает sentence-transformer модель."""
    from sentence_transformers import SentenceTransformer
    print(f"Загрузка модели: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(
    model,
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Векторизует список текстов. Возвращает float32 numpy array."""
    print(f"Векторизация {len(texts)} чанков (batch={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-нормировка → cosine similarity = dot product
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    """
    Строит FAISS-индекс.
    До 100k векторов — IndexFlatIP (точный поиск по inner product / cosine).
    Больше — IVF с квантованием для скорости.
    """
    import faiss

    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    print(f"Построение FAISS-индекса: {n} векторов, dim={dim}")

    if n <= 100_000:
        # Точный поиск (inner product после L2-нормировки = cosine)
        index = faiss.IndexFlatIP(dim)
    else:
        # IVF + HNSW для больших коллекций
        nlist = min(int(np.sqrt(n)), 4096)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"  Обучение IVF (nlist={nlist})...")
        index.train(embeddings)
        index.nprobe = min(32, nlist)

    index.add(embeddings)
    print(f"  Индекс построен. Всего векторов: {index.ntotal}")
    return index


# ════════════════════════════════════════════════════════════════════════════
# 4. СОХРАНЕНИЕ / ЗАГРУЗКА
# ════════════════════════════════════════════════════════════════════════════

def save_index(
    index,
    metas: list[dict],
    texts: list[str],
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL,
) -> None:
    """Сохраняет FAISS-индекс + метаданные + тексты чанков."""
    import faiss

    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "protocols.faiss"))

    with open(out / "metadata.pkl", "wb") as f:
        pickle.dump({"metas": metas, "texts": texts, "model": model_name}, f)

    # Также сохраняем читаемый JSON с метаданными (без текстов — для отладки)
    with open(out / "metadata_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_chunks": len(texts),
                "model": model_name,
                "sample": metas[:5],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nИндекс сохранён в: {out.resolve()}")
    print(f"  protocols.faiss  — FAISS-индекс")
    print(f"  metadata.pkl     — метаданные + тексты чанков")
    print(f"  metadata_summary.json — краткая сводка")


# ════════════════════════════════════════════════════════════════════════════
# 5. ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Построение FAISS-индекса из клинических протоколов РК"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Путь к corpus.zip или директории с PDF/DOCX файлами",
    )
    parser.add_argument(
        "--index-dir", "-o",
        default=DEFAULT_INDEX_DIR,
        help=f"Директория для сохранения индекса (по умолч.: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers модель (по умолч.: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Размер чанка в символах (по умолч.: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Перекрытие чанков в символах (по умолч.: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер батча для эмбеддингов (по умолч.: 64)",
    )
    args = parser.parse_args()

    source = args.source

    # ── 1. Парсинг ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ШАГ 1: Парсинг документов")
    print(f"{'='*60}")

    if source.endswith(".zip"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Файл не найден: {source}")
        records = parse_json_corpus(source)
    elif os.path.isdir(source):
        records = parse_directory(source)
    else:
        raise ValueError(
            f"--source должен быть .zip-файлом или директорией, получено: {source}"
        )

    if not records:
        print("[!] Документы не найдены. Проверьте путь к источнику.")
        return

    print(f"Загружено документов: {len(records)}")

    # ── 2. Разбивка на чанки ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ШАГ 2: Разбивка на чанки")
    print(f"{'='*60}")
    texts, metas = records_to_chunks(records, args.chunk_size, args.overlap)

    # ── 3. Эмбеддинги ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ШАГ 3: Векторизация (sentence-transformers)")
    print(f"{'='*60}")
    model = load_embedder(args.model)
    embeddings = embed_texts(model, texts, batch_size=args.batch_size)

    # ── 4. FAISS ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ШАГ 4: Построение FAISS-индекса")
    print(f"{'='*60}")
    index = build_faiss_index(embeddings)

    # ── 5. Сохранение ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ШАГ 5: Сохранение")
    print(f"{'='*60}")
    save_index(index, metas, texts, args.index_dir, args.model)

    print("\nГотово!")


if __name__ == "__main__":
    main()
