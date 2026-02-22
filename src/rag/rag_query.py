"""
rag_query.py — Поиск по FAISS-индексу клинических протоколов.

Использование как библиотеки:
    from rag_query import RAGRetriever
    retriever = RAGRetriever("./index")
    results = retriever.search("кашель, температура 38.5, одышка", top_k=5)

Использование из командной строки:
    python rag_query.py "кашель, температура 38.5, одышка" --top-k 5
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional


class RAGRetriever:
    """
    Загружает FAISS-индекс и выполняет семантический поиск
    по клиническим протоколам.
    """

    def __init__(self, index_dir: str = "./index"):
        self.index_dir = Path(index_dir)
        self._load()

    def _load(self):
        import faiss

        index_path = self.index_dir / "protocols.faiss"
        meta_path = self.index_dir / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS-индекс не найден: {index_path}\n"
                "Сначала запустите: python build_index.py --source <corpus.zip или директория>"
            )

        print(f"Загрузка FAISS-индекса из {index_path}...")
        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        self.texts = data["texts"]
        self.metas = data["metas"]
        self.model_name = data.get("model", "paraphrase-multilingual-MiniLM-L12-v2")

        print(f"  Векторов в индексе: {self.index.ntotal}")
        print(f"  Модель эмбеддингов: {self.model_name}")

        # Загружаем модель только при первом поиске (lazy loading)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Загрузка модели: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(
        self,
        query: str,
        top_k: int = 5,
        icd_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Поиск релевантных фрагментов протоколов по запросу (симптомам).

        Args:
            query:      Текст запроса (симптомы пациента, диагноз, ключевые слова).
            top_k:      Количество возвращаемых результатов.
            icd_filter: Фильтрация по ICD-кодам (если задано, возвращает только
                        чанки из протоколов с этими кодами).

        Returns:
            Список словарей:
              - rank:      порядковый номер
              - score:     косинусная схожесть [0..1]
              - text:      текст чанка
              - doc_id:    идентификатор документа
              - source:    имя исходного файла
              - title:     название протокола
              - icd_codes: ICD-10 коды протокола
              - chunk_id:  номер чанка в документе
        """
        import numpy as np

        model = self._get_model()

        # Векторизуем запрос
        q_vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Ищем с запасом, чтобы было что фильтровать
        search_k = top_k * 10 if icd_filter else top_k
        scores, indices = self.index.search(q_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metas[idx]
            text = self.texts[idx]

            # Применяем ICD-фильтр
            if icd_filter:
                doc_icds = set(meta.get("icd_codes", []))
                if not doc_icds.intersection(icd_filter):
                    continue

            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "text": text,
                "doc_id": meta["doc_id"],
                "source": meta["source"],
                "title": meta["title"],
                "icd_codes": meta["icd_codes"],
                "chunk_id": meta["chunk_id"],
            })

            if len(results) >= top_k:
                break

        return results

    def search_for_diagnosis(
        self,
        symptoms: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Специализированный поиск для задачи постановки диагноза.
        Агрегирует чанки по документу и возвращает уникальные протоколы.
        """
        # Берём больше чанков, потом агрегируем
        raw = self.search(symptoms, top_k=top_k * 4)

        # Группируем по doc_id, берём лучший score
        seen: dict[str, dict] = {}
        for r in raw:
            doc_id = r["doc_id"]
            if doc_id not in seen or r["score"] > seen[doc_id]["score"]:
                seen[doc_id] = r

        # Сортируем по score
        aggregated = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

        # Переназначаем ранги
        for i, r in enumerate(aggregated[:top_k]):
            r["rank"] = i + 1

        return aggregated[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Поиск по FAISS-индексу клинических протоколов РК"
    )
    parser.add_argument(
        "query",
        help="Запрос: симптомы пациента или ключевые слова",
    )
    parser.add_argument(
        "--index-dir", "-i",
        default="./index",
        help="Директория с индексом (по умолч.: ./index)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Количество результатов (по умолч.: 5)",
    )
    parser.add_argument(
        "--icd", "-c",
        nargs="*",
        help="Фильтр по ICD-кодам, напр.: J18 K29",
    )
    parser.add_argument(
        "--aggregate", "-a",
        action="store_true",
        default=False,
        help="Агрегировать результаты по документу (для диагностики)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Вывод в формате JSON",
    )
    args = parser.parse_args()

    retriever = RAGRetriever(args.index_dir)

    if args.aggregate:
        results = retriever.search_for_diagnosis(args.query, top_k=args.top_k)
    else:
        results = retriever.search(args.query, top_k=args.top_k, icd_filter=args.icd)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # Красивый текстовый вывод
    print(f"\nЗапрос: «{args.query}»")
    print(f"Найдено результатов: {len(results)}\n")

    for r in results:
        icd_str = ", ".join(r["icd_codes"]) if r["icd_codes"] else "—"
        print(f"{'─'*70}")
        print(f"[{r['rank']}] {r['title'] or r['source']}  (score={r['score']:.4f})")
        print(f"    ICD-10: {icd_str}")
        print(f"    Источник: {r['source']}  (чанк {r['chunk_id']})")
        print(f"    Текст:")
        # Выводим первые 400 символов чанка
        snippet = r["text"][:400].replace("\n", " ")
        print(f"    {snippet}{'...' if len(r['text']) > 400 else ''}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
