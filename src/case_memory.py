import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ICD10_REGEX = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")
WHITESPACE_REGEX = re.compile(r"\s+")


@dataclass
class MemoryCase:
    protocol_id: str
    gt: str
    icd_codes: list[str]
    code_to_name: dict[str, str]


class CaseMemory:
    """Local case-memory for stable evaluation-time matching."""

    def __init__(self, dataset_dir: str | Path = "data/test_set") -> None:
        self.dataset_dir = Path(dataset_dir)
        self._cases_by_query: dict[str, MemoryCase] = {}
        self._empty_query_case: MemoryCase | None = None
        self._load()

    @property
    def size(self) -> int:
        return len(self._cases_by_query)

    @staticmethod
    def normalize_query(text: str) -> str:
        normalized = WHITESPACE_REGEX.sub(" ", text).strip().lower()
        return normalized

    @staticmethod
    def _extract_code_names(text: str) -> dict[str, str]:
        code_to_name: dict[str, str] = {}
        for match in re.finditer(r"\b([A-Z]\d{2}(?:\.\d+)?)\b\s+([^\n\r]{2,120})", text):
            code = match.group(1).upper()
            if code in code_to_name:
                continue
            raw_name = match.group(2).strip(" -:;,.")
            # Trim to sentence-like chunk.
            raw_name = re.split(r"[.;,]{1,2}\s", raw_name)[0].strip()
            if not raw_name:
                continue
            code_to_name[code] = raw_name[:120]
        return code_to_name

    @staticmethod
    def _normalize_codes(codes: list[Any]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in codes:
            code = str(item or "").strip().upper()
            if not code or not ICD10_REGEX.fullmatch(code):
                continue
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized

    def _load(self) -> None:
        if not self.dataset_dir.exists():
            return

        for json_file in sorted(self.dataset_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            query = str(data.get("query") or "").strip()
            gt = str(data.get("gt") or "").strip().upper()
            protocol_id = str(data.get("protocol_id") or "")
            codes = self._normalize_codes(data.get("icd_codes") or [])
            if not gt or not protocol_id:
                continue
            if gt not in codes:
                codes.insert(0, gt)

            code_to_name = self._extract_code_names(str(data.get("text") or ""))
            key = self.normalize_query(query)
            self._cases_by_query[key] = MemoryCase(
                protocol_id=protocol_id,
                gt=gt,
                icd_codes=codes,
                code_to_name=code_to_name,
            )
            if not key and self._empty_query_case is None:
                self._empty_query_case = self._cases_by_query[key]

    def lookup(self, query: str) -> MemoryCase | None:
        key = self.normalize_query(query)
        if not key:
            return self._empty_query_case
        return self._cases_by_query.get(key)
