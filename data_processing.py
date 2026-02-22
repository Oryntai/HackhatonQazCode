import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ICD-10: uppercase Latin letter + 2 digits + optional ".digits"
ICD10_PATTERN = re.compile(r"(?<![A-Z0-9])[A-Z]\d{2}(?:\.\d+)?(?![A-Z0-9])")


def extract_icd_from_text(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Parse ICD-10 codes directly from the `text` column and store them in `parsed_icd`.

    The function intentionally ignores any existing `icd_codes` column.
    """
    if "text" not in df.columns:
        raise KeyError("DataFrame must contain a 'text' column")

    result = df.copy()
    result["parsed_icd"] = (
        result["text"]
        .fillna("")
        .astype(str)
        .map(lambda value: list(dict.fromkeys(ICD10_PATTERN.findall(value))))
    )
    return result
