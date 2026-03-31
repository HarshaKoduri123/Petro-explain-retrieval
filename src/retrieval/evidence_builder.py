from __future__ import annotations

from typing import Any

from src.retrieval.dense_retriever import RetrievalResult


def _safe_get(row: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _format_oxide(row: dict[str, Any], oxide: str) -> str | None:
    value = row.get(oxide)
    if value is None:
        return None
    try:
        if str(value).lower() == "nan":
            return None
        return f"{oxide}={float(value):.3f}"
    except Exception:
        return None


def build_compact_evidence(results: list[RetrievalResult]) -> str:

    lines: list[str] = []

    for result in results:
        row = result.row

        oxides = []
        for oxide in [
            "SiO2",
            "TiO2",
            "Al2O3",
            "Cr2O3",
            "FeO",
            "MnO",
            "MgO",
            "CaO",
            "Na2O",
            "K2O",
        ]:
            formatted = _format_oxide(row, oxide)
            if formatted is not None:
                oxides.append(formatted)

        line = (
            f"[Rank {result.rank} | score={result.score:.4f}] "
            f"record_id={result.record_id}; "
            f"mineral={_safe_get(row, 'mineral')}; "
            f"rock_name={_safe_get(row, 'rock_name')}; "
            f"tectonic_setting={_safe_get(row, 'tectonic_setting')}; "
            f"location={_safe_get(row, 'location')}; "
            f"rock_texture={_safe_get(row, 'rock_texture')}; "
            f"rim_core={_safe_get(row, 'rim_core')}; "
            f"primary_secondary={_safe_get(row, 'primary_secondary')}; "
            f"chemistry=({', '.join(oxides) if oxides else 'no chemistry available'}); "
            f"text_context={_safe_get(row, 'text_context', default='')}"
        )
        lines.append(line)

    return "\n".join(lines)


def build_structured_evidence(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for result in results:
        row = result.row
        evidence.append(
            {
                "rank": result.rank,
                "score": result.score,
                "record_id": result.record_id,
                "metadata": {
                    "mineral": row.get("mineral"),
                    "rock_name": row.get("rock_name"),
                    "tectonic_setting": row.get("tectonic_setting"),
                    "location": row.get("location"),
                    "rock_texture": row.get("rock_texture"),
                    "rim_core": row.get("rim_core"),
                    "primary_secondary": row.get("primary_secondary"),
                },
                "chemistry": {
                    "SiO2": row.get("SiO2"),
                    "TiO2": row.get("TiO2"),
                    "Al2O3": row.get("Al2O3"),
                    "Cr2O3": row.get("Cr2O3"),
                    "FeO": row.get("FeO"),
                    "MnO": row.get("MnO"),
                    "MgO": row.get("MgO"),
                    "CaO": row.get("CaO"),
                    "Na2O": row.get("Na2O"),
                    "K2O": row.get("K2O"),
                },
                "text_context": row.get("text_context", ""),
            }
        )

    return evidence