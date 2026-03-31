from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.summarization.templates import (
    build_final_summary,
    categorical_sentence,
    confidence_sentence,
    format_range,
    oxides_sentence,
)


OXIDE_COLUMNS = [
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
]


@dataclass
class SummaryArtifacts:
    summary: str
    stats: dict[str, Any]


def _clean_text(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "unknown"
    return text


def _counter_from_results(results: list[dict[str, Any]], field: str) -> Counter:
    values = []
    for r in results:
        metadata = r.get("metadata", {})
        values.append(_clean_text(metadata.get(field)))
    return Counter(values)


def _top_non_unknown(counter: Counter, top_n: int = 3) -> list[tuple[str, int]]:
    items = [(k, v) for k, v in counter.items() if k != "unknown"]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:top_n]


def _format_counter_list(counter: Counter, top_n: int = 3) -> str:
    items = _top_non_unknown(counter, top_n=top_n)
    if not items:
        return ""
    names = [name.lower() for name, _ in items]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _oxide_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}

    for oxide in OXIDE_COLUMNS:
        vals = []
        for r in results:
            chemistry = r.get("chemistry", {})
            value = chemistry.get(oxide)
            try:
                if value is not None and str(value).lower() != "nan":
                    vals.append(float(value))
            except Exception:
                continue

        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            stats[oxide] = {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "std": float(arr.std()),
            }

    return stats


def _build_oxide_summary_strings(stats: dict[str, dict[str, float]]) -> list[str]:
    selected = []

    important_order = ["SiO2", "MgO", "CaO", "FeO", "Al2O3", "TiO2", "Na2O", "K2O"]
    for oxide in important_order:
        if oxide not in stats:
            continue
        s = stats[oxide]
        selected.append(
            f"{oxide} around {s['mean']:.2f} wt% "
            f"(range {format_range(s['min'], s['max'])} wt%)"
        )

    return selected[:6]


def _agreement_note(
    mineral_counter: Counter,
    rock_counter: Counter,
    rim_core_counter: Counter,
    primary_counter: Counter,
    n_results: int,
) -> str:
    notes = []

    if mineral_counter:
        mineral, mineral_count = mineral_counter.most_common(1)[0]
        if mineral != "unknown":
            notes.append(f"{mineral_count}/{n_results} top matches share the mineral label {mineral.lower()}")

    host_rocks = _top_non_unknown(rock_counter, top_n=3)
    if host_rocks:
        rock_names = [name.lower() for name, _ in host_rocks]
        if len(rock_names) == 1:
            notes.append(f"the host-rock context is dominated by {rock_names[0]}")
        else:
            notes.append(f"the host-rock context clusters around {' / '.join(rock_names)}")

    rim_core = _top_non_unknown(rim_core_counter, top_n=1)
    if rim_core:
        notes.append(f"most matches correspond to {rim_core[0][0].lower()} analyses")

    primary = _top_non_unknown(primary_counter, top_n=1)
    if primary:
        notes.append(f"most matches are {primary[0][0].lower()} grains")

    if not notes:
        return "The top retrieved samples show moderate internal consistency."

    sentence = "The top retrieved samples show strong internal consistency: " + "; ".join(notes) + "."
    return sentence


def summarize_retrieval_results(results: list[dict[str, Any]]) -> SummaryArtifacts:
    if not results:
        return SummaryArtifacts(
            summary="No retrieved evidence was available, so no grounded explanation could be generated.",
            stats={},
        )

    n_results = len(results)

    scores = [float(r.get("score", 0.0)) for r in results]
    top_score = max(scores)
    mean_score = float(np.mean(scores))

    mineral_counter = _counter_from_results(results, "mineral")
    rock_counter = _counter_from_results(results, "rock_name")
    texture_counter = _counter_from_results(results, "rock_texture")
    rim_core_counter = _counter_from_results(results, "rim_core")
    primary_counter = _counter_from_results(results, "primary_secondary")
    tectonic_counter = _counter_from_results(results, "tectonic_setting")

    dominant_mineral = mineral_counter.most_common(1)[0][0] if mineral_counter else "unknown"
    host_rocks_text = _format_counter_list(rock_counter, top_n=4)
    texture_text = _format_counter_list(texture_counter, top_n=2)
    rim_core_text = _format_counter_list(rim_core_counter, top_n=1)
    primary_secondary_text = _format_counter_list(primary_counter, top_n=1)
    tectonic_text = _format_counter_list(tectonic_counter, top_n=3)

    stats = _oxide_stats(results)
    oxide_summary_strings = _build_oxide_summary_strings(stats)

    para1 = categorical_sentence(
        dominant_mineral=dominant_mineral,
        host_rocks_text=f"{host_rocks_text} host-rock settings" if host_rocks_text else "",
        texture_text=f"{texture_text} textures" if texture_text and texture_text != "unknown" else "",
        rim_core_text=rim_core_text,
        primary_secondary_text=primary_secondary_text,
    )

    if tectonic_text:
        para1 += f" Tectonic-setting information is limited, but retrieved labels include {tectonic_text.lower()}."
    else:
        para1 += " Tectonic-setting evidence is limited in the current retrieved neighborhood."

    para2 = oxides_sentence(oxide_summary_strings)

    agreement = _agreement_note(
        mineral_counter=mineral_counter,
        rock_counter=rock_counter,
        rim_core_counter=rim_core_counter,
        primary_counter=primary_counter,
        n_results=n_results,
    )
    para3 = confidence_sentence(
        top_score=top_score,
        mean_score=mean_score,
        agreement_note=agreement,
    )

    summary = build_final_summary(para1, para2, para3)

    return SummaryArtifacts(
        summary=summary,
        stats={
            "top_score": top_score,
            "mean_score": mean_score,
            "n_results": n_results,
            "dominant_mineral": dominant_mineral,
            "rock_counter": dict(rock_counter),
            "tectonic_counter": dict(tectonic_counter),
            "oxide_stats": stats,
        },
    )