from __future__ import annotations

from typing import Iterable


def format_range(min_val: float, max_val: float, decimals: int = 2) -> str:
    return f"{min_val:.{decimals}f}–{max_val:.{decimals}f}"


def oxides_sentence(oxide_summaries: Iterable[str]) -> str:
    oxide_summaries = [x for x in oxide_summaries if x]
    if not oxide_summaries:
        return "No stable oxide trend could be summarized from the retrieved neighbors."
    return "Across the retrieved neighbors, the chemistry is characterized by " + ", ".join(oxide_summaries) + "."


def categorical_sentence(
    dominant_mineral: str,
    host_rocks_text: str,
    texture_text: str,
    rim_core_text: str,
    primary_secondary_text: str,
) -> str:
    sentence_1_parts = []

    if dominant_mineral != "unknown":
        sentence_1_parts.append(
            f"The retrieved nearest neighbors consistently identify the sample as {dominant_mineral.lower()}"
        )
    else:
        sentence_1_parts.append(
            "The retrieved nearest neighbors do not provide a stable mineral identity"
        )

    if host_rocks_text:
        sentence_1_parts.append(f"and place it in {host_rocks_text}")

    sentence_1 = " ".join(sentence_1_parts).strip()
    if not sentence_1.endswith("."):
        sentence_1 += "."

    sentence_2_parts = []

    if texture_text:
        sentence_2_parts.append(f"The retrieved matches show {texture_text} textures")
    if rim_core_text:
        sentence_2_parts.append(f"and are predominantly {rim_core_text}")
    if primary_secondary_text:
        sentence_2_parts.append(f"and mostly {primary_secondary_text}")

    if sentence_2_parts:
        sentence_2 = " ".join(sentence_2_parts).strip()
        if sentence_2.startswith("and "):
            sentence_2 = sentence_2[4:]
        if not sentence_2.endswith("."):
            sentence_2 += "."
        return sentence_1 + " " + sentence_2

    return sentence_1


def confidence_sentence(
    top_score: float,
    mean_score: float,
    agreement_note: str,
) -> str:
    return (
        f"Retrieval confidence is supported by a top similarity score of {top_score:.3f} "
        f"and an average top-k score of {mean_score:.3f}. {agreement_note}"
    )


def build_final_summary(
    paragraph_1: str,
    paragraph_2: str,
    paragraph_3: str,
) -> str:
    return "\n\n".join([p.strip() for p in [paragraph_1, paragraph_2, paragraph_3] if p and p.strip()])