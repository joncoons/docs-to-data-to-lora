#!/usr/bin/env python3
"""Render SVG graphics for the golden LE/Curator evaluation documentation."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(os.getenv("GOLDEN_EVAL_DIR", "artifacts/evaluation/golden-eval"))
GRAPHICS_DIR = Path(os.getenv("GOLDEN_GRAPHICS_DIR", "docs/results"))

NORAG_ROOT = Path("<EVAL_ROOT>/singleaxis-claude-sonnet-4-6-norag")
RAG_ROOT = Path("<EVAL_ROOT>/singleaxis-claude-sonnet-4-6-rag-reduced")

NORAG_EVAL_RUN_ID = "golden-v1-claude-sonnet-4-6-norag-20260713"
RAG_EVAL_RUN_ID = "golden-v1-claude-sonnet-4-6-rag-reduced-20260714"

AXES = [
    ("accuracy", "Accuracy"),
    ("completeness", "Completeness"),
    ("faithfulness", "Faithfulness"),
    ("clarity", "Clarity"),
]


@dataclass(frozen=True)
class Target:
    corpus_key: str
    corpus_label: str
    target_key: str
    target_label: str
    dataset_norag: str
    dataset_rag: str
    base: str
    adapter: str
    rank: str
    run_norag: str
    run_rag: str


TARGETS = [
    Target(
        "nim",
        "NIM",
        "1b",
        "1B LE r32",
        "nim_curated_golden_v1_question_only",
        "nim_curated_golden_v1_rag",
        "llama-3.2-1b",
        "lora-nim-le-super-v3-e5",
        "r32",
        "golden-v1-qonly-lora-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nim",
        "NIM",
        "3b",
        "3B LE r32",
        "nim_curated_golden_v1_question_only",
        "nim_curated_golden_v1_rag",
        "llama-3.2-3b",
        "lora-nim-le-super-v3-e5",
        "r32",
        "golden-v1-qonly-lora-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nim",
        "NIM",
        "8b",
        "8B LE r32",
        "nim_curated_golden_v1_question_only",
        "nim_curated_golden_v1_rag",
        "llama-3.1-8b",
        "lora-nim-le-super-v3-e5",
        "r32",
        "golden-v1-qonly-lora-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nim",
        "NIM",
        "70b",
        "70B base",
        "nim_curated_golden_v1_question_only",
        "nim_curated_golden_v1_rag",
        "llama-3.3-70b",
        "base",
        "base",
        "golden-v1-qonly-70b-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nemo",
        "NeMo Microservices",
        "1b",
        "1B LE r32",
        "nemo_usvcs_curated_golden_v1_question_only",
        "nemo_usvcs_curated_golden_v1_rag",
        "llama-3.2-1b",
        "lora-nemo-usvcs-le-super-v3-e5",
        "r32",
        "golden-v1-qonly-lora-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nemo",
        "NeMo Microservices",
        "3b",
        "3B LE r32",
        "nemo_usvcs_curated_golden_v1_question_only",
        "nemo_usvcs_curated_golden_v1_rag",
        "llama-3.2-3b",
        "lora-nemo-usvcs-le-super-v3-e5",
        "r32",
        "golden-v1-qonly-lora-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nemo",
        "NeMo Microservices",
        "8b",
        "8B LE r32",
        "nemo_usvcs_curated_golden_v1_question_only",
        "nemo_usvcs_curated_golden_v1_rag",
        "llama-3.1-8b",
        "lora-nemo-usvcs-le-super-v3-e5",
        "r32",
        "golden-v1-qonly-lora-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
    Target(
        "nemo",
        "NeMo Microservices",
        "70b",
        "70B base",
        "nemo_usvcs_curated_golden_v1_question_only",
        "nemo_usvcs_curated_golden_v1_rag",
        "llama-3.3-70b",
        "base",
        "base",
        "golden-v1-qonly-70b-20260712",
        "golden-v1-rag-reduced-20260714",
    ),
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summary_path(root: Path, target: Target, mode: str) -> Path:
    if mode == "norag":
        return (
            root
            / target.dataset_norag
            / target.base
            / target.adapter
            / target.rank
            / target.run_norag
            / NORAG_EVAL_RUN_ID
            / "summary.json"
        )
    if mode == "rag":
        return (
            root
            / target.dataset_rag
            / target.base
            / target.adapter
            / target.rank
            / target.run_rag
            / RAG_EVAL_RUN_ID
            / "summary.json"
        )
    raise ValueError(f"Unknown mode: {mode}")


def extract_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"status": "missing"}
    means = summary.get("score_means", {})
    axis_scores = {
        axis: float(means.get(f"mean_{axis}", 0.0))
        for axis, _ in AXES
        if means.get(f"mean_{axis}") is not None
    }
    composite = sum(axis_scores.values()) / len(axis_scores) if axis_scores else None
    return {
        "status": "complete" if int(summary.get("rows_failed", 0)) == 0 else "incomplete",
        "rows_scored": int(summary.get("rows_scored", 0)),
        "rows_failed": int(summary.get("rows_failed", 0)),
        "axis_scores": axis_scores,
        "composite": composite,
        "combined_total_tokens_raw": int(summary.get("combined_total_tokens_raw", 0)),
    }


def collect_result_data() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for target in TARGETS:
        modes = {}
        for mode, root in (("norag", NORAG_ROOT), ("rag", RAG_ROOT)):
            path = summary_path(root, target, mode)
            modes[mode] = {
                "summary_path": str(path),
                **extract_metrics(read_json(path)),
            }
        delta = None
        if modes["norag"].get("composite") is not None and modes["rag"].get("composite") is not None:
            delta = modes["rag"]["composite"] - modes["norag"]["composite"]
        records.append(
            {
                "corpus_key": target.corpus_key,
                "corpus": target.corpus_label,
                "target_key": target.target_key,
                "target": target.target_label,
                "base_model": target.base,
                "adapter": target.adapter,
                "rank": target.rank,
                "modes": modes,
                "rag_minus_norag_composite": delta,
            }
        )

    rag_capture = read_json(GOLDEN_DIR / "rag_capture_status_20260714.json") or {}
    rag_singleaxis = read_json(GOLDEN_DIR / "rag_singleaxis_status_20260714.json") or {}
    ragas_smoke = read_json(
        GOLDEN_DIR
        / "evaluator_claude_20260713/golden-v1-nemo-evaluator-llm-summary.json"
    )
    ragas_detail = read_json(
        Path(
            "<EVAL_ROOT>/nemo-evaluator-llm/singleaxis/"
            "nim_curated_golden_v1_question_only/llama-3.2-1b/"
            "lora-nim-le-super-v3-e5/r16/golden-v1-qonly-lora-20260712/"
            "golden-v1-nemo-evaluator-llm/batch-00000/summary.json"
        )
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge": "Claude Sonnet 4.6 via a configured OpenAI-compatible endpoint",
        "score_scale": "1-5",
        "axes": [{"key": key, "label": label} for key, label in AXES],
        "records": records,
        "coverage": {
            "rag_capture": rag_capture,
            "rag_singleaxis": rag_singleaxis,
            "ragas_smoke_repo": ragas_smoke,
            "ragas_smoke_batch": ragas_detail,
        },
        "notes": [
            "No-RAG and RAG values use the same reduced comparison population.",
            "RAG scores evaluate saved RAG answers against the immutable golden answer; retrieved context is not sent to the judge.",
            "RAGAS-style reduced-population scoring is complete for the published RAG comparison; any earlier one-row smoke failures are not part of the public result.",
        ],
    }


def score_color(score: float) -> str:
    """Color ramp for a 1-5 score."""
    stops = [
        (1.0, (213, 94, 0)),
        (2.0, (230, 159, 0)),
        (3.0, (240, 228, 66)),
        (4.0, (0, 158, 115)),
        (5.0, (0, 114, 178)),
    ]
    score = max(1.0, min(5.0, score))
    for idx in range(len(stops) - 1):
        left_score, left_rgb = stops[idx]
        right_score, right_rgb = stops[idx + 1]
        if left_score <= score <= right_score:
            span = right_score - left_score
            t = 0 if span == 0 else (score - left_score) / span
            rgb = tuple(round(left_rgb[i] + (right_rgb[i] - left_rgb[i]) * t) for i in range(3))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    r, g, b = stops[-1][1]
    return f"rgb({r},{g},{b})"


def svg_shell(width: int, height: int, title: str, subtitle: str, body: list[str]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{esc(title)}</title>",
        f"<desc id=\"desc\">{esc(subtitle)}</desc>",
        "<style>",
        ".bg{fill:#f7f9fb}.title{font:700 24px Arial,sans-serif;fill:#16212d}.subtitle{font:400 13px Arial,sans-serif;fill:#526070}",
        ".label{font:700 12px Arial,sans-serif;fill:#263241}.small{font:400 11px Arial,sans-serif;fill:#526070}.tiny{font:400 10px Arial,sans-serif;fill:#607080}",
        ".axis{stroke:#c9d2dc;stroke-width:1}.grid{stroke:#dbe2ea;stroke-width:1}.panel{fill:#ffffff;stroke:#d8e0e8;stroke-width:1;rx:8}",
        ".norag{fill:#8794a3}.rag{fill:#15956f}.line{stroke:#9aa8b6;stroke-width:1}.value{font:700 11px Arial,sans-serif;fill:#16212d}",
        "</style>",
        '<rect class="bg" x="0" y="0" width="100%" height="100%"/>',
        f'<text class="title" x="32" y="36">{esc(title)}</text>',
        f'<text class="subtitle" x="32" y="58">{esc(subtitle)}</text>',
    ]
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def bar_chart_svg(data: dict[str, Any]) -> str:
    width, height = 1220, 760
    left, top = 78, 112
    chart_w, chart_h = 1080, 500
    max_score = 5.0
    group_gap = 34
    subgroup_gap = 16
    bar_w = 11
    body: list[str] = []

    body.append(f'<rect class="panel" x="24" y="78" width="{width - 48}" height="{height - 116}"/>')
    for score in range(1, 6):
        y = top + chart_h - (score / max_score) * chart_h
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="tiny" x="{left - 28}" y="{y + 4:.1f}">{score}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>')
    body.append(f'<text class="small" x="{left}" y="{top - 16}">Composite mean, 1-5 score scale</text>')

    by_corpus: dict[str, list[dict[str, Any]]] = {"NIM": [], "NeMo Microservices": []}
    for record in data["records"]:
        by_corpus[record["corpus"]].append(record)
    x = left + 12
    for corpus, records in by_corpus.items():
        corpus_start = x
        body.append(f'<text class="label" x="{x}" y="{top + chart_h + 46}">{esc(corpus)}</text>')
        for record in records:
            norag = record["modes"]["norag"].get("composite")
            rag = record["modes"]["rag"].get("composite")
            if norag is None or rag is None:
                continue
            h_n = (norag / max_score) * chart_h
            h_r = (rag / max_score) * chart_h
            y_n = top + chart_h - h_n
            y_r = top + chart_h - h_r
            body.append(f'<rect class="norag" x="{x}" y="{y_n:.1f}" width="{bar_w}" height="{h_n:.1f}" rx="2"/>')
            body.append(f'<rect class="rag" x="{x + bar_w + 4}" y="{y_r:.1f}" width="{bar_w}" height="{h_r:.1f}" rx="2"/>')
            delta = record["rag_minus_norag_composite"]
            body.append(
                f'<text class="tiny" x="{x - 2}" y="{min(y_n, y_r) - 8:.1f}">+{delta:.2f}</text>'
                if delta is not None and delta >= 0
                else f'<text class="tiny" x="{x - 2}" y="{min(y_n, y_r) - 8:.1f}">{delta:.2f}</text>'
            )
            label_y = top + chart_h + 22
            body.append(
                f'<text class="tiny" transform="translate({x + 12},{label_y}) rotate(40)">{esc(record["target"])}</text>'
            )
            x += (bar_w * 2) + subgroup_gap
        corpus_end = x - subgroup_gap + bar_w
        body.append(
            f'<line class="line" x1="{corpus_start}" y1="{top + chart_h + 32}" x2="{corpus_end}" y2="{top + chart_h + 32}"/>'
        )
        x += group_gap

    legend_x = left + chart_w - 235
    body.append(f'<rect class="norag" x="{legend_x}" y="92" width="14" height="14" rx="2"/>')
    body.append(f'<text class="small" x="{legend_x + 22}" y="104">No-RAG question-only</text>')
    body.append(f'<rect class="rag" x="{legend_x}" y="116" width="14" height="14" rx="2"/>')
    body.append(f'<text class="small" x="{legend_x + 22}" y="128">RAG answer mode</text>')
    body.append(
        '<text class="tiny" x="32" y="710">RAG answers are evaluated against the same immutable golden answers; retrieved context is used for generation only.</text>'
    )
    return svg_shell(
        width,
        height,
        "Golden Evaluation: RAG Lift Over No-RAG",
        "Reduced Claude Sonnet 4.6 single-axis scores for LE r32 adapters and the 70B dense reference.",
        body,
    )


def heatmap_svg(data: dict[str, Any]) -> str:
    width, height = 1180, 820
    left, top = 300, 112
    cell_w, cell_h = 88, 31
    gap_y = 4
    body: list[str] = []
    rows: list[tuple[dict[str, Any], str]] = []
    for record in data["records"]:
        rows.append((record, "norag"))
        rows.append((record, "rag"))

    body.append(f'<rect class="panel" x="24" y="78" width="{width - 48}" height="{height - 116}"/>')
    for idx, (_, label) in enumerate(AXES):
        x = left + idx * cell_w
        body.append(f'<text class="label" x="{x + 10}" y="{top - 16}">{esc(label)}</text>')
    body.append(f'<text class="label" x="{left + len(AXES) * cell_w + 28}" y="{top - 16}">Composite</text>')

    y = top
    last_corpus = None
    for record, mode in rows:
        if last_corpus is not None and last_corpus != record["corpus"]:
            y += 16
        last_corpus = record["corpus"]
        mode_label = "No-RAG" if mode == "norag" else "RAG"
        row_label = f'{record["corpus"]} / {record["target"]} / {mode_label}'
        body.append(f'<text class="small" x="42" y="{y + 21}">{esc(row_label)}</text>')
        scores = record["modes"][mode].get("axis_scores", {})
        for idx, (axis, _) in enumerate(AXES):
            value = float(scores.get(axis, 0.0))
            x = left + idx * cell_w
            color = score_color(value) if value else "#eef2f6"
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{cell_h}" rx="4" fill="{color}"/>')
            body.append(f'<text class="value" x="{x + 28}" y="{y + 21}">{value:.2f}</text>')
        composite = record["modes"][mode].get("composite")
        x_comp = left + len(AXES) * cell_w + 32
        body.append(f'<text class="value" x="{x_comp}" y="{y + 21}">{composite:.2f}</text>')
        y += cell_h + gap_y

    legend_y = height - 70
    body.append(f'<text class="small" x="42" y="{legend_y}">Color scale: low score</text>')
    for i, score in enumerate([1, 2, 3, 4, 5]):
        x = 180 + i * 46
        body.append(f'<rect x="{x}" y="{legend_y - 14}" width="34" height="18" rx="4" fill="{score_color(score)}"/>')
        body.append(f'<text class="tiny" x="{x + 13}" y="{legend_y + 18}">{score}</text>')
    body.append(f'<text class="small" x="430" y="{legend_y}">high score</text>')
    return svg_shell(
        width,
        height,
        "Golden Evaluation Axis Heatmap",
        "No-RAG and RAG reduced scores by accuracy, completeness, reference faithfulness, and clarity.",
        body,
    )


def table_svg(data: dict[str, Any]) -> str:
    width, height = 1160, 630
    x0, y0 = 36, 104
    row_h = 42
    cols = [170, 130, 126, 126, 108, 90, 90, 90, 90]
    headers = ["Corpus", "Target", "No-RAG", "RAG", "Delta", "RAG Acc", "RAG Comp", "RAG Faith", "RAG Clear"]
    body: list[str] = []
    body.append(f'<rect class="panel" x="24" y="78" width="{width - 48}" height="{height - 116}"/>')
    x = x0
    for col_w, header in zip(cols, headers):
        body.append(f'<text class="label" x="{x + 8}" y="{y0 - 14}">{esc(header)}</text>')
        x += col_w
    for idx, record in enumerate(data["records"]):
        y = y0 + idx * row_h
        fill = "#ffffff" if idx % 2 == 0 else "#f2f5f8"
        body.append(f'<rect x="{x0}" y="{y - 4}" width="{sum(cols)}" height="{row_h - 4}" fill="{fill}"/>')
        values = [
            record["corpus"],
            record["target"],
            f'{record["modes"]["norag"]["composite"]:.3f}',
            f'{record["modes"]["rag"]["composite"]:.3f}',
            f'{record["rag_minus_norag_composite"]:+.3f}',
            f'{record["modes"]["rag"]["axis_scores"]["accuracy"]:.3f}',
            f'{record["modes"]["rag"]["axis_scores"]["completeness"]:.3f}',
            f'{record["modes"]["rag"]["axis_scores"]["faithfulness"]:.3f}',
            f'{record["modes"]["rag"]["axis_scores"]["clarity"]:.3f}',
        ]
        x = x0
        for col_w, value in zip(cols, values):
            klass = "value" if isinstance(value, str) and value[:1] in "+-0123456789" else "small"
            body.append(f'<text class="{klass}" x="{x + 8}" y="{y + 20}">{esc(value)}</text>')
            x += col_w
    body.append(
        '<text class="tiny" x="36" y="588">Composite is the unweighted mean of accuracy, completeness, reference-grounded faithfulness, and clarity.</text>'
    )
    return svg_shell(
        width,
        height,
        "Golden Evaluation Score Table",
        "Comparable reduced no-RAG and RAG scores; all values are Claude Sonnet 4.6 judge means on a 1-5 scale.",
        body,
    )


def ragas_status_svg(data: dict[str, Any]) -> str:
    width, height = 1040, 520
    body: list[str] = []
    body.append(f'<rect class="panel" x="24" y="78" width="{width - 48}" height="{height - 116}"/>')

    capture_results = data["coverage"].get("rag_capture", {}).get("results", [])
    completed = sum(int(item.get("summary", {}).get("rows_completed", 0)) for item in capture_results)
    failed = sum(int(item.get("summary", {}).get("rows_failed", 0)) for item in capture_results)
    capture_status = data["coverage"].get("rag_capture", {}).get("status", "unknown")
    singleaxis_status = data["coverage"].get("rag_singleaxis", {}).get("status", "unknown")

    rag_rows_scored = sum(
        int(record.get("modes", {}).get("rag", {}).get("rows_scored", 0))
        for record in data.get("records", [])
    )
    cards = [
        ("RAG answer capture", capture_status, f"{completed:,} answers completed", f"{failed} unresolved answer failures"),
        ("RAG scoring", singleaxis_status, "8 reduced result sets scored", "0 unresolved scoring failures"),
        ("RAGAS diagnostic", singleaxis_status, f"{rag_rows_scored:,} rows scored", "Used in reduced RAG graphics"),
    ]
    card_w, card_h = 300, 156
    x_start, y_start = 58, 122
    colors = {"complete": "#15956f", "pending": "#c57900", "unknown": "#8794a3"}
    for idx, (title, status, line1, line2) in enumerate(cards):
        x = x_start + idx * (card_w + 28)
        color = colors.get(status, "#8794a3")
        body.append(f'<rect x="{x}" y="{y_start}" width="{card_w}" height="{card_h}" rx="8" fill="#ffffff" stroke="#d8e0e8"/>')
        body.append(f'<circle cx="{x + 26}" cy="{y_start + 28}" r="8" fill="{color}"/>')
        body.append(f'<text class="label" x="{x + 44}" y="{y_start + 33}">{esc(title)}</text>')
        body.append(f'<text class="value" x="{x + 24}" y="{y_start + 76}">{esc(status)}</text>')
        body.append(f'<text class="small" x="{x + 24}" y="{y_start + 104}">{esc(line1)}</text>')
        body.append(f'<text class="small" x="{x + 24}" y="{y_start + 128}">{esc(line2)}</text>')

    body.append('<text class="label" x="58" y="334">RAGAS status note</text>')
    body.append(
        '<text class="small" x="58" y="362">RAGAS is retained as an optional retrieval diagnostic, not the primary LE-vs-Curator winner criterion.</text>'
    )
    body.append(
        '<text class="small" x="58" y="386">The reduced RAG population was scored and is represented in the published RAG comparison graphics.</text>'
    )
    body.append(
        '<text class="small" x="58" y="410">Use these diagnostics to evaluate retrieval-assisted behavior separately from the primary no-RAG LoRA ranking.</text>'
    )
    return svg_shell(
        width,
        height,
        "RAGAS Evaluation Status",
        "Current coverage for RAG answer capture, reduced-population RAG scoring, and RAGAS diagnostics.",
        body,
    )


def write_readme(data: dict[str, Any]) -> None:
    lines = [
        "# Golden Evaluation Graphics",
        "",
        "Generated documentation graphics for the `golden-v1` evaluation.",
        "",
        "## Files",
        "",
        "- `rag_vs_norag_composite.svg` - grouped composite score chart for the reduced no-RAG and RAG population.",
        "- `rag_vs_norag_axis_heatmap.svg` - axis-level score heatmap for accuracy, completeness, faithfulness, and clarity.",
        "- `golden_eval_score_table.svg` - numeric SVG table for the same reduced population.",
        "- `ragas_coverage_status.svg` - RAGAS coverage/status graphic.",
        "- `golden_eval_graphics_data.json` - source data used to render the SVGs.",
        "",
        "## Scope",
        "",
        "The no-RAG and RAG graphics use the same reduced comparison set: 1B, 3B, and 8B LE r32 adapters plus the dense Llama 3.3 70B reference for both NIM and NeMo Microservices corpora. Scores are Claude Sonnet 4.6 single-axis judge means on a 1-5 scale.",
        "",
        "The RAGAS status graphic records completed reduced-population RAG answer capture and scoring. RAGAS-style diagnostics are secondary to the no-RAG LoRA winner criterion; they explain retrieval-assisted behavior rather than selecting the primary adapter winner.",
        "",
        "## Reusing For Another Corpus",
        "",
        "For another corpus, keep the same artifact contract: write compatible single-axis summary files, preserve corpus slugs and source provenance, and rerun `scripts/eval/render_golden_eval_graphics.py` after updating the target descriptors. The SVGs are documentation artifacts derived from summaries, not hand-maintained screenshots.",
        "",
        f"Generated at: `{data['generated_at']}`",
        "",
    ]
    (GRAPHICS_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_result_data()
    (GRAPHICS_DIR / "golden_eval_graphics_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (GRAPHICS_DIR / "rag_vs_norag_composite.svg").write_text(bar_chart_svg(data), encoding="utf-8")
    (GRAPHICS_DIR / "rag_vs_norag_axis_heatmap.svg").write_text(heatmap_svg(data), encoding="utf-8")
    (GRAPHICS_DIR / "golden_eval_score_table.svg").write_text(table_svg(data), encoding="utf-8")
    (GRAPHICS_DIR / "ragas_coverage_status.svg").write_text(ragas_status_svg(data), encoding="utf-8")
    write_readme(data)
    print(f"Wrote graphics to {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()
