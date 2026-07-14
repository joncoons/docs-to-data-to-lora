#!/usr/bin/env python3
"""Compute Super-vs-Ultra semantic quotient with Llama 3.2 embeddings."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_EMBED_URL = "http://10.43.101.173:8000/v1/embeddings"
DEFAULT_EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_INPUT_TYPE = "passage"
CORPORA = ("nim_curated", "nemo_usvcs_curated")
APPROACHES = ("curator", "le")
MODEL_DIRS = {"super": "super-v3", "ultra": "ultra-550b"}
LABELS = {
    "nim_curated": "NIM",
    "nemo_usvcs_curated": "NeMo Microservices",
    "curator": "Curator",
    "le": "LE",
    "super": "Super",
    "ultra": "Ultra",
}


@dataclass(frozen=True)
class RowRecord:
    approach: str
    corpus: str
    side: str
    row_id: str
    source_key: str
    exact_key: str
    text: str
    text_sha256: str
    source_path: str
    source_line_number: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def stable_id(prefix: str, material: str) -> str:
    return f"{prefix}_{sha256_text(material)[:24]}"


def iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            rows.append((line_number, json.loads(line)))
    return rows


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(EXPERIMENT_ROOT.resolve()))


def row_text(question: str, answer: str) -> str:
    return f"Question: {clean(question)}\nAnswer: {clean(answer)}"


def load_curator_rows(corpus: str, side: str) -> list[RowRecord]:
    path = EXPERIMENT_ROOT / "runs" / "curator_micro" / corpus / MODEL_DIRS[side] / "normalized" / "sft.jsonl"
    records: list[RowRecord] = []
    for line_number, row in iter_jsonl(path):
        lineage = row.get("lineage") or {}
        document_id = clean(lineage.get("document_id"))
        segment_id = clean(lineage.get("segment_id"))
        pair_index = str(lineage.get("pair_index") if lineage.get("pair_index") is not None else "")
        text = row_text(row.get("prompt"), row.get("completion"))
        row_id = clean(row.get("sample_id")) or stable_id("curator", f"{corpus}\n{side}\n{line_number}\n{text}")
        records.append(
            RowRecord(
                approach="curator",
                corpus=corpus,
                side=side,
                row_id=row_id,
                source_key=document_id,
                exact_key="\t".join([document_id, segment_id, pair_index]),
                text=text,
                text_sha256=sha256_text(text),
                source_path=rel(path),
                source_line_number=line_number,
            )
        )
    return records


def load_le_rows(corpus: str, side: str) -> list[RowRecord]:
    path = EXPERIMENT_ROOT / "runs" / "le_micro" / corpus / MODEL_DIRS[side] / "stage1a_le.jsonl"
    records: list[RowRecord] = []
    for line_number, row in iter_jsonl(path):
        passage_id = clean(row.get("passage_id"))
        entailment_index = str(row.get("entailment_index") if row.get("entailment_index") is not None else "")
        premise_index = str(row.get("premise_index") if row.get("premise_index") is not None else "")
        text = row_text(row.get("question"), row.get("answer"))
        row_id = clean(row.get("sample_id")) or stable_id("le", f"{corpus}\n{side}\n{line_number}\n{passage_id}\n{text}")
        records.append(
            RowRecord(
                approach="le",
                corpus=corpus,
                side=side,
                row_id=row_id,
                source_key=passage_id,
                exact_key="\t".join([passage_id, entailment_index, premise_index]),
                text=text,
                text_sha256=sha256_text(text),
                source_path=rel(path),
                source_line_number=line_number,
            )
        )
    return records


def load_all_rows() -> dict[tuple[str, str, str], list[RowRecord]]:
    out: dict[tuple[str, str, str], list[RowRecord]] = {}
    for corpus in CORPORA:
        for side in ("super", "ultra"):
            out[("curator", corpus, side)] = load_curator_rows(corpus, side)
            out[("le", corpus, side)] = load_le_rows(corpus, side)
    return out


def post_embeddings(url: str, model: str, input_type: str, texts: list[str], timeout_s: float) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts, "input_type": input_type}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    data = sorted(body["data"], key=lambda item: int(item.get("index", 0)))
    return [item["embedding"] for item in data]


def embed_texts(
    texts: list[str],
    *,
    url: str,
    model: str,
    input_type: str,
    batch_size: int,
    timeout_s: float,
    max_attempts: int,
    retry_sleep_s: float,
) -> dict[str, np.ndarray]:
    unique = sorted(set(texts), key=sha256_text)
    vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                raw_vectors = post_embeddings(url, model, input_type, batch, timeout_s)
                if len(raw_vectors) != len(batch):
                    raise RuntimeError(f"expected {len(batch)} embeddings, got {len(raw_vectors)}")
                for text, vector in zip(batch, raw_vectors):
                    arr = np.asarray(vector, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm == 0:
                        raise RuntimeError("embedding vector had zero norm")
                    vectors[text] = arr / norm
                print(json.dumps({"embedded": min(start + len(batch), len(unique)), "total": len(unique)}), flush=True)
                break
            except Exception as exc:  # noqa: BLE001 - endpoint retries with captured final failure.
                last_error = exc
                if attempt < max_attempts - 1:
                    time.sleep(retry_sleep_s * (attempt + 1))
                else:
                    raise RuntimeError(f"embedding batch {start}:{start + len(batch)} failed: {exc}") from exc
        if last_error is not None and any(text not in vectors for text in batch):
            raise last_error
    return vectors


def matrix_for(rows: list[RowRecord], vectors: dict[str, np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([vectors[row.text] for row in rows])


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None, "std": None}
    sorted_values = sorted(float(value) for value in values)
    def pct(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = (len(sorted_values) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_values[lo]
        frac = idx - lo
        return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
    return {
        "count": len(sorted_values),
        "mean": round(statistics.mean(sorted_values), 6),
        "median": round(statistics.median(sorted_values), 6),
        "p10": round(pct(0.10), 6),
        "p25": round(pct(0.25), 6),
        "p75": round(pct(0.75), 6),
        "p90": round(pct(0.90), 6),
        "min": round(min(sorted_values), 6),
        "max": round(max(sorted_values), 6),
        "std": round(statistics.pstdev(sorted_values), 6) if len(sorted_values) > 1 else 0.0,
    }


def grouped_by(rows: list[RowRecord], attr: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = getattr(row, attr)
        groups.setdefault(key, []).append(index)
    return groups


def exact_pair_similarities(super_rows: list[RowRecord], ultra_rows: list[RowRecord], s_mat: np.ndarray, u_mat: np.ndarray) -> tuple[list[float], list[dict[str, Any]]]:
    super_by_key = {row.exact_key: idx for idx, row in enumerate(super_rows)}
    ultra_by_key = {row.exact_key: idx for idx, row in enumerate(ultra_rows)}
    records: list[dict[str, Any]] = []
    values: list[float] = []
    for key in sorted(set(super_by_key) & set(ultra_by_key)):
        s_idx = super_by_key[key]
        u_idx = ultra_by_key[key]
        cosine = float(np.dot(s_mat[s_idx], u_mat[u_idx]))
        values.append(cosine)
        records.append(
            {
                "match_type": "exact_lineage",
                "exact_key_sha256": sha256_text(key),
                "source_key": super_rows[s_idx].source_key,
                "super_row_id": super_rows[s_idx].row_id,
                "ultra_row_id": ultra_rows[u_idx].row_id,
                "super_source_line_number": super_rows[s_idx].source_line_number,
                "ultra_source_line_number": ultra_rows[u_idx].source_line_number,
                "cosine": round(cosine, 6),
            }
        )
    return values, records


def nearest_neighbor_records(
    source_rows: list[RowRecord],
    target_rows: list[RowRecord],
    source_mat: np.ndarray,
    target_mat: np.ndarray,
    *,
    direction: str,
    source_constrained: bool,
) -> tuple[list[float], list[dict[str, Any]]]:
    target_groups = grouped_by(target_rows, "source_key")
    values: list[float] = []
    records: list[dict[str, Any]] = []
    for s_idx, source_row in enumerate(source_rows):
        candidate_indices = target_groups.get(source_row.source_key, []) if source_constrained else list(range(len(target_rows)))
        if not candidate_indices:
            continue
        candidates = target_mat[candidate_indices]
        sims = candidates @ source_mat[s_idx]
        local_best = int(np.argmax(sims))
        target_idx = candidate_indices[local_best]
        cosine = float(sims[local_best])
        values.append(cosine)
        records.append(
            {
                "match_type": "source_nearest_neighbor" if source_constrained else "global_nearest_neighbor",
                "direction": direction,
                "source_constrained": source_constrained,
                "source_key": source_row.source_key,
                "source_row_id": source_row.row_id,
                "target_row_id": target_rows[target_idx].row_id,
                "source_line_number": source_row.source_line_number,
                "target_line_number": target_rows[target_idx].source_line_number,
                "cosine": round(cosine, 6),
            }
        )
    return values, records


def centroid_cosine(super_mat: np.ndarray, ultra_mat: np.ndarray) -> float | None:
    if super_mat.size == 0 or ultra_mat.size == 0:
        return None
    s = super_mat.mean(axis=0)
    u = ultra_mat.mean(axis=0)
    s_norm = np.linalg.norm(s)
    u_norm = np.linalg.norm(u)
    if s_norm == 0 or u_norm == 0:
        return None
    return round(float(np.dot(s / s_norm, u / u_norm)), 6)


def compute_summary(all_rows: dict[tuple[str, str, str], list[RowRecord]], vectors: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    for approach in APPROACHES:
        for corpus in CORPORA:
            super_rows = all_rows[(approach, corpus, "super")]
            ultra_rows = all_rows[(approach, corpus, "ultra")]
            s_mat = matrix_for(super_rows, vectors)
            u_mat = matrix_for(ultra_rows, vectors)
            exact_values, exact_records = exact_pair_similarities(super_rows, ultra_rows, s_mat, u_mat)
            s2u_source, s2u_source_records = nearest_neighbor_records(super_rows, ultra_rows, s_mat, u_mat, direction="super_to_ultra", source_constrained=True)
            u2s_source, u2s_source_records = nearest_neighbor_records(ultra_rows, super_rows, u_mat, s_mat, direction="ultra_to_super", source_constrained=True)
            s2u_global, _ = nearest_neighbor_records(super_rows, ultra_rows, s_mat, u_mat, direction="super_to_ultra", source_constrained=False)
            u2s_global, _ = nearest_neighbor_records(ultra_rows, super_rows, u_mat, s_mat, direction="ultra_to_super", source_constrained=False)
            source_values = s2u_source + u2s_source
            global_values = s2u_global + u2s_global
            quotient = statistics.mean(source_values) if source_values else None
            cell = {
                "approach": approach,
                "corpus": corpus,
                "rows": {"super": len(super_rows), "ultra": len(ultra_rows)},
                "embedding_text": "Question + answer only; source context excluded to avoid context-dominated similarity.",
                "exact_lineage_pair_count": len(exact_values),
                "exact_lineage_cosine": stats(exact_values),
                "source_nearest_neighbor_cosine": {
                    "super_to_ultra": stats(s2u_source),
                    "ultra_to_super": stats(u2s_source),
                    "symmetric": stats(source_values),
                },
                "global_nearest_neighbor_cosine": {
                    "super_to_ultra": stats(s2u_global),
                    "ultra_to_super": stats(u2s_global),
                    "symmetric": stats(global_values),
                },
                "semantic_quotient": round(float(quotient), 6) if quotient is not None else None,
                "semantic_dissimilarity": round(float(1 - quotient), 6) if quotient is not None else None,
                "centroid_cosine": centroid_cosine(s_mat, u_mat),
            }
            cells.append(cell)
            for record in exact_records + s2u_source_records + u2s_source_records:
                record.update({"approach": approach, "corpus": corpus})
                match_records.append(record)
    summary = {"cells": cells}
    return summary, match_records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Super vs Ultra Semantic Quotient",
        "",
        f"Created: {summary['created_at']}",
        "",
        f"Embedding model: `{summary['embedding']['model']}` via `{summary['embedding']['url']}` with `input_type={summary['embedding']['input_type']}`.",
        "",
        "Primary quotient: symmetric same-source nearest-neighbor cosine between Super and Ultra QA rows. `1 - quotient` is the semantic dissimilarity score. Text embedded is generated question plus answer only; source context is excluded.",
        "",
        "| Approach | Corpus | Super rows | Ultra rows | Exact pairs | Exact-pair mean | Source-NN quotient | Dissimilarity | Centroid cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in summary["cells"]:
        exact_mean = cell["exact_lineage_cosine"]["mean"]
        lines.append(
            f"| {LABELS[cell['approach']]} | {LABELS[cell['corpus']]} | {cell['rows']['super']} | {cell['rows']['ultra']} | "
            f"{cell['exact_lineage_pair_count']} | {exact_mean} | {cell['semantic_quotient']} | {cell['semantic_dissimilarity']} | {cell['centroid_cosine']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for cell in summary["cells"]:
        quotient = cell["semantic_quotient"]
        dissimilarity = cell["semantic_dissimilarity"]
        lines.append(
            f"- {LABELS[cell['approach']]} / {LABELS[cell['corpus']]}: quotient {quotient:.3f}, dissimilarity {dissimilarity:.3f}; "
            f"exact-pair mean {cell['exact_lineage_cosine']['mean']}."
        )
    lines.append("")
    lines.append("Lower quotient means Ultra is adding more semantically different QA coverage relative to Super; higher quotient means outputs are closer paraphrases or repeated coverage of the same concepts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(path: Path, summary: dict[str, Any]) -> None:
    width, height = 1000, 520
    colors = {"curator": "#2f6f73", "le": "#b55a30"}
    cells = summary["cells"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Super versus Ultra semantic quotient">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append('<text x="50" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">Super vs Ultra semantic quotient</text>')
    svg.append('<text x="50" y="68" font-family="Arial, sans-serif" font-size="13" fill="#52616b">Symmetric same-source nearest-neighbor cosine; lower dissimilarity means more similar outputs</text>')
    chart_x, chart_y, chart_w, chart_h = 90, 115, 820, 270
    svg.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#a7b0b8"/>')
    svg.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#a7b0b8"/>')
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y + chart_h - tick * chart_h
        svg.append(f'<line x1="{chart_x - 5}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" stroke="#e4e7eb"/>')
        svg.append(f'<text x="{chart_x - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#52616b">{tick:.2f}</text>')
    group_w = chart_w / 4
    bar_w = 42
    for i, cell in enumerate(cells):
        x_mid = chart_x + group_w * i + group_w / 2
        q = cell["semantic_quotient"] or 0
        h = q * chart_h
        x = x_mid - bar_w / 2
        y = chart_y + chart_h - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="2" fill="{colors[cell["approach"]]}"/>')
        svg.append(f'<text x="{x_mid:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#1f2933">{q:.3f}</text>')
        label = f"{LABELS[cell['approach']]}\n{LABELS[cell['corpus']].replace(' Microservices', '')}"
        parts = label.split("\n")
        svg.append(f'<text x="{x_mid:.1f}" y="{chart_y + chart_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{html.escape(parts[0])}</text>')
        svg.append(f'<text x="{x_mid:.1f}" y="{chart_y + chart_h + 41}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{html.escape(parts[1])}</text>')
    for idx, approach in enumerate(["curator", "le"]):
        lx = 90 + idx * 130
        svg.append(f'<rect x="{lx}" y="465" width="16" height="16" fill="{colors[approach]}" rx="2"/>')
        svg.append(f'<text x="{lx + 24}" y="478" font-family="Arial, sans-serif" font-size="13" fill="#1f2933">{LABELS[approach]}</text>')
    svg.append('<text x="90" y="505" font-family="Arial, sans-serif" font-size="12" fill="#52616b">Primary score uses generated QA text only, with source-constrained nearest-neighbor matching.</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--input-type", default=DEFAULT_INPUT_TYPE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-sleep-s", type=float, default=2.0)
    parser.add_argument("--output-json", type=Path, default=EXPERIMENT_ROOT / "semantic_quotient.json")
    parser.add_argument("--output-jsonl", type=Path, default=EXPERIMENT_ROOT / "semantic_quotient_matches.jsonl")
    parser.add_argument("--output-md", type=Path, default=EXPERIMENT_ROOT / "semantic_quotient.md")
    parser.add_argument("--output-svg", type=Path, default=EXPERIMENT_ROOT / "semantic_quotient.svg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows = load_all_rows()
    all_texts = [row.text for rows in all_rows.values() for row in rows]
    vectors = embed_texts(
        all_texts,
        url=args.embed_url,
        model=args.embed_model,
        input_type=args.input_type,
        batch_size=args.batch_size,
        timeout_s=args.timeout_s,
        max_attempts=args.max_attempts,
        retry_sleep_s=args.retry_sleep_s,
    )
    partial, match_records = compute_summary(all_rows, vectors)
    vector_dims = sorted({int(v.shape[0]) for v in vectors.values()})
    summary = {
        "schema_version": "ultra_550b_smoke.semantic_quotient.v1",
        "created_at": utc_now(),
        "embedding": {
            "url": args.embed_url,
            "model": args.embed_model,
            "input_type": args.input_type,
            "dimensions": vector_dims[0] if len(vector_dims) == 1 else vector_dims,
            "unique_texts_embedded": len(vectors),
            "text_definition": "Question + answer only; source context excluded.",
        },
        "metric_definition": {
            "semantic_quotient": "Mean of Super->Ultra and Ultra->Super nearest-neighbor cosine, constrained to rows from the same source document/passage.",
            "semantic_dissimilarity": "1 - semantic_quotient",
            "exact_lineage_cosine": "Cosine for rows with identical lineage/index keys; LE exact-index pairs are weaker evidence because entailment order can differ by generator.",
        },
        **partial,
    }
    write_json(args.output_json, summary)
    write_jsonl(args.output_jsonl, match_records)
    write_report(args.output_md, summary)
    write_svg(args.output_svg, summary)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "rows": len(match_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
