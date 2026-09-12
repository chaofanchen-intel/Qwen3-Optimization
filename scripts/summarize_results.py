#!/usr/bin/env python3
"""Normalize raw Python/C++ reports and render a comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    ("ttft_ms", "TTFT (ms)"),
    ("tpot_ms", "TPOT (ms/token)"),
    ("first_infer_ms", "First infer (ms)"),
    ("other_infer_ms", "Other infer (ms)"),
    ("total_tps", "Total TPS (token/s)"),
    ("decode_tps", "Decode TPS (token/s)"),
    ("e2e_ms", "E2E (ms)"),
    ("prefill_ms", "Prefill (ms)"),
    ("decode_ms", "Decode (ms)"),
    ("generation_s", "Generation (s)"),
    ("tokenization_ms", "Tokenization (ms)"),
    ("detokenization_ms", "Detokenization (ms)"),
)


def number(value):
    if value is None or value == "" or value == "N/A":
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    clean = [number(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return {"mean": None, "std": None, "p50": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
    }


def make_record(model, backend, device, input_tokens, output_tokens, raw, source, measured_iters):
    record = {
        "model_id": model["id"],
        "family": model["family"],
        "precision": model["precision"],
        "int4_variant": model.get("int4_variant"),
        "backend": backend,
        "device": device,
        "input_tokens": number(input_tokens),
        "output_tokens": number(output_tokens),
        "measured_iters": int(measured_iters),
        "compile_s": number(source.get("compile_time_s", source.get("compile_time"))),
        "metrics": raw,
    }
    for metric, _label in METRICS:
        record[metric] = raw.get(metric, {}).get("mean")
    return record


def python_record(path: Path, model: dict, device: str) -> dict:
    data = json.loads(path.read_text())
    rows = [row for row in data.get("perfdata", {}).get("results", []) if int(row.get("iteration", 0)) > 0]
    if not rows:
        raise ValueError(f"no measured Python rows in {path}")

    def values(key):
        return [row.get(key) for row in rows]

    e2e_values = [
        number(row.get("generation_time")) * 1000.0
        for row in rows
        if number(row.get("generation_time")) is not None
    ]
    decode_values = [
        number(row.get("generation_time")) * 1000.0 - number(row.get("first_latency"))
        for row in rows
        if number(row.get("generation_time")) is not None and number(row.get("first_latency")) is not None
    ]
    raw = {
        "generation_s": stats(values("generation_time")),
        "e2e_ms": stats(e2e_values),
        "ttft_ms": stats(values("first_latency")),
        "tpot_ms": stats(values("second_avg_latency")),
        "first_infer_ms": stats(values("first_infer_latency")),
        "other_infer_ms": stats(values("second_infer_avg_latency")),
        "total_tps": stats(values("total_throughput")),
        "decode_tps": stats(values("second_token_throughput")),
        "tokenization_ms": stats(values("tokenization_time")),
        "detokenization_ms": stats(values("detokenization_time")),
        "prefill_ms": stats(values("first_latency")),
        "decode_ms": stats(decode_values),
    }
    first = rows[0]
    return make_record(model, "python", data.get("metadata", {}).get("device", device), first.get("input_size"), first.get("output_size"), raw, data.get("perfdata", {}), len(rows))


def cpp_record(path: Path, model: dict, device: str) -> dict:
    entries = json.loads(path.read_text())
    if not entries:
        raise ValueError(f"empty C++ report: {path}")
    entry = entries[0]
    raw = {}
    source_names = {
        "generation_s": "generation_time",
        "e2e_ms": "e2e",
        "ttft_ms": "ttft",
        "tpot_ms": "tpot",
        "first_infer_ms": "first_token_infer",
        "other_infer_ms": "other_tokens_infer",
        "total_tps": "throughput",
        "decode_tps": "decode_throughput",
        "prefill_ms": "prefill",
        "decode_ms": "decode",
        "tokenization_ms": "tokenization",
        "detokenization_ms": "detokenization",
    }
    units = {"generation_s": "_s", "total_tps": "_tps", "decode_tps": "_tps"}
    for metric, stem in source_names.items():
        unit = units.get(metric, "_ms")
        raw[metric] = {
            stat: number(entry.get(f"{stem}_{stat}{unit}"))
            for stat in ("mean", "std", "p50", "p90", "p95", "min", "max")
        }
    return make_record(model, "cpp", entry.get("device", device), entry.get("input_tokens"), entry.get("output_tokens_mean"), raw, entry, entry.get("measured_iters", 0))


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, records: list[dict]) -> None:
    fields = ["model_id", "family", "precision", "int4_variant", "backend", "device", "input_tokens", "output_tokens", "measured_iters", "compile_s"]
    fields += [metric for metric, _label in METRICS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def write_markdown(path: Path, records: list[dict], comparisons: list[dict], run_dir: Path) -> None:
    lines = [
        f"# Qwen3 Benchmark Results: `{run_dir.name}`",
        "",
        "Raw reports remain under `raw/`; this file is generated by `scripts/summarize_results.py`.",
        "",
        "## Matched Backend Metrics",
        "",
        "| Model | Precision | Backend | Input | Output | Iters | Compile s | TTFT ms | TPOT ms | Total TPS | Decode TPS | E2E ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['family']} | {record['precision']} {record.get('int4_variant') or ''} | {record['backend']} | "
            f"{fmt(record['input_tokens'], 0)} | {fmt(record['output_tokens'], 0)} | {record['measured_iters']} | "
            f"{fmt(record['compile_s'])} | {fmt(record['ttft_ms'])} | {fmt(record['tpot_ms'])} | "
            f"{fmt(record['total_tps'])} | {fmt(record['decode_tps'])} | {fmt(record['e2e_ms'])} |"
        )
    lines += ["", "## Percentile Detail", "", "The columns below are mean, population standard deviation, p50, p90, p95, min, and max.", ""]
    lines += ["| Model | Backend | Metric | Mean | Std | P50 | P90 | P95 | Min | Max |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        for metric, label in METRICS:
            values = record["metrics"].get(metric, {})
            if values.get("mean") is None:
                continue
            lines.append(
                f"| {record['family']} {record['precision']} {record.get('int4_variant') or ''} | {record['backend']} | {label} | "
                f"{fmt(values.get('mean'))} | {fmt(values.get('std'))} | {fmt(values.get('p50'))} | {fmt(values.get('p90'))} | {fmt(values.get('p95'))} | {fmt(values.get('min'))} | {fmt(values.get('max'))} |"
            )
    lines += ["", "## Python Versus C++", "", "For latency metrics, a positive percentage means Python is slower. For throughput metrics, a positive percentage means Python is faster.", "", "| Model | Metric | C++ mean | Python mean | Python minus C++ |", "|---|---|---:|---:|---:|"]
    for item in comparisons:
        lines.append(f"| {item['model_id']} | {item['metric']} | {fmt(item['cpp'])} | {fmt(item['python'])} | {fmt(item['python_minus_cpp_percent'])}% |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    records: list[dict] = []
    for backend in metadata.get("backends", ["python", "cpp"]):
        for model_id, model in metadata["models"].items():
            report = run_dir / "raw" / backend / model_id / "detailed.json"
            if not report.exists():
                continue
            device = metadata["protocol"]["device"]
            records.append(python_record(report, model, device) if backend == "python" else cpp_record(report, model, device))
    records.sort(key=lambda row: (row["family"], row["precision"], row.get("int4_variant") or "", row["backend"]))
    comparisons = []
    by_key = {(record["model_id"], record["backend"]): record for record in records}
    for model_id in metadata["models"]:
        cpp = by_key.get((model_id, "cpp"))
        python = by_key.get((model_id, "python"))
        if not cpp or not python:
            continue
        for metric, label in METRICS:
            cpp_value = cpp.get(metric)
            python_value = python.get(metric)
            delta = None if cpp_value in (None, 0) or python_value is None else (python_value / cpp_value - 1.0) * 100.0
            comparisons.append({"model_id": model_id, "metric": label, "cpp": cpp_value, "python": python_value, "python_minus_cpp_percent": delta})
    (run_dir / "summary.json").write_text(json.dumps({"records": records, "comparisons": comparisons}, indent=2) + "\n")
    write_csv(run_dir / "summary.csv", records)
    write_markdown(run_dir / "comparison.md", records, comparisons, run_dir)
    print(f"records: {len(records)}")
    print(f"summary: {run_dir / 'summary.csv'}")
    print(f"comparison: {run_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())