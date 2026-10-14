from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import click

BASELINE_ENV_VAR = "RETRIEVAL_BASELINE"
BASELINE_TRACE_STAGE = "baseline"
# Stage names that only exist on the hybrid path. Their absence in a
# baseline trace is expected; their presence means the baseline guard
# in app/retrieval/hybrid.py did not take effect.
HYBRID_ONLY_STAGES = ("graph_search", "rrf_fusion", "rerank")


def load_dataset(paths: list[str], limit: int | None) -> list[dict]:
    cases: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["dataset"] = path.name
                cases.append(record)
    if limit:
        cases = cases[:limit]
    return cases


def stage_ms_map(trace: list) -> dict[str, float]:
    stages: dict[str, float] = {}
    for entry in trace or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("stage")
        ms = entry.get("ms")
        if name and isinstance(ms, (int, float)):
            stages[str(name)] = float(ms)
    return stages


def collect_rows(cases: list[dict]) -> list[dict]:
    # Deferred on purpose: the caller sets RETRIEVAL_BASELINE before this
    # import so the baseline guard in the retrieval layer takes effect.
    import time

    from app.crag.graph import run_crag

    def run_with_retry(question: str, tries: int = 5):
        delay = 5.0
        for attempt in range(tries):
            try:
                return run_crag(question)
            except Exception as exc:
                # Groq free-tier 429s: back off and retry instead of
                # aborting the whole benchmark run.
                if "429" in str(exc) and attempt < tries - 1:
                    print(f"  rate-limited, retrying in {delay:.0f}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    rows: list[dict] = []
    for index, case in enumerate(cases):
        result = run_with_retry(case["question"])
        contexts = []
        for context in result.contexts:
            if isinstance(context, dict):
                contexts.append(
                    f"{context.get('title', '')}\n{context.get('snippet', '') or context.get('text', '')}"
                )
            else:
                contexts.append(f"{context.title}\n{context.text}")
        rows.append(
            {
                "user_input": case["question"],
                "retrieved_contexts": contexts,
                "response": result.answer,
                "reference": case.get("reference", ""),
                "used_web": result.used_web,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "stage_ms": stage_ms_map(result.trace),
            }
        )
        print(f"[{index + 1}/{len(cases)}] {case['question'][:70]}...")
    return rows


def median_stage_ms(rows: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for stage, ms in row.get("stage_ms", {}).items():
            buckets.setdefault(stage, []).append(ms)
    return {
        stage: round(float(statistics.median(samples)), 1)
        for stage, samples in sorted(buckets.items())
    }


def validate_baseline(rows: list[dict]) -> str:
    """Check the trace of every row for the baseline marker.

    Returns "PASS" only if every case ran the dense-only path; otherwise
    a FAIL message naming how many rows missed the marker (which means
    the RETRIEVAL_BASELINE guard in app/retrieval/hybrid.py is absent
    or was bypassed).
    """
    missing = sum(
        1
        for row in rows
        if BASELINE_TRACE_STAGE not in row.get("stage_ms", {})
        or any(stage in row.get("stage_ms", {}) for stage in HYBRID_ONLY_STAGES)
    )
    if missing:
        return (
            f"FAIL: {missing}/{len(rows)} rows lack the '{BASELINE_TRACE_STAGE}' "
            "trace marker (baseline guard not active in app/retrieval/hybrid.py)"
        )
    return "PASS"


def default_report_path(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    return Path(f"evaluation/results/full_eval_{mode}_{stamp}.json")


@click.command()
@click.option("--dataset", "datasets", multiple=True, default=("evaluation/golden_multi_hop.jsonl",))
@click.option("--limit", default=None, type=int)
@click.option(
    "--out",
    default=None,
    help="Explicit report path (default: evaluation/results/full_eval_<mode>_<yyyymmdd_hhmm>.json)",
)
@click.option(
    "--baseline",
    is_flag=True,
    default=False,
    help="Vector-only baseline: dense ANN retrieval, single retrieve+generate pass",
)
@click.option("--deepeval", is_flag=True, default=False, help="Also run DeepEval G-Eval correctness")
def main(
    datasets: tuple[str, ...],
    limit: int | None,
    out: str | None,
    baseline: bool,
    deepeval: bool,
) -> None:
    mode = "vector-only" if baseline else "hybrid"
    if baseline:
        # Must happen before ANY app.* import so the baseline guard in
        # app/retrieval/hybrid.py and app/crag/nodes.py takes effect.
        os.environ[BASELINE_ENV_VAR] = "1"

    cases = load_dataset(list(datasets), limit)
    if not cases:
        raise SystemExit("No evaluation cases loaded.")
    out_path = Path(out) if out else default_report_path(mode)
    print(f"Running {mode} pipeline over {len(cases)} cases...")
    rows = collect_rows(cases)

    timestamp = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "mode": mode,
        "timestamp": timestamp,
        "cases": len(rows),
        "datasets": list(datasets),
        "median_stage_ms": median_stage_ms(rows),
        "rows": rows,
    }
    if baseline:
        report["baseline_validation"] = validate_baseline(rows)

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        from app.core.llm import chat_llm
        from evaluation.adapters import BGEEmbeddingsAdapter

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": row["user_input"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "response": row["response"],
                    "reference": row["reference"],
                }
                for row in rows
            ]
        )
        print("Computing Ragas metrics (faithfulness, answer relevancy, context precision/recall)...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=chat_llm(),
            embeddings=BGEEmbeddingsAdapter(),
            run_config=RunConfig(max_workers=2, max_retries=2),
        )
        frame = result.to_pandas()

        metrics_summary: dict[str, float] = {}
        for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            if metric_name in frame.columns:
                value = frame[metric_name].dropna()
                if len(value):
                    metrics_summary[metric_name] = round(float(value.mean()), 4)
        report["ragas"] = metrics_summary

        web_used = sum(1 for row in rows if row["used_web"])
        report["web_fallback_rate"] = round(web_used / len(rows), 3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path.with_suffix(".csv"), index=False)
    except ImportError as exc:
        report["ragas_error"] = f"Install eval extras: pip install -e '.[eval]' ({exc})"

    if deepeval:
        try:
            from evaluation.deepeval_runner import run_deepeval

            report["deepeval"] = run_deepeval(cases, rows)
        except Exception as exc:
            report["deepeval_error"] = str(exc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    summary = {
        "mode": report["mode"],
        "timestamp": report["timestamp"],
        "cases": report["cases"],
        "baseline_validation": report.get("baseline_validation"),
        "ragas": report.get("ragas"),
        "web_fallback_rate": report.get("web_fallback_rate"),
        "median_stage_ms": report["median_stage_ms"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Full report saved to {out_path}")


if __name__ == "__main__":
    main()
