"""Human-readable rendering of one bounded research-diagnostic report."""

from __future__ import annotations

from datetime import UTC

from torchinstruments.reporting.records import (
    CategoryFindingsRecord,
    FindingRecord,
    MergedReportRecord,
    ReportRecord,
)


def render_run_index(report: ReportRecord) -> str:
    """Render a compact diagnosis and evidence-constrained LLM prompt."""
    run = report.run
    coverage = report.coverage
    created_at = run.created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    lines = [
        "# TorchInstruments research report",
        "",
        "> Human entry point. `report.json` contains the same ranked findings as typed data.",
        "> No exhaustive per-layer JSON document is written by default.",
        "",
        "## Run overview",
        "",
        f"- TorchInstruments version: `{run.observer_version}`",
        f"- PyTorch version: `{run.torch_version}`",
        f"- Created at: `{created_at}`",
        f"- Rank: `{report.rank.rank}` of `{report.rank.world_size}` processes",
        f"- Sampling policy: `{run.sampling.type}`",
        f"- Invocation capture: `{run.collection.invocation_capture}`",
        f"- Selected modules: `{coverage.selected_modules}`",
        f"- Tensor paths summarized locally: `{coverage.tensor_paths}`",
        f"- Temporal series ranked locally: `{coverage.temporal_series}`",
        f"- Sampled forwards observed: `{coverage.samples_observed}`",
        f"- Correlated backwards observed: `{coverage.backward_samples_observed}`",
        f"- Findings returned: `{coverage.findings_returned}`",
        f"- Findings omitted: `{coverage.findings_omitted}`",
        f"- Report byte-budget truncation: `{coverage.report_truncated_by_byte_budget}`",
        "",
        "## Ranked findings",
        "",
    ]
    for category in report.findings:
        lines.extend(_render_category(category, report))
    lines.extend(
        [
            "## Use this report with an LLM",
            "",
            "Give the LLM only `index.md` and `report.json`, then ask:",
            "",
            "```text",
            "Analyze the ranked TorchInstruments findings in report.json. For every material",
            "finding, cite the exact category, module, call index, signal, tensor path, metric,",
            "values, and evidence fields. Separate measured interpretation from plausible",
            "mechanisms. State missing evidence and propose the smallest controlled experiment.",
            "Treat warmup_complete=false as weak temporal evidence. Do not infer losses, labels,",
            "optimizer updates, inputs, or parameter gradients that were not observed.",
            "```",
            "",
            "## Evidence limits",
            "",
            "TorchInstruments ranks selected-module outputs and correlated output gradients. A",
            "finding is a measured signature, not a causal diagnosis. `findings_omitted` and the",
            "drop counters in `report.json` describe evidence excluded by ranking or hard limits.",
            "",
        ]
    )
    return "\n".join(lines)


def render_merged_index(report: MergedReportRecord) -> str:
    """Render rank coverage and top global findings from one merged report."""
    coverage = report.coverage
    lines = [
        "# TorchInstruments merged DDP report",
        "",
        "> `global-report.json` is the bounded typed source for this human-readable view.",
        "",
        "## Rank coverage",
        "",
        f"- Expected ranks: `{coverage.expected_ranks}`",
        f"- Ranks present: `{list(coverage.ranks_present)}`",
        f"- Complete rank coverage: `{coverage.rank_coverage_complete}`",
        f"- Source reports truncated: `{coverage.source_reports_truncated}`",
        f"- Global findings returned: `{coverage.findings_returned}`",
        f"- Findings omitted after ranking: `{coverage.findings_omitted_after_global_ranking}`",
        f"- Byte-budget truncation: `{coverage.report_truncated_by_byte_budget}`",
        "",
        "## Ranked findings across ranks",
        "",
    ]
    for category in report.findings:
        lines.append(f"### {category.category.value.replace('_', ' ').title()}")
        lines.append("")
        for finding in category.findings[
            : report.report_configuration.markdown_findings_per_category
        ]:
            module = finding.module or "@root"
            lines.append(
                f"- Rank `{finding.source_rank}`, #{finding.rank}: `{module}` "
                f"`{finding.signal}.{finding.tensor_path}.{finding.metric}`; "
                f"score `{finding.ranking_score:.6g}`."
            )
        if not category.findings:
            lines.append("No retained finding.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Use `global-report.json` as the LLM input. If rank coverage is incomplete or any",
            "source report was truncated, state that limitation before comparing ranks.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_category(category: CategoryFindingsRecord, report: ReportRecord) -> list[str]:
    """Render a bounded number of findings from one independent category."""
    lines = [f"### {category.category.value.replace('_', ' ').title()}", ""]
    limit = report.report_configuration.markdown_findings_per_category
    if not category.findings:
        lines.extend(["No positive-scoring finding is currently retained.", ""])
        return lines
    for finding in category.findings[:limit]:
        lines.extend(_render_finding(finding))
    if len(category.findings) > limit:
        lines.extend(
            [
                f"{len(category.findings) - limit} additional finding(s) remain in `report.json`.",
                "",
            ]
        )
    return lines


def _render_finding(finding: FindingRecord) -> list[str]:
    """Render one exact observation without turning its interpretation into causality."""
    module = finding.module or "@root"
    evidence = ", ".join(f"`{item.name}={item.value:.6g}`" for item in finding.evidence)
    return [
        (
            f"- **#{finding.rank} `{module}` call `{finding.call_index}`** — "
            f"`{finding.signal}.{finding.tensor_path}.{finding.metric}`; "
            f"first `{finding.first.value:.6g}`, latest `{finding.latest.value:.6g}`, "
            f"score `{finding.ranking_score:.6g}`."
        ),
        f"  Evidence: {evidence or 'first/latest values only'}.",
        f"  Interpretation: {finding.interpretation}",
        "",
    ]
