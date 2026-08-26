from __future__ import annotations

import html
from io import BytesIO
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    KeepTogether,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import AnalysisResponse, Metric


NAVY = colors.HexColor("#0B1F33")
BLUE = colors.HexColor("#0070F3")
AMBER = colors.HexColor("#B9770E")
PALE_BLUE = colors.HexColor("#EAF3FF")
PALE_AMBER = colors.HexColor("#FFF5E3")
PALE_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#66717D")
LINE = colors.HexColor("#D6DCE2")


def _ascii_dashes(value: object) -> str:
    return (
        str(value)
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )


def _safe(value: object) -> str:
    return html.escape(_ascii_dashes(value))


def _metric(run: AnalysisResponse, metric_id: str) -> Metric | None:
    return next((item for item in run.metrics if item.id == metric_id), None)


def _chart_data(run: AnalysisResponse, chart_id: str) -> list[dict]:
    chart = next((item for item in run.charts if item.id == chart_id), None)
    return chart.data if chart else []


def _format_number(value: object, decimals: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _ascii_dashes(value)
    if number.is_integer():
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def _metric_value(metric: Metric | None) -> str:
    if metric is None:
        return "Not available"
    value = _format_number(metric.value)
    if metric.unit == "USD NPV":
        return f"${_format_number(metric.value, 0)} NPV"
    return f"{value} {metric.unit}".strip()


def _humanize(value: object) -> str:
    return _ascii_dashes(value).replace("_", " ").title()


def _evidence_class(run: AnalysisResponse, prefix: str) -> str:
    evidence = next((item for item in run.evidence if item.id.startswith(prefix)), None)
    return evidence.data_class.value.upper() if evidence else "NOT AVAILABLE"


def _lineage_banner(
    run: AnalysisResponse, *, styles: dict[str, ParagraphStyle]
) -> Table:
    thermal = next(
        (
            item
            for item in run.evidence
            if item.id.startswith("thermal-")
            and item.source.startswith("https://api.fortyguard.com/v1/heatmap")
        ),
        None,
    )
    satellite = next(
        (
            item
            for item in run.evidence
            if item.metadata.get("dataset_id") == "GOOGLE/DYNAMICWORLD/V1"
        ),
        None,
    )
    years = sorted(int(year) for year in (thermal.metadata.get("observed_years", []) if thermal else []))
    backcasts = thermal.metadata.get("backcast_years", []) if thermal else []
    thermal_text = (
        f"FortyGuard observed history: {years[0]}-{years[-1]}; "
        f"backcast years: {len(backcasts)}."
        if years
        else "FortyGuard observed history did not pass the memo verification gate."
    )
    satellite_text = (
        "Dynamic World: OBSERVED; historical control stability gate passed."
        if satellite
        and satellite.data_class.value == "observed"
        and satellite.metadata.get("historical_control_stability_verified")
        else "Dynamic World historical control stability was not verified."
    )
    operations_text = (
        f"Operational forecast: {_evidence_class(run, 'forecast-weather-')}; "
        f"BMS: {_evidence_class(run, 'bms-telemetry-')}; "
        f"cooling model: {_evidence_class(run, 'cooling-model-')}."
    )
    body = Paragraph(
        "<b>INPUT LINEAGE</b><br/>"
        f"{_safe(thermal_text)}<br/>{_safe(satellite_text)}<br/>{_safe(operations_text)}",
        styles["body"],
    )
    table = Table([[body]], colWidths=[6.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_AMBER),
                ("BOX", (0, 0), (-1, -1), 0.65, AMBER),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _display_warnings(run: AnalysisResponse) -> list[str]:
    stability_verified = any(
        item.metadata.get("historical_control_stability_verified")
        for item in run.evidence
    )
    stale_warning = (
        "Historical regional controls were selected using current satellite land-cover "
        "similarity; their historical land-cover stability is not yet verified."
    )
    warnings: list[str] = []
    for warning in run.warnings:
        if warning == stale_warning and stability_verified:
            warnings.append(
                "Historical control stability was verified by the saved Dynamic World "
                "evidence record; the earlier provider-stage caution is superseded."
            )
        else:
            warnings.append(warning)
    return warnings or ["No workflow warnings were recorded."]


def _styles() -> dict[str, ParagraphStyle]:
    samples = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MemoTitle",
            parent=samples["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "MemoSubtitle",
            parent=samples["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=MID_GRAY,
            spaceAfter=18,
        ),
        "section": ParagraphStyle(
            "MemoSection",
            parent=samples["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "MemoBody",
            parent=samples["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=NAVY,
            spaceAfter=6,
            splitLongWords=True,
        ),
        "small": ParagraphStyle(
            "MemoSmall",
            parent=samples["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=9.5,
            textColor=MID_GRAY,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "MemoTableHeader",
            parent=samples["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "table": ParagraphStyle(
            "MemoTable",
            parent=samples["Normal"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.5,
            textColor=NAVY,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "kpi": ParagraphStyle(
            "MemoKpi",
            parent=samples["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "MemoKpiLabel",
            parent=samples["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.5,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
        ),
    }


def _footer(canvas, document) -> None:
    canvas.saveState()
    width, _ = LETTER
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(0.5 * inch, 0.45 * inch, width - 0.5 * inch, 0.45 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(0.5 * inch, 0.28 * inch, "FortyCool - evidence-first advisory memorandum")
    canvas.drawRightString(width - 0.5 * inch, 0.28 * inch, f"Page {document.page}")
    canvas.restoreState()


def _evidence_links(
    evidence_ids: Iterable[str], *, run_id: str, base_url: str, style: ParagraphStyle
) -> Paragraph:
    links = [
        f'<link href="{_safe(base_url)}/runs/{_safe(run_id)}/evidence/{_safe(evidence_id)}" '
        f'color="#0070F3">{_safe(evidence_id)}</link>'
        for evidence_id in evidence_ids
    ]
    return Paragraph("<br/>".join(links) if links else "No evidence reference", style)


def _metric_table(
    run: AnalysisResponse, *, base_url: str, styles: dict[str, ParagraphStyle]
) -> LongTable:
    metric_ids = [
        "local_thermal_drift_c",
        "local_drift_rate_c_per_year",
        "temperature_eligible_hours_lost",
        "local_excess_built_surface_change_percentage_points",
        "thermal_land_cover_association_correlation",
        "annual_thermal_drift_energy_penalty_kwh",
        "thermal_drift_npv",
        "forecast_savings_kwh",
        "forecast_safety_margin_c",
    ]
    selected = [metric for metric_id in metric_ids if (metric := _metric(run, metric_id))]
    header = [
        Paragraph("Metric", styles["table_header"]),
        Paragraph("Result", styles["table_header"]),
        Paragraph("Confidence", styles["table_header"]),
        Paragraph("Evidence", styles["table_header"]),
    ]
    rows = [header]
    for metric in selected:
        confidence = f"{metric.confidence * 100:.0f}%<br/>{_safe(metric.data_class.value)}"
        rows.append(
            [
                Paragraph(_safe(metric.label), styles["table"]),
                Paragraph(_safe(_metric_value(metric)), styles["table"]),
                Paragraph(confidence, styles["table"]),
                _evidence_links(
                    metric.evidence_ids,
                    run_id=run.run_id,
                    base_url=base_url,
                    style=styles["small"],
                ),
            ]
        )
    table = LongTable(rows, colWidths=[1.75 * inch, 1.15 * inch, 0.85 * inch, 3.05 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
            ]
        )
    )
    return table


def _history_table(
    data: list[dict], *, styles: dict[str, ParagraphStyle]
) -> LongTable | Paragraph:
    if not data:
        return Paragraph("Annual temperature history was not available.", styles["body"])
    rows = [
        [
            Paragraph("Year", styles["table_header"]),
            Paragraph("Site air temp", styles["table_header"]),
            Paragraph("Control air temp", styles["table_header"]),
            Paragraph("Site-control gap", styles["table_header"]),
            Paragraph("Site eligible hours", styles["table_header"]),
            Paragraph("Control eligible hours", styles["table_header"]),
        ]
    ]
    for item in data:
        rows.append(
            [
                _format_number(item.get("year"), 0),
                f'{_format_number(item.get("site_mean_temperature_c"))} C',
                f'{_format_number(item.get("control_mean_temperature_c"))} C',
                f'{_format_number(item.get("site_control_gap_c"))} C',
                _format_number(item.get("site_eligible_hours"), 0),
                _format_number(item.get("control_eligible_hours"), 0),
            ]
        )
    table = LongTable(rows, colWidths=[0.55 * inch, 1.05 * inch, 1.1 * inch, 1.0 * inch, 1.25 * inch, 1.3 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _land_cover_table(
    data: list[dict], *, styles: dict[str, ParagraphStyle]
) -> LongTable | Paragraph:
    if not data:
        return Paragraph("Historical land-cover series was not available.", styles["body"])
    rows = [
        [
            Paragraph("Year", styles["table_header"]),
            Paragraph("Site built surface", styles["table_header"]),
            Paragraph("Control built surface", styles["table_header"]),
            Paragraph("Local excess", styles["table_header"]),
            Paragraph("Site tree canopy", styles["table_header"]),
            Paragraph("Control tree canopy", styles["table_header"]),
        ]
    ]
    for item in data:
        rows.append(
            [
                _format_number(item.get("year"), 0),
                f'{_format_number(item.get("site_built_surface_percent"))}%',
                f'{_format_number(item.get("control_built_surface_percent"))}%',
                f'{_format_number(item.get("local_excess_built_surface_percent"))} pp',
                f'{_format_number(item.get("site_tree_canopy_percent"))}%',
                f'{_format_number(item.get("control_tree_canopy_percent"))}%',
            ]
        )
    table = LongTable(rows, colWidths=[0.55 * inch, 1.12 * inch, 1.18 * inch, 0.95 * inch, 1.1 * inch, 1.2 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _evidence_appendix(
    run: AnalysisResponse, *, base_url: str, styles: dict[str, ParagraphStyle]
) -> list:
    flowables: list = []
    for evidence in run.evidence:
        endpoint_url = f"{base_url}/runs/{run.run_id}/evidence/{evidence.id}"
        source = _safe(evidence.source)
        if evidence.source.startswith("http"):
            source = f'<link href="{source}" color="#0070F3">{source}</link>'
        activity_ids = evidence.metadata.get("activity_ids", [])
        if evidence.activity_id and evidence.activity_id not in activity_ids:
            activity_ids = [evidence.activity_id, *activity_ids]
        activity_text = ", ".join(str(item) for item in activity_ids[:8]) or "Not applicable"
        if len(activity_ids) > 8:
            activity_text += f" (+{len(activity_ids) - 8} more in the evidence record)"
        block = [
            [
                Paragraph(
                    f'<link href="{_safe(endpoint_url)}" color="#0070F3"><b>{_safe(evidence.id)}</b></link>',
                    styles["table"],
                ),
                Paragraph(_safe(evidence.data_class.value.upper()), styles["table"]),
            ],
            [Paragraph("Description", styles["small"]), Paragraph(_safe(evidence.description), styles["table"])],
            [Paragraph("Source", styles["small"]), Paragraph(source, styles["small"])],
            [Paragraph("Provider endpoint", styles["small"]), Paragraph(_safe(evidence.endpoint or "Not applicable"), styles["small"])],
            [Paragraph("Activity IDs", styles["small"]), Paragraph(_safe(activity_text), styles["small"])],
            [Paragraph("Retrieved", styles["small"]), Paragraph(_safe(evidence.retrieved_at.isoformat()), styles["small"])],
        ]
        table = Table(block, colWidths=[1.25 * inch, 5.55 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
                    ("SPAN", (0, 0), (0, 0)),
                    ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        flowables.extend([KeepTogether([table, Spacer(1, 8)])])
    return flowables


def build_investment_memo(run: AnalysisResponse, *, base_url: str) -> bytes:
    styles = _styles()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.58 * inch,
        title="FortyCool ThermalDrift Investment Memorandum",
        author="FortyCool",
        subject=f"Evidence-first memorandum for run {run.run_id}",
    )
    story: list = [
        Paragraph("FORTYCOOL", styles["section"]),
        Paragraph("ThermalDrift Investment Memorandum", styles["title"]),
        Paragraph(
            f"Run {_safe(run.run_id)}<br/>Status: {_safe(run.status.value)} | "
            f"Confidence tier: {_safe(run.confidence_tier.value)}<br/>"
            "Advisory only - no commands were sent to facility infrastructure.",
            styles["subtitle"],
        ),
        _lineage_banner(run, styles=styles),
        Spacer(1, 12),
    ]

    kpis = [
        ("Local thermal drift", _metric_value(_metric(run, "local_thermal_drift_c"))),
        ("Eligible hours lost", _metric_value(_metric(run, "temperature_eligible_hours_lost"))),
        ("Local excess buildout", _metric_value(_metric(run, "local_excess_built_surface_change_percentage_points"))),
        ("Lease-life exposure", _metric_value(_metric(run, "thermal_drift_npv"))),
    ]
    kpi_table = Table(
        [
            [Paragraph(_safe(value), styles["kpi"]) for _, value in kpis],
            [Paragraph(_safe(label.upper()), styles["kpi_label"]) for label, _ in kpis],
        ],
        colWidths=[1.7 * inch] * 4,
    )
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("TOPPADDING", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
            ]
        )
    )
    story.extend(
        [
            kpi_table,
            Spacer(1, 12),
            Paragraph("1. Executive conclusion", styles["section"]),
            Paragraph(_safe(run.summary), styles["body"]),
            Paragraph(
                "The key analytical control is difference-in-differences: FortyCool compares the "
                "change at the facility with the change at disclosed, land-cover-matched regional "
                "controls. The local gap is interpreted as site-specific screening evidence, not "
                "as proof of causation.",
                styles["body"],
            ),
            Paragraph("2. Decision metrics and lineage", styles["section"]),
            _metric_table(run, base_url=base_url, styles=styles),
            Paragraph("3. Historical thermal control", styles["section"]),
            Paragraph(
                "Temperature values are site and matched-control air-temperature screening "
                "summaries. Eligible cooling hours are annualized from the disclosed seasonal "
                "window and are not equivalent to verified facility free-cooling operation.",
                styles["body"],
            ),
            _history_table(_chart_data(run, "thermal_drift_timeseries"), styles=styles),
            Paragraph("4. Land-cover attribution context", styles["section"]),
            Paragraph(
                "Dynamic World provides modeled annual land-cover probabilities. The temporal "
                "association with the thermal gap supports an underwriting hypothesis but does "
                "not establish a causal mechanism.",
                styles["body"],
            ),
            _land_cover_table(_chart_data(run, "historical_land_cover_timeseries"), styles=styles),
            Paragraph("5. Operational impact", styles["section"]),
        ]
    )

    if run.recommendations:
        recommendation = run.recommendations[0]
        operation_rows = [
            [Paragraph("Verdict", styles["table_header"]), Paragraph("Advisory action", styles["table_header"]), Paragraph("12-hour savings", styles["table_header"]), Paragraph("Safety margin", styles["table_header"])],
            [
                Paragraph(_safe(_humanize(recommendation.verdict.value)), styles["table"]),
                Paragraph(_safe(recommendation.description), styles["table"]),
                Paragraph(f"{_format_number(recommendation.expected_savings_kwh)} kWh", styles["table"]),
                Paragraph(f"{_format_number(recommendation.safety_margin_c)} C", styles["table"]),
            ],
        ]
        operation_table = Table(operation_rows, colWidths=[1.0 * inch, 3.85 * inch, 0.95 * inch, 1.0 * inch])
        operation_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(operation_table)
    else:
        story.append(Paragraph("No operational recommendation was produced.", styles["body"]))
    story.append(
        Paragraph(
            "Operational savings are modeled outputs. When BMS evidence is simulated, the result "
            "is indicative and must not be represented as measured facility performance.",
            styles["body"],
        )
    )

    story.extend(
        [
            Paragraph("6. Assumptions and limitations", styles["section"]),
            Paragraph("<b>Assumptions</b>", styles["body"]),
        ]
    )
    if run.assumptions:
        for assumption in run.assumptions:
            story.append(
                Paragraph(
                    f"- {_safe(assumption.field)} = {_safe(assumption.value)}. "
                    f"{_safe(assumption.reason)} User confirmed: {_safe(assumption.user_confirmed)}.",
                    styles["body"],
                )
            )
    else:
        story.append(Paragraph("- No unconfirmed assumptions were recorded.", styles["body"]))
    warning_paragraphs = [
        Paragraph(f"- {_safe(warning)}", styles["body"])
        for warning in _display_warnings(run)
    ]
    story.append(
        KeepTogether(
            [Paragraph("<b>Warnings and limitations</b>", styles["body"]), warning_paragraphs[0]]
        )
    )
    story.extend(warning_paragraphs[1:])

    story.extend(
        [
            PageBreak(),
            Paragraph("7. Evidence appendix", styles["section"]),
            Paragraph(
                "Every decision metric above resolves to one or more evidence records. Provider "
                "activity IDs are included when available; open the linked evidence record for "
                "the complete metadata payload.",
                styles["body"],
            ),
            *_evidence_appendix(run, base_url=base_url, styles=styles),
            Paragraph("Important notice", styles["section"]),
            Paragraph(
                "This memorandum is an automated screening deliverable for due diligence. It is "
                "not an engineering design, valuation opinion, guarantee of savings, or command "
                "to operate cooling infrastructure. Confirm facility telemetry, tariff inputs, "
                "equipment limits, and operator-approved safety constraints before relying on it.",
                styles["body"],
            ),
        ]
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
