"""
backend/report.py — Detailed county flood-risk PDF report.

A multi-section, print-ready PDF covering current nowcast, month-ahead and
6-month compiled outlook, the climate inputs actually used, historical
context, model performance, and data-quality caveats — meant to stand on its
own if forwarded to a partner agency without access to the dashboard.
"""

import io
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ACCENT = colors.HexColor("#1D4ED8")
INK = colors.HexColor("#101828")
MUTED = colors.HexColor("#4B5768")
FAINT = colors.HexColor("#8792A2")
BORDER = colors.HexColor("#DDE3EC")
RISK_COLORS = {
    "Low": colors.HexColor("#15803D"),
    "Moderate": colors.HexColor("#B7791E"),
    "High": colors.HexColor("#C2410C"),
    "Critical": colors.HexColor("#DC2626"),
}

FEATURE_LABELS = {
    "rainfall_mm": "Rainfall (mm)", "soil_moisture_mm": "Soil Moisture (mm)",
    "max_temperature_celsius": "Max Temp (°C)", "min_temperature_celsius": "Min Temp (°C)",
    "vapor_pressure_deficit_kPa": "VPD (kPa)", "wetland_fraction": "Wetland Fraction",
    "elevation_m": "Elevation (m)", "slope_deg": "Slope (°)", "ndvi": "NDVI",
    "flood_prev_month": "Flooded Last Month",
}


def _line_chart_png(labels: list, values: list, title: str, color="#1D4ED8", ylabel="Probability (%)") -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 2.1), dpi=150)
    ax.plot(labels, [v * 100 for v in values], color=color, linewidth=2, marker="o", markersize=4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_county_report(
    county: str,
    generated_at: dt.datetime,
    nowcast: dict,
    outlook: dict | None,
    historical: dict,
    seasonal: list,
    nowcast_advice: str,
    outlook_advice: str | None,
    model_name: str,
    outlook_model_name: str,
    for_period: str | None = None,
    climate_inputs: dict | None = None,
    field_source: dict | None = None,
    multistep: list | None = None,
    model_metrics: dict | None = None,
    data_quality_note: str | None = None,
) -> bytes:
    """Returns the rendered PDF as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=16 * mm, bottomMargin=14 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, textColor=ACCENT, spaceAfter=2)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=9, textColor=MUTED)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=5, textColor=INK)
    body = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=9.5, leading=13.5)
    small = ParagraphStyle("SmallX", parent=styles["Normal"], fontSize=7.7, textColor=FAINT, leading=10.5)

    story = []
    story.append(Paragraph("South Sudan Flood Early Warning System", sub_style))
    story.append(Paragraph(f"{county} — Risk Report{f' for {for_period}' if for_period else ''}", title_style))
    story.append(Paragraph(
        f"Generated {generated_at.strftime('%d %B %Y, %H:%M')} UTC · Nowcast model: {model_name} · "
        f"Outlook model: {outlook_model_name}", sub_style,
    ))
    story.append(Spacer(1, 8 * mm))

    # ── KPI row ──────────────────────────────────────────────────────────────
    risk_color = RISK_COLORS.get(nowcast["risk_tier"], ACCENT)
    kpi_data = [
        ["Current Nowcast", "Month-Ahead Outlook", "Historical Rate", "National Rank"],
        [
            f"{nowcast['probability']*100:.1f}%  ({nowcast['risk_tier']})",
            f"{outlook['probability']*100:.1f}%  ({outlook['risk_tier']})" if outlook else "n/a",
            f"{historical['flood_rate']*100:.1f}%  ({historical['flood_events']} events)",
            f"#{historical['rank']} of {historical['n_counties']}",
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[42 * mm] * 4)
    kpi_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
        ("FONTSIZE", (0, 1), (-1, 1), 12),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (0, 1), risk_color),
        ("TEXTCOLOR", (1, 1), (1, 1), RISK_COLORS.get(outlook["risk_tier"], ACCENT) if outlook else colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
    ]))
    story.append(kpi_table)

    # ── Recommendations ──────────────────────────────────────────────────────
    story.append(Paragraph("Recommended Action — Now", h2))
    story.append(Paragraph(nowcast_advice, body))
    if outlook and outlook_advice:
        story.append(Paragraph("Recommended Action — Next Month", h2))
        story.append(Paragraph(outlook_advice, body))

    # ── Climate inputs actually used ─────────────────────────────────────────
    if climate_inputs:
        story.append(Paragraph("Climate Inputs Used for This Nowcast", h2))
        rows = [["Field", "Value", "Source"]]
        for key, label in FEATURE_LABELS.items():
            if key not in climate_inputs:
                continue
            src = (field_source or {}).get(key, "historical_fallback")
            src_label = "Live" if src == "live" else "Historical median"
            rows.append([label, f"{climate_inputs[key]:.2f}", src_label])
        t = Table(rows, colWidths=[55 * mm, 35 * mm, 45 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#EEF1F5")),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)

    # ── Charts: seasonal + compiled outlook ──────────────────────────────────
    if seasonal:
        labels = [s["month"] for s in seasonal]
        values = [s["probability"] for s in seasonal]
        chart_png = _line_chart_png(labels, values, "Seasonal Risk Pattern (historical climate medians)")
        story.append(Spacer(1, 4 * mm))
        story.append(RLImage(io.BytesIO(chart_png), width=160 * mm, height=50 * mm))

    if multistep:
        labels = [m["label"] for m in multistep]
        values = [m["probability"] for m in multistep]
        chart_png = _line_chart_png(labels, values, "Compiled Outlook — Next 6 Months (recursive GRU forecast)", color="#7C3AED")
        story.append(Spacer(1, 4 * mm))
        story.append(RLImage(io.BytesIO(chart_png), width=160 * mm, height=50 * mm))
        story.append(Paragraph(
            "Each month's forecast feeds the next step recursively, so uncertainty compounds further out — "
            "read this as an indicative trend, not independent monthly forecasts.", small,
        ))

    # ── Model performance ─────────────────────────────────────────────────────
    if model_metrics:
        story.append(Paragraph("Model Performance (National Test Set, 2024–2025)", h2))
        rows = [["Model", "AUC-ROC", "F1", "Precision", "Recall"]]
        for name, m in model_metrics.items():
            rows.append([name, f"{m['auc_roc']:.4f}", f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"])
        t = Table(rows, colWidths=[45 * mm, 25 * mm, 22 * mm, 25 * mm, 22 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#EEF1F5")),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(t)

    # ── Data quality & methodology ────────────────────────────────────────────
    story.append(Paragraph("Data Quality Note", h2))
    story.append(Paragraph(
        data_quality_note or (
            "The satellite water-extent signal underlying flood labels has a growing rate of missing "
            "values for 2022–2025 (roughly 4.6% missing in 2022 rising to 7.9% by 2025, vs. 0% missing "
            "every year 2011–2021), while rainfall shows no corresponding anomaly — consistent with a "
            "processing-lag gap in the satellite source rather than a real decline in flooding. Historical "
            "rates for 2022–2025 above likely undercount real events."
        ), body,
    ))

    story.append(Paragraph("Methodology", h2))
    story.append(Paragraph(
        "The nowcast model predicts the current month's flood probability from that month's own climate "
        "inputs (live data where available, historical medians otherwise). The outlook model is a recurrent "
        "neural network (GRU) that forecasts using only history strictly before the target month, with no "
        "contemporaneous data — a genuine forward-looking forecast. Neither model is trained on "
        "daily-resolution flood ground truth; the source dataset records flood occurrence at monthly "
        "granularity only (2011–2025, 79 South Sudan counties). Predictions themselves are always forward-"
        "looking from the current date, regardless of the historical training window.", body,
    ))

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "South Sudan Flood Early Warning System — automatically generated report. Not a substitute for "
        "official government or humanitarian agency flood advisories.", small,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _calendar_heatmap_png(calendar_rows: list) -> bytes:
    """calendar_rows: list of {year, month, flood} -> a year x month heatmap."""
    years = sorted({r["year"] for r in calendar_rows})
    grid = {(r["year"], r["month"]): r["flood"] for r in calendar_rows}
    data = [[grid.get((y, m), 0) for m in range(1, 13)] for y in years]

    fig, ax = plt.subplots(figsize=(6.6, max(2.2, 0.16 * len(years))), dpi=150)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=6)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=6)
    ax.set_title("National Flood Rate by Year & Month", fontsize=9, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Share of counties flooded")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_national_report(
    generated_at: dt.datetime,
    total_flood_events: int,
    national_mean_rate: float,
    n_counties: int,
    risk_tier_counts: dict,
    highest_risk_county: str,
    watchlist: list,
    calendar: list,
    model_name: str,
    outlook_model_name: str,
    model_metrics: dict,
    model_selection_criterion: str,
    data_quality_note: str,
) -> bytes:
    """National executive summary — proposal-ready overview across all
    counties, not scoped to one county's nowcast."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=16 * mm, bottomMargin=14 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=19, textColor=ACCENT, spaceAfter=2)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=9, textColor=MUTED)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=5, textColor=INK)
    body = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=9.5, leading=13.5)
    small = ParagraphStyle("SmallX", parent=styles["Normal"], fontSize=7.7, textColor=FAINT, leading=10.5)

    story = []
    story.append(Paragraph("South Sudan Flood Early Warning System", sub_style))
    story.append(Paragraph("National Executive Summary", title_style))
    story.append(Paragraph(
        f"Generated {generated_at.strftime('%d %B %Y, %H:%M')} UTC · {n_counties} counties · "
        f"Nowcast: {model_name} · Outlook: {outlook_model_name}", sub_style,
    ))
    story.append(Spacer(1, 8 * mm))

    kpi_data = [
        ["Total Flood Events", "National Mean Rate", "Critical Counties", "Highest-Risk County"],
        [
            f"{total_flood_events:,}", f"{national_mean_rate*100:.1f}%",
            str(risk_tier_counts.get("Critical", 0)), highest_risk_county,
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[42 * mm] * 4)
    kpi_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, 0), 7.5), ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
        ("FONTSIZE", (0, 1), (-1, 1), 13), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (0, 1), RISK_COLORS["Critical"]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3), ("TOPPADDING", (0, 1), (-1, 1), 2),
    ]))
    story.append(kpi_table)

    story.append(Paragraph("Risk Distribution", h2))
    dist_rows = [["Risk Tier", "Counties"]] + [[t, str(risk_tier_counts.get(t, 0))] for t in ["Critical", "High", "Moderate", "Low"]]
    dist_table = Table(dist_rows, colWidths=[40 * mm, 30 * mm])
    dist_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(dist_table)

    if calendar:
        story.append(Paragraph("Historical Flood Calendar (Training Record)", h2))
        chart_png = _calendar_heatmap_png(calendar)
        story.append(RLImage(io.BytesIO(chart_png), width=160 * mm, height=160 * mm * (max(2.2, 0.16 * len({r["year"] for r in calendar})) / 6.6)))

    story.append(Paragraph("Watchlist — Highest Historical Risk", h2))
    wl_rows = [["#", "County", "Historical Rate", "Events", "Risk"]]
    for i, w in enumerate(watchlist, start=1):
        wl_rows.append([str(i), w["county"], f"{w['flood_rate']*100:.1f}%", str(w["flood_events"]), w["risk_tier"]])
    wl_table = Table(wl_rows, colWidths=[10 * mm, 45 * mm, 35 * mm, 25 * mm, 25 * mm])
    wl_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#EEF1F5")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(wl_table)

    story.append(Paragraph("Model Performance", h2))
    story.append(Paragraph(model_selection_criterion, body))
    mm_rows = [["Model", "AUC-ROC", "F1", "Precision", "Recall"]]
    for name, m in model_metrics.items():
        mm_rows.append([name, f"{m['auc_roc']:.4f}", f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"])
    mm_table = Table(mm_rows, colWidths=[45 * mm, 25 * mm, 22 * mm, 25 * mm, 22 * mm])
    mm_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TEXTCOLOR", (0, 0), (-1, 0), FAINT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#EEF1F5")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(mm_table)

    story.append(Paragraph("Data Quality Note", h2))
    story.append(Paragraph(data_quality_note, body))

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "South Sudan Flood Early Warning System — automatically generated national summary. Not a "
        "substitute for official government or humanitarian agency flood advisories.", small,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
