import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { ModelInfo, OverviewData } from "../types";
import { Card, Typography } from "antd";
import DocPage from "../components/DocPage";

const { Text } = Typography;

const DATA_SOURCES = [
  {
    name: "Historical Flood Record",
    period: "2011–2025, static",
    detail: "Monthly, per-county flood occurrence and climate/terrain covariates for all 79 counties — the labeled dataset the deployed models are trained and validated on.",
  },
  {
    name: "Open-Meteo",
    period: "Live, free, no API key",
    detail: "Global weather model reanalysis and forecast. Feeds the flood model's live climate inputs — rainfall, soil moisture, temperature, vapour pressure deficit.",
  },
  {
    name: "NASA POWER",
    period: "Live, free, no API key",
    detail: "Automatic fallback for the same climate inputs above whenever Open-Meteo is unreachable — a second independent provider, so the nowcast doesn't silently drop to historical medians the moment one provider has an outage.",
  },
  {
    name: "OpenWeatherMap",
    period: "Live, free tier, API key required",
    detail: "Human-facing current conditions, the Live Updates terminal, and a 5-day/hourly forecast, shown for situational awareness only. Kept deliberately separate from the model's own climate inputs above, so the two are never confused for one another.",
  },
];

const TOC = [
  { id: "quick-facts", label: "Quick Facts" },
  { id: "data-sources", label: "Data Sources" },
  { id: "nowcast-outlook", label: "Nowcast vs. Outlook" },
  { id: "methodology", label: "Methodology" },
  { id: "data-quality", label: "Data Quality & Limitations" },
  { id: "intended-use", label: "Intended Use" },
];

export default function About() {
  const [meta, setMeta] = useState<ModelInfo | null>(null);
  const [overview, setOverview] = useState<OverviewData | null>(null);

  useEffect(() => { api.modelInfo().then(setMeta).catch(console.error); }, []);
  useEffect(() => { api.overview().then(setOverview).catch(console.error); }, []);

  return (
    <DocPage eyebrow="South Sudan Flood Early Warning System" title="About This System" toc={TOC}>

          <p style={{ fontFamily: "inherit", fontSize: "1.2rem", lineHeight: 1.65, color: "var(--color-text)", margin: "1.6rem 0 0" }}>
            This system combines fifteen years of historical flood records with live climate data to give South
            Sudan's 79 counties a genuine early-warning capability — a same-day nowcast of flood risk, a
            month-ahead statistical outlook, and a national bulletin — built to hold up under the same scrutiny a
            government or UN humanitarian program would apply before relying on it.
          </p>

          <h2 id="quick-facts" style={{ ...h2, marginTop: "2.2rem", fontSize: "1.1rem" }}>Quick Facts</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.9rem", marginTop: "1rem" }}>
            <Fact label="Counties Covered" value="79" />
            <Fact label="Historical Record" value="2011–2025" />
            <Fact label="Recorded Flood-Months" value={overview ? overview.total_flood_events.toLocaleString() : "…"} />
            <Fact label="National Mean Rate" value={overview ? `${(overview.national_mean_rate * 100).toFixed(1)}%` : "…"} />
            <Fact label="Nowcast Refresh" value="Every 20 min" />
            <Fact label="Deployed Model" value={meta?.best_model_name ?? "…"} />
          </div>

          <Rule />

          <h2 id="data-sources" style={h2}>Data Sources</h2>
          {DATA_SOURCES.map((s, i) => (
            <div key={s.name} style={{ marginTop: i === 0 ? "1.4rem" : "1.6rem" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: "0.7rem", flexWrap: "wrap" }}>
                <span style={{ fontWeight: 700, fontSize: "1.02rem" }}>{s.name}</span>
                <span style={{ fontSize: "0.74rem", color: "var(--color-primary)", fontWeight: 600 }}>{s.period}</span>
              </div>
              <p style={{ fontSize: "0.94rem", lineHeight: 1.7, color: "var(--color-text-muted)", marginTop: "0.35rem", maxWidth: 640 }}>
                {s.detail}
              </p>
            </div>
          ))}

          <Rule />

          <h2 id="nowcast-outlook" style={h2}>Nowcast vs. Outlook</h2>
          <p style={bodyText}>{meta ? meta.nowcast_vs_outlook : "Loading…"}</p>

          <Rule />

          <h2 id="methodology" style={h2}>Methodology, in Brief</h2>
          {meta ? (
            <dl style={{ margin: "1.2rem 0 0" }}>
              {Object.entries(meta.methodology).map(([k, v]) => (
                <div key={k} style={{ marginBottom: "1rem" }}>
                  <dt style={{ fontWeight: 700, textTransform: "uppercase", fontSize: "0.68rem", color: "var(--color-text-muted)", letterSpacing: "0.05em" }}>
                    {k.replace(/_/g, " ")}
                  </dt>
                  <dd style={{ margin: "0.2rem 0 0", fontSize: "0.94rem", color: "var(--color-text-muted)", lineHeight: 1.7 }}>{v}</dd>
                </div>
              ))}
            </dl>
          ) : <p style={bodyText}>Loading…</p>}
          <p style={{ fontSize: "0.88rem", marginTop: "0.4rem" }}>
            <Link to="/model" style={{ color: "var(--color-primary)", textDecoration: "none", fontWeight: 600 }}>
              Full model comparison, statistical significance tests &amp; ablation study →
            </Link>
          </p>

          <Rule />

          <h2 id="data-quality" style={h2}>Data Quality &amp; Known Limitations</h2>
          <p style={bodyText}>
            The satellite water-extent signal underlying historical flood labels shows a growing rate of missing
            values from 2022 onward — roughly 4.6% missing in 2022, rising to 7.9% by 2025 — while rainfall in the
            same period shows no corresponding anomaly.
          </p>
          <blockquote style={{
            margin: "1.1rem 0", padding: "0.2rem 0 0.2rem 1.1rem", borderLeft: "3px solid var(--color-primary)",
            fontFamily: "inherit", fontSize: "1.02rem", lineHeight: 1.65, color: "var(--color-text)", fontStyle: "italic",
          }}>
            This is consistent with a processing gap in the satellite data pipeline, not a real decline in flooding.
            Historical rates for 2022–2025 likely undercount actual events — a caveat repeated on every county
            profile rather than silently corrected, since the underlying gap has not been independently verified or
            fixed at the source.
          </blockquote>
          <p style={bodyText}>
            Neither the nowcast nor the outlook model is trained on daily-resolution flood labels, because none
            exist in the source data — flood occurrence is recorded at monthly granularity. "Daily" in this system
            refers to daily-updating live climate inputs feeding a monthly-resolution prediction, not a
            daily-resolution ground truth. Static terrain covariates (wetland fraction, elevation, slope) and
            antecedent flood state are not available live and fall back to each county's historical median when
            live data can't supply them — always labeled as such, never presented as current.
          </p>

          <Rule />

          <h2 id="intended-use" style={h2}>Intended Use</h2>
          <p style={bodyText}>
            Built as a decision-support tool for county administrations, national disaster management authorities,
            and humanitarian response coordinators — to prioritize attention and pre-positioning, not as a sole
            basis for emergency action. It is not a substitute for official government flood advisories or
            on-the-ground verification, and its live-data dependencies mean forecast quality is bounded by those
            providers' own coverage of South Sudan.
          </p>
          <p style={{ fontSize: "0.88rem", marginTop: "0.6rem" }}>
            <Link to="/contact" style={{ color: "var(--color-primary)", textDecoration: "none", fontWeight: 600 }}>
              Who to contact in an emergency, and how to report an issue with this system →
            </Link>
          </p>

          <Rule />

          <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", lineHeight: 1.7 }}>
            Developed by Chol Monykuch as an academic research project on flood early warning for South Sudan.
            <br />
            See <Link to="/model" style={{ color: "var(--color-primary)" }}>Model &amp; Validation</Link> for full
            statistics, <Link to="/glossary" style={{ color: "var(--color-primary)" }}>Glossary</Link> for term
            definitions, or <Link to="/" style={{ color: "var(--color-primary)" }}>Overview</Link> for the national
            dashboard.
          </p>
    </DocPage>
  );
}

const h2: React.CSSProperties = { fontSize: "1.4rem", fontWeight: 700, letterSpacing: "-0.01em", color: "var(--color-text)", marginTop: 0 };
const bodyText: React.CSSProperties = { fontSize: "0.94rem", lineHeight: 1.75, color: "var(--color-text-muted)", marginTop: "1rem", maxWidth: 660 };

function Rule() {
  return <div style={{ height: 1, background: "var(--color-border)", margin: "2.4rem 0" }} />;
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <Card size="small" styles={{ body: { padding: "10px 12px" } }} style={{ background: "var(--color-bg)" }}>
      <Text type="secondary" style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.05em", display: "block" }}>
        {label}
      </Text>
      <Text strong className="tabular" style={{ fontSize: 16 }}>
        {value}
      </Text>
    </Card>
  );
}
