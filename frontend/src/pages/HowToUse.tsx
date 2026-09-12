import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { ModelInfo } from "../types";
import DocPage from "../components/DocPage";

const TIERS: Array<{ tier: string; range: string; color: string; meaning: string; action: string }> = [
  { tier: "Low", range: "under 25%", color: "var(--risk-low)", meaning: "No elevated signal in current conditions.", action: "Routine monitoring. No special action needed." },
  { tier: "Moderate", range: "25–50%", color: "var(--risk-moderate)", meaning: "Conditions are above the normal baseline for this county.", action: "Alert local officials; review evacuation routes and supply positions." },
  { tier: "High", range: "50–75%", color: "var(--risk-high)", meaning: "A majority-likelihood signal — more likely than not to flood.", action: "Issue a public warning; activate response teams; prepare evacuations." },
  { tier: "Critical", range: "75% and above", color: "var(--risk-critical)", meaning: "Strong, high-confidence signal of flood risk.", action: "Begin evacuations now; notify national emergency management immediately." },
];

const PAGE_TOUR: Array<{ name: string; to: string; desc: string }> = [
  { name: "Overview", to: "/", desc: "The national picture — live risk snapshot, historical baseline, and the full 79-county table in one place." },
  { name: "Alerts", to: "/alerts", desc: "A written bulletin, grouped by severity — the fastest way to see which counties need attention right now." },
  { name: "Live Updates", to: "/live", desc: "A live terminal of current weather and flood-risk readings across all 79 counties, refreshed automatically." },
  { name: "Map", to: "/map", desc: "The same live risk data plotted geographically — click any county to open its prediction." },
  { name: "Prediction", to: "/prediction", desc: "The core tool: pick a county to see today's nowcast, next month's outlook, and what's driving the number." },
  { name: "Historical", to: "/historical", desc: "Filter 2011–2025 flood records by county, year, and month to see long-run patterns." },
  { name: "Model & Validation", to: "/model", desc: "For the technically curious — how the model was built, tested, and how confident it actually is." },
  { name: "Reports", to: "/reports", desc: "Download a national summary or a detailed per-county report as a document." },
  { name: "Glossary", to: "/glossary", desc: "Plain-language definitions for every technical term used across the app." },
  { name: "Contact & Emergency Resources", to: "/contact", desc: "Who to actually call in a real flood emergency, and how to report a problem with this system." },
  { name: "About", to: "/about", desc: "Data sources, methodology, and known limitations in full." },
];

const TOC = [
  { id: "start-here", label: "Start Here" },
  { id: "risk-tiers", label: "Reading a Risk Tier" },
  { id: "nowcast-outlook", label: "Nowcast vs. Outlook" },
  { id: "charts", label: "Charts You'll See" },
  { id: "who-for", label: "Who This Is For" },
  { id: "limitations", label: "Important Limitations" },
  { id: "page-tour", label: "A Tour of Every Page" },
];

export default function HowToUse() {
  const [meta, setMeta] = useState<ModelInfo | null>(null);
  useEffect(() => { api.modelInfo().then(setMeta).catch(console.error); }, []);

  return (
    <DocPage eyebrow="A Guide For Everyone" title="How to Use This System" toc={TOC}>

          <p style={{ fontFamily: "inherit", fontSize: "1.15rem", lineHeight: 1.65, color: "var(--color-text)", margin: "1.4rem 0 0" }}>
            This system is free and open to anyone — a county administrator, a humanitarian coordinator, a
            researcher, or someone simply checking on their own community. It gives an up-to-date estimate of flood
            risk for each of South Sudan's 79 counties, updated automatically as live weather conditions change.
            This page explains what it's telling you and, just as importantly, what it isn't.
          </p>

          <Rule />

          <h2 id="start-here" style={h2}>Start Here — Three Steps</h2>
          <ol style={{ margin: "1rem 0 0", paddingLeft: "1.3rem", display: "flex", flexDirection: "column", gap: "0.9rem" }}>
            <li style={bodyLi}>
              Check <Link to="/" style={link}>Overview</Link> or <Link to="/alerts" style={link}>Alerts</Link> for
              the national picture — is anything currently elevated, and where.
            </li>
            <li style={bodyLi}>
              Open <Link to="/prediction" style={link}>Prediction</Link> and choose your county to see today's
              specific risk, expressed as a percentage and a tier (Low/Moderate/High/Critical).
            </li>
            <li style={bodyLi}>
              If the tier is High or Critical, click <b>View Recommendations</b> on that page for a concrete,
              South-Sudan-specific action checklist — not generic disaster-response boilerplate.
            </li>
          </ol>

          <Rule />

          <h2 id="risk-tiers" style={h2}>Reading a Risk Tier</h2>
          <p style={bodyText}>
            Every prediction is a probability — the model's estimate of how likely a flood is for that county this
            month, given current conditions. That percentage is grouped into four tiers so it's usable at a glance:
          </p>
          <div style={{ margin: "1.2rem 0 0", display: "flex", flexDirection: "column", gap: "0.9rem" }}>
            {TIERS.map((t) => (
              <div key={t.tier} style={{ display: "flex", gap: "1rem", alignItems: "baseline", borderLeft: `3px solid ${t.color}`, paddingLeft: "0.9rem" }}>
                <div style={{ width: 92, flexShrink: 0 }}>
                  <div style={{ fontWeight: 700, fontSize: "0.95rem", color: t.color }}>{t.tier}</div>
                  <div className="tabular" style={{ fontSize: "0.72rem", color: "var(--color-text-muted)" }}>{t.range}</div>
                </div>
                <div>
                  <div style={{ fontSize: "0.9rem", color: "var(--color-text)" }}>{t.meaning}</div>
                  <div style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", marginTop: 2 }}>{t.action}</div>
                </div>
              </div>
            ))}
          </div>
          <p style={{ ...bodyText, fontSize: "0.86rem", color: "var(--color-text-muted)" }}>
            These thresholds and the underlying advice come from the same file the Prediction page itself reads
            from — this page doesn't paraphrase separately, so it can't drift out of sync with what the app
            actually does.
          </p>

          <Rule />

          <h2 id="nowcast-outlook" style={h2}>Nowcast vs. Month-Ahead Outlook</h2>
          <p style={bodyText}>
            {meta?.nowcast_vs_outlook ?? "Loading…"}
          </p>
          <p style={bodyText}>
            In short: the <b>nowcast</b> answers "what does today's actual weather say about this month," using
            live climate data. The <b>outlook</b> answers "what does 15 years of history suggest about next
            month," using no live data at all — a statistical pattern, not a weather forecast. They can disagree,
            and that's expected.
          </p>

          <Rule />

          <h2 id="charts" style={h2}>Charts You'll See, and What They Mean</h2>
          <p style={bodyText}>
            Several pages use visualizations that go beyond a single number — a percentile bar showing whether
            today's rainfall is unusual for that specific county, a confusion matrix on the Model &amp; Validation
            page, a co-occurrence matrix on Historical. Rather than repeat those explanations here where they'd go
            stale, every one of those charts has its own small <b>"ⓘ What does this mean?"</b> toggle directly
            underneath it — click it for a plain-language explanation in context, right next to the chart it's
            describing. For any single term or acronym (AUC, VPD, percentile…), see the{" "}
            <Link to="/glossary" style={link}>Glossary</Link>.
          </p>

          <Rule />

          <h2 id="who-for" style={h2}>Who This Is For</h2>
          <dl style={{ margin: "1.2rem 0 0" }}>
            {[
              ["County & National Authorities", "Prioritize attention and pre-position resources across counties using the live Alerts bulletin and per-county Prediction pages."],
              ["Humanitarian Coordinators", "Cross-reference live risk against historical patterns (Historical page) to plan response staging ahead of a season."],
              ["Researchers & Students", "The Model & Validation page publishes full performance statistics, significance tests, and an ablation study — the same rigor a peer-reviewed methods section would expect."],
              ["Communities & the Public", "Anyone can look up their own county's current risk in plain language, with no login or technical background required."],
            ].map(([k, v]) => (
              <div key={k} style={{ marginBottom: "1rem" }}>
                <dt style={{ fontWeight: 700, fontSize: "0.95rem" }}>{k}</dt>
                <dd style={{ margin: "0.25rem 0 0", fontSize: "0.9rem", color: "var(--color-text-muted)", lineHeight: 1.6, maxWidth: 640 }}>{v}</dd>
              </div>
            ))}
          </dl>

          <Rule />

          <h2 id="limitations" style={h2}>Please Read — Important Limitations</h2>
          <blockquote style={{
            margin: "1.1rem 0", padding: "0.2rem 0 0.2rem 1.1rem", borderLeft: "3px solid var(--risk-critical)",
            fontFamily: "inherit", fontSize: "1.02rem", lineHeight: 1.65, color: "var(--color-text)", fontStyle: "italic",
          }}>
            This system is a decision-support tool, not a substitute for official government flood advisories or
            on-the-ground verification. Always follow guidance from local authorities and the Relief and
            Rehabilitation Commission (RRC) if the two ever disagree.
          </blockquote>
          <p style={bodyText}>
            Every prediction is a statistical estimate with real, quantified uncertainty — see the Model &amp;
            Validation page for exactly how accurate it has proven to be on data it never saw during training. The
            historical record also has a known gap: satellite-derived flood labels from 2022 onward show a rising
            rate of missing data (see the About page), meaning recent years likely <i>undercount</i> real flood
            events rather than overcount them.
          </p>
          <p style={bodyText}>
            In an actual emergency, don't wait on this page — see{" "}
            <Link to="/contact" style={link}>Contact &amp; Emergency Resources</Link> for who to call.
          </p>

          <Rule />

          <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", lineHeight: 1.7 }}>
            See <Link to="/about" style={link}>About</Link> for full data sources and methodology, or{" "}
            <Link to="/model" style={link}>Model &amp; Validation</Link> for complete performance statistics.
          </p>

          <Rule />

          <h2 id="page-tour" style={h2}>A Tour of Every Page</h2>
          <dl style={{ margin: "1.2rem 0 0" }}>
            {PAGE_TOUR.map((p) => (
              <div key={p.to} style={{ marginBottom: "0.9rem" }}>
                <dt>
                  <Link to={p.to} style={{ ...link, fontWeight: 700, fontSize: "0.95rem" }}>{p.name}</Link>
                </dt>
                <dd style={{ margin: "0.2rem 0 0", fontSize: "0.88rem", color: "var(--color-text-muted)", lineHeight: 1.6, maxWidth: 640 }}>{p.desc}</dd>
              </div>
            ))}
          </dl>
    </DocPage>
  );
}

const h2: React.CSSProperties = { fontSize: "1.35rem", fontWeight: 700, letterSpacing: "-0.01em", color: "var(--color-text)", marginTop: 0 };
const bodyText: React.CSSProperties = { fontSize: "0.94rem", lineHeight: 1.75, color: "var(--color-text-muted)", marginTop: "1rem", maxWidth: 660 };
const bodyLi: React.CSSProperties = { fontSize: "0.94rem", lineHeight: 1.65, color: "var(--color-text-muted)" };
const link: React.CSSProperties = { color: "var(--color-primary)", textDecoration: "none", fontWeight: 600 };

function Rule() {
  return <div style={{ height: 1, background: "var(--color-border)", margin: "2.4rem 0" }} />;
}
