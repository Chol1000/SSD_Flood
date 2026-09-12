import { Link } from "react-router-dom";
import DocPage from "../components/DocPage";

interface Term { term: string; def: string }

const CLIMATE_TERMS: Term[] = [
  { term: "Nowcast", def: "This month's flood-risk estimate, computed from live climate data as it comes in today — not a forecast of the future, a read of the present." },
  { term: "Outlook", def: "Next month's flood-risk estimate, computed purely from historical monthly patterns with no live data at all — a statistical trend, not a weather forecast." },
  { term: "Rainfall (MTD)", def: "\"Month-to-date\" rainfall — the running total for the current calendar month so far, updated daily as more rain falls." },
  { term: "Soil Moisture", def: "Water content held in the top soil layer, in millimetres of equivalent depth. High soil moisture means the ground is already saturated and can't absorb much more rain before it runs off or floods." },
  { term: "Vapour Pressure Deficit (VPD)", def: "The gap between how much moisture the air is actually holding and how much it could hold at that temperature. Low VPD means the air is nearly saturated — a classic pre-flood atmospheric signature in this region." },
  { term: "Percentile", def: "Where a value ranks against history. \"90th percentile rainfall\" means only 10% of recorded months for that county had more rain than this one — a way of saying \"unusually high for here,\" not just high in absolute terms." },
  { term: "Historical Median", def: "The middle value of a county's 2011–2025 record for a given variable — used as the fallback whenever a live reading isn't available, always labeled as such rather than presented as current." },
  { term: "Antecedent Flood State", def: "Shorthand for \"did this county flood last month\" — one of the model's most influential inputs, since flooding often persists or recurs across consecutive months." },
];

const MODEL_TERMS: Term[] = [
  { term: "Risk Tier", def: "The four-band grouping of a probability into Low (under 25%), Moderate (25–50%), High (50–75%), or Critical (75%+) — see the How to Use page for what each means practically." },
  { term: "AUC-ROC", def: "\"Area under the ROC curve\" — a single score from 0.5 (no better than a coin flip) to 1.0 (perfect) summarizing how well the model ranks flood months above non-flood months, across every possible decision threshold at once." },
  { term: "Precision", def: "Of every flood alert the model raised, what fraction were real floods. High precision means few false alarms." },
  { term: "Recall", def: "Of every real flood that happened, what fraction the model actually caught. High recall means few missed events." },
  { term: "F1 Score", def: "A single number balancing precision and recall — useful when you care about both false alarms and missed events, not just one." },
  { term: "Confusion Matrix", def: "A 2×2 breakdown of every prediction into True Positive (correctly flagged), True Negative (correctly cleared), False Positive (false alarm), and False Negative (missed event)." },
  { term: "Decision Threshold", def: "The probability cutoff above which a prediction counts as \"flood expected.\" Set deliberately, not just 50/50, based on which error type (false alarms vs. missed events) matters more for this use case." },
  { term: "Cross-Validation (CV)", def: "Testing a model on several different train/test splits of the same historical data during development, to check its score isn't just luck from one particular split." },
  { term: "Held-Out Test Set", def: "Data (2024–2025) deliberately kept completely separate from training and cross-validation — the model never saw it until final evaluation, the closest thing to an honest real-world test available." },
  { term: "Ablation Study", def: "Removing one group of input features at a time and re-training, to measure how much each group actually contributes to the final score." },
  { term: "Persistence Baseline", def: "The simplest possible forecast — \"whatever happened last month will happen again\" — used as a floor that any real model must beat to prove it adds value." },
  { term: "DeLong's Test / McNemar's Test", def: "Statistical tests checking whether one model's advantage over another is real (\"statistically significant\") or could plausibly be random chance, given the test-set size." },
  { term: "p-value", def: "The probability that a result this strong could occur by chance alone. Below 0.05 is the conventional cutoff for calling a difference \"real\" rather than noise." },
  { term: "Feature Importance", def: "How much a given input variable moves the model's predictions on average — a measure of reliance, not proof that the variable causes flooding." },
  { term: "GRU (Gated Recurrent Unit)", def: "A type of neural network suited to sequences — used here for the month-ahead Outlook, since it can learn from a county's chronological history of past months." },
  { term: "Sensitivity Simulation", def: "A \"what if\" experiment on the Prediction page: re-running the model while sweeping one input variable across its full range and holding everything else fixed, to see how much that one variable alone is driving today's number." },
  { term: "Co-occurrence (Pearson Correlation)", def: "On the Historical page, a measure from -1 to +1 of whether two counties tend to flood in the same months — positive means they flood together, near zero means no relationship." },
];

const TOC = [
  { id: "climate", label: "Climate & Weather Terms" },
  { id: "model", label: "Model & Statistics Terms" },
];

export default function Glossary() {
  return (
    <DocPage eyebrow="Reference" title="Glossary" toc={TOC}>
          <p style={{ fontFamily: "inherit", fontSize: "1.1rem", lineHeight: 1.65, color: "var(--color-text)", margin: "1.4rem 0 0" }}>
            Plain-language definitions for every technical term used across this system — so a chart or statistic
            never requires a search engine to understand. See also the small <b>"ⓘ What does this mean?"</b> toggle
            under individual charts for context specific to that exact visualization.
          </p>

          <Rule />

          <h2 id="climate" style={h2}>Climate &amp; Weather Terms</h2>
          <dl style={{ margin: "1.2rem 0 0" }}>
            {CLIMATE_TERMS.map((t) => (
              <div key={t.term} style={{ marginBottom: "1.1rem" }}>
                <dt style={{ fontWeight: 700, fontSize: "0.98rem" }}>{t.term}</dt>
                <dd style={{ margin: "0.25rem 0 0", fontSize: "0.9rem", color: "var(--color-text-muted)", lineHeight: 1.65, maxWidth: 660 }}>{t.def}</dd>
              </div>
            ))}
          </dl>

          <Rule />

          <h2 id="model" style={h2}>Model &amp; Statistics Terms</h2>
          <dl style={{ margin: "1.2rem 0 0" }}>
            {MODEL_TERMS.map((t) => (
              <div key={t.term} style={{ marginBottom: "1.1rem" }}>
                <dt style={{ fontWeight: 700, fontSize: "0.98rem" }}>{t.term}</dt>
                <dd style={{ margin: "0.25rem 0 0", fontSize: "0.9rem", color: "var(--color-text-muted)", lineHeight: 1.65, maxWidth: 660 }}>{t.def}</dd>
              </div>
            ))}
          </dl>

          <Rule />

          <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)", lineHeight: 1.7 }}>
            See <Link to="/model" style={{ color: "var(--color-primary)" }}>Model &amp; Validation</Link> for these terms
            applied to the actual deployed model's numbers, or{" "}
            <Link to="/guide" style={{ color: "var(--color-primary)" }}>How to Use This System</Link> for the bigger
            picture.
          </p>
    </DocPage>
  );
}

const h2: React.CSSProperties = { fontSize: "1.4rem", fontWeight: 700, letterSpacing: "-0.01em", color: "var(--color-text)", marginTop: 0 };

function Rule() {
  return <div style={{ height: 1, background: "var(--color-border)", margin: "2.4rem 0" }} />;
}
