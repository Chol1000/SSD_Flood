import type { RiskTier } from "./types";

export const NOWCAST_ADVICE: Record<RiskTier, string> = {
  Low: "No immediate action required. Maintain routine monitoring.",
  Moderate: "Alert county officials. Pre-position emergency supplies and review evacuation routes.",
  High: "Issue a public warning. Activate emergency response teams and prepare for evacuations.",
  Critical: "Immediate action. Initiate evacuations and notify national emergency management.",
};

export const OUTLOOK_ADVICE: Record<RiskTier, string> = {
  Low: "No elevated risk expected next month based on recent history — routine monitoring is sufficient.",
  Moderate: "Risk is above baseline for next month. Begin reviewing preparedness plans and supply positions.",
  High: "Historical pattern points to elevated risk next month. Start preparedness planning and resource staging now.",
  Critical: "Strong signal of flood risk next month. Begin pre-positioning resources and briefing response teams ahead of time.",
};

/** Concrete action checklist per tier — deliberately more than one sentence,
 * grounded in what a county-level responder in South Sudan can actually do
 * (evacuation, RRC/commissioner notification, boat staging), not generic
 * disaster-response boilerplate. */
export const ACTION_CHECKLIST: Record<RiskTier, string[]> = {
  Critical: [
    "Initiate evacuation of low-lying households and livestock to designated high ground.",
    "Notify the County Commissioner's office and the Relief and Rehabilitation Commission (RRC) immediately.",
    "Activate emergency shelters and pre-position boats or canoes where roads are likely to become impassable.",
    "Establish a 24-hour watch on river and wetland levels, and on any nearby dyke or levee sections.",
  ],
  High: [
    "Issue a public warning through local radio and community/chief networks.",
    "Pre-position emergency response teams and relief supplies at accessible staging points.",
    "Begin phased evacuation planning for the most exposed low-lying settlements.",
    "Inspect and reinforce any local flood-control earthworks (dykes, levees, sandbag lines) before onset.",
  ],
  Moderate: [
    "Alert county officials and community leaders to elevated conditions.",
    "Review evacuation routes and confirm emergency contact chains are current.",
    "Pre-position basic emergency supplies — water purification, first aid, shelter material.",
  ],
  Low: [
    "Maintain routine monitoring of rainfall and river levels.",
    "No special action required beyond standard seasonal preparedness.",
  ],
};

/** Known flood-control infrastructure context for specific counties. Kept
 * narrow and hedged — only counties with a documented dyke/levee history,
 * not asserted uniformly across every flood-prone county. The Bor–Baidit
 * dyke system (Jonglei) has breached in past flood seasons (notably
 * 2020–2022) and is explicitly not a guaranteed safeguard. */
export const COUNTY_INFRASTRUCTURE_NOTES: Record<string, string> = {
  "Bor South": "Bor South is partly protected by the Bor–Baidit dyke system along the Nile. This earthen levee has breached in past flood seasons — treat it as a risk-reducing measure, not a guarantee, and inspect for weak points before water levels rise.",
  "Duk": "Duk sits in the same Nile/Sudd floodplain corridor as the Bor dyke system. Areas without direct levee protection remain exposed when upstream sections are overtopped or breached.",
  "Twic East": "Twic East lies along the same Nile floodplain corridor as Bor South and Duk. Flood-control earthworks in the area are limited and have historically been overwhelmed in major flood years (e.g. 2020–2022).",
};

export interface RecommendationContext {
  county: string;
  tier: RiskTier;
  probability: number; // 0-1
  historicalTier?: RiskTier;
  historicalRank?: number; // 1 = highest historical rate
  nCounties?: number;
}

/** Composes a full, county-aware recommendation — headline + probability
 * framing + historical context + infrastructure note (where known) + a
 * concrete action checklist. Pure function of live inputs, so it regenerates
 * fresh every time tier/probability change (a live nowcast refresh, or the
 * user selecting a different county) rather than reusing a cached sentence. */
export function buildRecommendation(ctx: RecommendationContext) {
  const pct = Math.round(ctx.probability * 100);
  const sentences: string[] = [NOWCAST_ADVICE[ctx.tier]];

  if (ctx.tier === "Critical" && pct >= 90) {
    sentences.push(`At ${pct}%, this is among the highest live readings the system produces — treat this as an active, unfolding event, not an early warning.`);
  } else if (ctx.tier === "Critical") {
    sentences.push(`Live flood probability is ${pct}% under current climate conditions.`);
  }

  if (ctx.historicalTier && ctx.historicalTier !== ctx.tier) {
    const rank = ctx.historicalRank && ctx.nCounties ? ` (ranked #${ctx.historicalRank} of ${ctx.nCounties} nationally for historical flood rate)` : "";
    sentences.push(`This reading is ${rankDirection(ctx.historicalTier, ctx.tier)} the county's 2011–2025 historical baseline of ${ctx.historicalTier}${rank}.`);
  }

  const infrastructure = COUNTY_INFRASTRUCTURE_NOTES[ctx.county];

  return {
    summary: sentences.join(" "),
    actions: ACTION_CHECKLIST[ctx.tier],
    infrastructure,
  };
}

function rankDirection(hist: RiskTier, live: RiskTier): string {
  const order: Record<RiskTier, number> = { Low: 0, Moderate: 1, High: 2, Critical: 3 };
  return order[live] > order[hist] ? "above" : "below";
}
