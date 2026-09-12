import type {
  CountyListItem, CountyDetail, LiveData, PredictionInputs, PredictionResult, ScanResult, ModelInfo, OverviewData,
  RiskTier, LiveWeather,
} from "./types";

// In dev, Vite proxies /api to the FastAPI backend (see vite.config.ts).
// In production, FastAPI serves the built frontend itself, so relative /api works too.
const BASE = "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  health: () => getJSON<{ status: string }>("/health"),
  counties: () => getJSON<CountyListItem[]>("/counties"),
  countyDetail: (county: string) => getJSON<CountyDetail>(`/county/${encodeURIComponent(county)}`),
  climatePercentiles: (county: string) =>
    getJSON<Record<string, { min: number; p10: number; p25: number; median: number; p75: number; p90: number; max: number }>>(
      `/county/${encodeURIComponent(county)}/climate-percentiles`
    ),
  live: (county: string) => getJSON<LiveData>(`/live/${encodeURIComponent(county)}`),
  weather: (county: string) => getJSON<LiveWeather>(`/weather/${encodeURIComponent(county)}`),
  weatherGrid: () => getJSON<Array<{ county: string; lat: number; lon: number; temp_c: number; description: string; icon: string; humidity_pct: number; wind_speed_ms: number; observed_at: number }>>("/weather-grid"),
  weatherTicker: () => getJSON<Array<{
    county: string; lat: number; lon: number;
    history: Array<{ t: number; temp_c: number; humidity_pct: number; wind_speed_ms: number; description: string; icon: string; observed_at: number }>;
  }>>("/weather-ticker"),
  riskTrend: () => getJSON<Array<{ county: string; history: Array<{ t: number; probability: number; tier: RiskTier }> }>>("/risk-trend"),
  modelInfo: () => getJSON<ModelInfo>("/model-info"),
  overview: () => getJSON<OverviewData>("/overview"),
  outlookMultistep: (county: string, months: number) =>
    getJSON<{ month: number; year: number; probability: number; risk_tier: RiskTier; confidence: number }[]>(
      `/outlook/${encodeURIComponent(county)}/multistep?months=${months}`
    ),
  reportUrl: (county: string, opts?: { month?: number; year?: number; rangeFrom?: number; rangeTo?: number }) => {
    const qs = new URLSearchParams();
    if (opts?.month) qs.set("month", String(opts.month));
    if (opts?.year) qs.set("year", String(opts.year));
    if (opts?.rangeFrom) qs.set("range_from", String(opts.rangeFrom));
    if (opts?.rangeTo) qs.set("range_to", String(opts.rangeTo));
    const q = qs.toString();
    return `${BASE}/report/${encodeURIComponent(county)}${q ? `?${q}` : ""}`;
  },
  nationalReportUrl: () => `${BASE}/report-national`,

  predict: async (body: PredictionInputs): Promise<PredictionResult> => {
    const res = await fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`predict -> ${res.status}`);
    return res.json();
  },

  scan: (params: {
    month: number; use_live: boolean; rainfall_mm?: number; soil_moisture_mm?: number;
    max_temperature_celsius?: number; min_temperature_celsius?: number;
    vapor_pressure_deficit_kPa?: number; flood_prev_month?: number;
  }) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)])
    );
    return getJSON<ScanResult[]>(`/scan?${qs.toString()}`);
  },

  historical: (counties: string[], yearMin: number, yearMax: number, months: number[]) => {
    const qs = new URLSearchParams({
      counties: counties.join(","),
      year_min: String(yearMin),
      year_max: String(yearMax),
      months: months.join(","),
    });
    return getJSON<{ records: any[]; national: { year: number; flood: number }[] }>(`/historical?${qs.toString()}`);
  },
};
