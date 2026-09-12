export type RiskTier = "Low" | "Moderate" | "High" | "Critical";

export interface CountyListItem {
  county: string;
  lat: number;
  lon: number;
  flood_rate: number;
  flood_events: number;
  risk_tier: RiskTier;
}

export interface CountyDetail {
  county: string;
  lat: number;
  lon: number;
  defaults: Record<string, number>;
  flood_rate: number;
  flood_events: number;
  rank: number;
  n_counties: number;
}

export interface LiveData {
  inputs: Record<string, number>;
  field_source: Record<string, "live" | "historical_fallback">;
  live_data_available: boolean;
  last_updated: string | null;
  source: "open-meteo" | "nasa-power" | null;
  error: string | null;
}

export interface WeatherForecastPoint {
  time: string;
  dt: number;
  day_key: string;
  temp_c: number;
  rain_probability: number;
  rain_mm_3h: number;
  description: string;
  icon: string;
}

export interface DailyWeatherSummary {
  day_key: string;
  label: string;
  temp_min: number;
  temp_max: number;
  icon: string;
  description: string;
  rain_probability_max: number;
}

export interface LiveWeather {
  temp_c: number;
  feels_like_c: number;
  humidity_pct: number;
  pressure_hpa: number;
  wind_speed_ms: number;
  description: string;
  icon: string;
  observed_at: number;
  sunrise: number;
  sunset: number;
  timezone_offset_s: number;
  forecast: WeatherForecastPoint[];
  daily: DailyWeatherSummary[];
  source: string;
}

export interface PredictionInputs {
  county: string;
  month: number;
  rainfall_mm: number;
  soil_moisture_mm: number;
  max_temperature_celsius: number;
  min_temperature_celsius: number;
  vapor_pressure_deficit_kPa: number;
  wetland_fraction: number;
  elevation_m: number;
  slope_deg: number;
  ndvi: number;
  flood_prev_month: number;
}

export interface PredictionResult {
  probability: number;
  risk_tier: RiskTier;
  threshold: number;
  above_threshold: boolean;
  outlook_probability: number | null;
  outlook_risk_tier: RiskTier | null;
  model_name: string;
  outlook_model_name: string;
}

export interface ScanResult {
  county: string;
  lat: number;
  lon: number;
  probability: number;
  risk_tier: RiskTier;
}

export interface AblationRow {
  "Feature Set": string; "N Features": number;
  "CV AUC": number; "CV AUC std": number; "CV F1": number;
  "Test AUC": number; "Test F1": number; "Test Precision": number; "Test Recall": number;
}

export interface OnsetRow {
  threshold: number; onset_detected: number; onset_total: number; fp: number;
  precision: number; recall: number; f1: number;
}

export interface SignificanceRow {
  comparison: string; auc_LR?: number; auc_other?: number; delta_auc?: number;
  z_stat?: number; b_LR_wins?: number; c_other_wins?: number; chi2_stat?: number;
  p_value: number; significant_05: number;
}

export interface ModelInfo {
  best_model_name: string;
  outlook_model_name: string;
  threshold: number;
  features: string[];
  cv_metrics: Record<string, { auc_roc_mean: number; auc_roc_std: number; f1_mean: number; f1_std: number }>;
  test_metrics: Record<string, {
    auc_roc: number; f1: number; precision: number; recall: number;
    tp: number; fp: number; fn: number; tn: number;
  }>;
  feature_importance: Record<string, number>;
  model_selection_criterion: string;
  nowcast_vs_outlook?: string;
  excluded_features: string[];
  exclusion_reason: string;
  feature_engineering: Record<string, string>;
  methodology: Record<string, string>;
  ablation: AblationRow[];
  persistence_baseline: {
    description: string; auc_roc: number; ap: number; f1: number; precision: number; recall: number;
    tp: number; fp: number; fn: number; tn: number; note: string;
  };
  onset_analysis: OnsetRow[];
  significance_tests: { method: string; delong: SignificanceRow[]; mcnemar: SignificanceRow[] };
}

export interface WatchlistEntry {
  county: string; flood_rate: number; flood_events: number; risk_tier: RiskTier;
}

export interface OverviewData {
  total_flood_events: number;
  national_mean_rate: number;
  n_counties: number;
  risk_tier_counts: Record<RiskTier, number>;
  highest_risk_county: string;
  watchlist: WatchlistEntry[];
  calendar: { year: number; month: number; flood: number }[];
  period: { year_min: number; year_max: number };
}
