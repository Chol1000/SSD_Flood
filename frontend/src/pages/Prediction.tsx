import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  Card, Row, Col, Select, Switch, Checkbox, Typography, Space, Button, Modal, Tag, Skeleton, Empty, Divider,
  Badge, Alert,
} from "antd";
import { ArrowRightOutlined, ExperimentOutlined } from "@ant-design/icons";
import {
  LineChart, Line, ComposedChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot, ReferenceLine,
} from "recharts";
import { api } from "../api";
import type { CountyListItem, CountyDetail, LiveData, PredictionResult } from "../types";
import MapView, { type MapPoint } from "../components/MapView";
import Gauge from "../components/Gauge";
import Slider from "../components/Slider";
import Sparkline from "../components/Sparkline";
import WeatherPanel from "../components/WeatherPanel";
import ReportGenerator from "../components/ReportGenerator";
import RecommendationView from "../components/RecommendationView";
import { OUTLOOK_ADVICE } from "../riskAdvice";
import Explainer from "../components/Explainer";
import { PageHeader, Caption, RiskTag, useRiskColors, useChartTheme, tooltipValue } from "../ui";

const { Text } = Typography;

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const TODAY = new Date();
const CURRENT_MONTH = TODAY.getMonth() + 1;
const CURRENT_YEAR = TODAY.getFullYear();

const CLIMATE_FIELDS: Array<{ key: string; label: string; unit: string; min: number; max: number; step: number }> = [
  { key: "rainfall_mm", label: "Rainfall", unit: "mm", min: 0, max: 340, step: 1 },
  { key: "soil_moisture_mm", label: "Soil Moisture", unit: "mm", min: 0.5, max: 225, step: 0.5 },
  { key: "max_temperature_celsius", label: "Max Temperature", unit: "°C", min: 26, max: 43.5, step: 0.1 },
  { key: "min_temperature_celsius", label: "Min Temperature", unit: "°C", min: 14.5, max: 28, step: 0.1 },
  { key: "vapor_pressure_deficit_kPa", label: "Vapour Pressure Deficit", unit: "kPa", min: 0.4, max: 5, step: 0.05 },
];
const TERRAIN_FIELDS: Array<{ key: string; label: string; unit: string; min: number; max: number; step: number }> = [
  { key: "wetland_fraction", label: "Wetland Fraction", unit: "", min: 0, max: 0.92, step: 0.01 },
  { key: "elevation_m", label: "Elevation", unit: "m", min: 392, max: 1145, step: 1 },
  { key: "slope_deg", label: "Slope", unit: "°", min: 0.9, max: 8.3, step: 0.1 },
  { key: "ndvi", label: "NDVI", unit: "", min: 0.19, max: 0.85, step: 0.01 },
];

export default function Prediction() {
  const { county: countyParam } = useParams();
  const navigate = useNavigate();
  const selected = countyParam || "Malakal";

  const [counties, setCounties] = useState<CountyListItem[]>([]);
  const [month, setMonth] = useState(CURRENT_MONTH);
  const [useLive, setUseLive] = useState(true);
  const [whatIf, setWhatIf] = useState(false);
  const [showRecommendation, setShowRecommendation] = useState(false);

  const [detail, setDetail] = useState<CountyDetail | null>(null);
  const [live, setLive] = useState<LiveData | null>(null);
  const [manual, setManual] = useState<Record<string, number>>({});
  const [floodPrev, setFloodPrev] = useState(0);
  const [floodPrevDefault, setFloodPrevDefault] = useState(0);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictLoading, setPredictLoading] = useState(false);

  const [rangeFrom, setRangeFrom] = useState(4);
  const [rangeTo, setRangeTo] = useState(12);
  const [seasonalTrend, setSeasonalTrend] = useState<{ month: string; probability: number; m: number }[]>([]);
  const [trendLoading, setTrendLoading] = useState(false);

  const [multistepAll, setMultistepAll] = useState<{ month: number; year: number; probability: number; confidence: number }[]>([]);
  const [multistepLoading, setMultistepLoading] = useState(false);
  const [targetYear, setTargetYear] = useState(CURRENT_YEAR);
  const YEAR_OPTIONS = [CURRENT_YEAR, CURRENT_YEAR + 1, CURRENT_YEAR + 2, CURRENT_YEAR + 3];

  const [sensitivityField, setSensitivityField] = useState<string>("rainfall_mm");
  const [sensitivityData, setSensitivityData] = useState<{ x: number; probability: number }[]>([]);
  const [sensitivityLoading, setSensitivityLoading] = useState(false);

  const [riskHistory, setRiskHistory] = useState<{ t: number; probability: number }[]>([]);

  type ClimatePercentiles = Record<string, { min: number; p10: number; p25: number; median: number; p75: number; p90: number; max: number }>;
  const [climatePercentiles, setClimatePercentiles] = useState<ClimatePercentiles | null>(null);

  useEffect(() => { api.counties().then(setCounties).catch(console.error); }, []);

  // This county's historical (2011-2025) distribution for each live climate
  // input — lets the situation panel say "this month's rainfall is in the
  // 92nd percentile for Malakal" instead of just a min-max normalized bar.
  useEffect(() => {
    let cancelled = false;
    setClimatePercentiles(null);
    api.climatePercentiles(selected).then((d) => { if (!cancelled) setClimatePercentiles(d); }).catch(console.error);
    return () => { cancelled = true; };
  }, [selected]);

  // Switching counties quickly (or a slow/rate-limited live fetch) means an
  // earlier request can resolve AFTER a newer one — every county-keyed fetch
  // below checks `cancelled` before applying its result so a stale response
  // for the PREVIOUS county can never clobber what's on screen now.
  useEffect(() => {
    let cancelled = false;
    // Clear immediately, not just on arrival — otherwise the map (keyed to
    // `selected`) remounts fresh while `detail` still holds the PREVIOUS
    // county's coordinates, and flies to the wrong place under the new
    // county's name (the flownToRef guard then blocks any later correction).
    setDetail(null);
    api.countyDetail(selected).then((d) => {
      if (cancelled) return;
      setDetail(d);
      setManual({
        rainfall_mm: d.defaults.rainfall_mm ?? 80,
        soil_moisture_mm: d.defaults.soil_moisture_mm ?? 25,
        max_temperature_celsius: d.defaults.max_temperature_celsius ?? 35,
        min_temperature_celsius: d.defaults.min_temperature_celsius ?? 21,
        vapor_pressure_deficit_kPa: d.defaults.vapor_pressure_deficit_kPa ?? 2.3,
        wetland_fraction: d.defaults.wetland_fraction ?? 0.1,
        elevation_m: d.defaults.elevation_m ?? 500,
        slope_deg: d.defaults.slope_deg ?? 1.5,
        ndvi: d.defaults.ndvi ?? 0.5,
      });
      setFloodPrev(d.defaults.flood_prev_month ?? 0);
      setFloodPrevDefault(d.defaults.flood_prev_month ?? 0);
    }).catch(console.error);
    return () => { cancelled = true; };
  }, [selected]);

  useEffect(() => {
    if (!useLive) { setLive(null); return; }
    let cancelled = false;
    setLive(null); // don't show the previous county's live data while the new one loads
    api.live(selected).then((d) => { if (!cancelled) setLive(d); }).catch(() => { if (!cancelled) setLive(null); });
    return () => { cancelled = true; };
  }, [selected, useLive]);

  const effectiveInputs = useMemo(() => (useLive && live ? { ...manual, ...live.inputs } : manual), [useLive, live, manual]);

  useEffect(() => {
    if (!detail || Object.keys(effectiveInputs).length === 0) return;
    let cancelled = false;
    setPredictLoading(true);
    const body = {
      county: selected, month,
      rainfall_mm: effectiveInputs.rainfall_mm, soil_moisture_mm: effectiveInputs.soil_moisture_mm,
      max_temperature_celsius: effectiveInputs.max_temperature_celsius, min_temperature_celsius: effectiveInputs.min_temperature_celsius,
      vapor_pressure_deficit_kPa: effectiveInputs.vapor_pressure_deficit_kPa, wetland_fraction: effectiveInputs.wetland_fraction,
      elevation_m: effectiveInputs.elevation_m, slope_deg: effectiveInputs.slope_deg, ndvi: effectiveInputs.ndvi,
      flood_prev_month: floodPrev,
    };
    const t = setTimeout(() => {
      api.predict(body)
        .then((r) => { if (!cancelled) setPrediction(r); })
        .catch(console.error)
        .finally(() => { if (!cancelled) setPredictLoading(false); });
    }, 180);
    return () => { cancelled = true; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, month, floodPrev, effectiveInputs, detail]);

  useEffect(() => {
    if (!detail) return;
    let cancelled = false;
    const months: number[] = [];
    if (rangeFrom <= rangeTo) { for (let m = rangeFrom; m <= rangeTo; m++) months.push(m); }
    else { for (let m = rangeFrom; m <= 12; m++) months.push(m); for (let m = 1; m <= rangeTo; m++) months.push(m); }
    setTrendLoading(true);
    Promise.all(months.map((m) =>
      api.predict({
        county: selected, month: m,
        rainfall_mm: detail.defaults.rainfall_mm ?? 80, soil_moisture_mm: detail.defaults.soil_moisture_mm ?? 25,
        max_temperature_celsius: detail.defaults.max_temperature_celsius ?? 35, min_temperature_celsius: detail.defaults.min_temperature_celsius ?? 21,
        vapor_pressure_deficit_kPa: detail.defaults.vapor_pressure_deficit_kPa ?? 2.3, wetland_fraction: detail.defaults.wetland_fraction ?? 0.1,
        elevation_m: detail.defaults.elevation_m ?? 500, slope_deg: detail.defaults.slope_deg ?? 1.5, ndvi: detail.defaults.ndvi ?? 0.5,
        flood_prev_month: detail.defaults.flood_prev_month ?? 0,
      }).then((r) => ({ month: MONTH_SHORT[m - 1], m, probability: r.probability }))
    )).then((rows) => { if (!cancelled) setSeasonalTrend(rows); })
      .catch(console.error)
      .finally(() => { if (!cancelled) setTrendLoading(false); });
    return () => { cancelled = true; };
  }, [selected, detail, rangeFrom, rangeTo]);

  // Sensitivity simulation — holds every other input at its current effective
  // value and sweeps just `sensitivityField` across its full observed range,
  // so the resulting curve shows how much THIS one variable is driving
  // today's number, not just what the number itself is.
  useEffect(() => {
    if (!detail || Object.keys(effectiveInputs).length === 0) return;
    const fieldDef = CLIMATE_FIELDS.find((f) => f.key === sensitivityField);
    if (!fieldDef) return;
    let cancelled = false;
    setSensitivityLoading(true);
    const STEPS = 16;
    const xs = Array.from({ length: STEPS }, (_, i) => fieldDef.min + (fieldDef.max - fieldDef.min) * (i / (STEPS - 1)));
    Promise.all(xs.map((x) => {
      const body: Record<string, number | string> = {
        county: selected, month,
        rainfall_mm: effectiveInputs.rainfall_mm, soil_moisture_mm: effectiveInputs.soil_moisture_mm,
        max_temperature_celsius: effectiveInputs.max_temperature_celsius, min_temperature_celsius: effectiveInputs.min_temperature_celsius,
        vapor_pressure_deficit_kPa: effectiveInputs.vapor_pressure_deficit_kPa, wetland_fraction: effectiveInputs.wetland_fraction,
        elevation_m: effectiveInputs.elevation_m, slope_deg: effectiveInputs.slope_deg, ndvi: effectiveInputs.ndvi,
        flood_prev_month: floodPrev,
      };
      body[sensitivityField] = x;
      return api.predict(body as any).then((r) => ({ x, probability: r.probability }));
    })).then((rows) => { if (!cancelled) setSensitivityData(rows); })
      .catch(console.error)
      .finally(() => { if (!cancelled) setSensitivityLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, month, floodPrev, effectiveInputs, detail, sensitivityField]);

  // Live flood-risk trend for this county — same server-shared cache the Live
  // Center and County Profile pages read, so the hero gauge can show "how has
  // this actually been moving," not just today's single snapshot.
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api.riskTrend().then((rows) => {
        if (cancelled) return;
        const row = rows.find((r) => r.county === selected);
        setRiskHistory(row ? row.history.map((h) => ({ t: h.t * 1000, probability: h.probability })) : []);
      }).catch(console.error);
    };
    load();
    const t = setInterval(load, 60000);
    return () => { cancelled = true; clearInterval(t); };
  }, [selected]);

  useEffect(() => {
    let cancelled = false;
    setMultistepLoading(true);
    api.outlookMultistep(selected, 48)
      .then((rows) => { if (!cancelled) setMultistepAll(rows); })
      .catch(() => { if (!cancelled) setMultistepAll([]); })
      .finally(() => { if (!cancelled) setMultistepLoading(false); });
    return () => { cancelled = true; };
  }, [selected]);

  // Slice the 48-month rollout down to whichever target year is selected —
  // no refetch needed, this is just a client-side window into the same
  // recursive forecast. Confidence (from the backend) fades further out.
  const multistep = useMemo(
    () => multistepAll
      .filter((r) => r.year === targetYear)
      .map((r) => ({ label: MONTH_SHORT[r.month - 1], probability: r.probability, confidence: r.confidence })),
    [multistepAll, targetYear]
  );

  const liveFieldSet = new Set(live ? Object.entries(live.field_source).filter(([, v]) => v === "live").map(([k]) => k) : []);
  const locatorPoint: MapPoint[] = detail ? [{ county: selected, lat: detail.lat, lon: detail.lon, value: prediction?.probability ?? 0, tier: prediction?.risk_tier ?? "Low" }] : [];

  const colors = useRiskColors();
  const chart = useChartTheme();
  const sensitivityFieldDef = CLIMATE_FIELDS.find((f) => f.key === sensitivityField);

  return (
    <>
      <PageHeader
        eyebrow="Flood risk model"
        title="Prediction"
        subtitle={
          detail
            ? `Historical rate ${(detail.flood_rate * 100).toFixed(1)}% (${detail.flood_events} events) · ranked #${detail.rank} of ${detail.n_counties} nationally`
            : "Loading county…"
        }
        extra={
          <Space wrap>
            <Select
              showSearch
              value={selected}
              onChange={(c) => navigate(`/prediction/${encodeURIComponent(c)}`)}
              options={counties.map((c) => ({ value: c.county, label: c.county }))}
              style={{ minWidth: 180 }}
            />
            <ReportGenerator county={selected} />
            <Link to={`/county/${encodeURIComponent(selected)}`}>
              <Button>Full profile →</Button>
            </Link>
          </Space>
        }
      />

      {/* Locator map + the at-a-glance reading for today. */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={14}>
          <Card styles={{ body: { padding: 0, height: "100%" } }} style={{ overflow: "hidden", height: 400, position: "relative" }}>
            {detail ? (
              <MapView key={selected} points={locatorPoint} selected={selected} onSelect={() => {}} focusZoom={9.5} />
            ) : (
              <div style={{ padding: 24 }}>
                <Skeleton active paragraph={{ rows: 6 }} />
              </div>
            )}
            <Card
              size="small"
              style={{ position: "absolute", top: 12, insetInlineStart: 12, minWidth: 130 }}
              styles={{ body: { padding: "8px 12px" } }}
            >
              <Text strong style={{ fontSize: 14, display: "block" }}>
                {selected}
              </Text>
              {detail && (
                <Text type="secondary" className="tabular" style={{ fontSize: 11 }}>
                  {detail.lat?.toFixed(3)}, {detail.lon?.toFixed(3)}
                </Text>
              )}
            </Card>
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          <Card
            style={{ height: "100%" }}
            title={
              <Text
                type="secondary"
                style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
              >
                {!whatIf ? `Nowcast — ${MONTHS[month - 1]} ${CURRENT_YEAR} (today)` : `What-if — ${MONTHS[month - 1]}`}
              </Text>
            }
            extra={
              prediction && (
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {prediction.model_name}
                </Text>
              )
            }
          >
            {/* Data-freshness line, deliberately NOT on the risk palette. This
                says when the inputs were read, not how dangerous anything is —
                colouring it green/red put a second severity-looking bar right
                above a gauge that already reads CRITICAL in red, so the card
                showed two coloured bars meaning entirely different things. A
                quiet caption keeps the risk colours meaning risk alone.
                The one case that genuinely changes how to read the number —
                no live data at all, so the model is running on historical
                medians — still gets a real warning. */}
            {useLive &&
              (live?.live_data_available ? (
                <div style={{ marginBottom: 14 }}>
                  <Space size={6} align="start">
                    <Badge status="success" />
                    <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.5 }}>
                      Climate inputs live as of <b>{live.last_updated}</b>
                      {live.source === "nasa-power" && " · via NASA POWER (Open-Meteo unavailable)"}
                    </Text>
                  </Space>
                </div>
              ) : (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginBottom: 14 }}
                  message={live?.error ?? "Fetching live climate data…"}
                />
              ))}

            <Space align="center" size={16} wrap>
              <Gauge probability={prediction?.probability ?? 0} tier={prediction?.risk_tier ?? "Low"} size={150} />
              <Space orientation="vertical" size={10}>
                <Space size={8}>
                  <Switch
                    size="small"
                    checked={whatIf}
                    onChange={(on) => {
                      setWhatIf(on);
                      if (!on) {
                        setMonth(CURRENT_MONTH);
                        setUseLive(true);
                      } else {
                        setUseLive(false);
                      }
                    }}
                  />
                  <Text style={{ fontSize: 13 }}>Explore a different month</Text>
                </Space>

                {whatIf ? (
                  <Select
                    size="small"
                    value={month}
                    onChange={setMonth}
                    style={{ width: 140 }}
                    options={MONTHS.map((m, i) => ({ value: i + 1, label: m }))}
                  />
                ) : (
                  <Space size={8}>
                    <Switch size="small" checked={useLive} onChange={setUseLive} />
                    <Text style={{ fontSize: 13 }}>Live climate data</Text>
                  </Space>
                )}
              </Space>
            </Space>

            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 10,
                flexWrap: "wrap",
                padding: "10px 12px",
                background: "var(--color-bg)",
                borderRadius: 6,
                marginTop: 16,
              }}
            >
              <Checkbox checked={floodPrev === 1} onChange={(e) => setFloodPrev(e.target.checked ? 1 : 0)}>
                <Text strong style={{ fontSize: 13 }}>
                  Flooded last month?
                </Text>
              </Checkbox>
              <Tag color={floodPrev === floodPrevDefault ? "default" : "blue"} style={{ marginInlineEnd: 0 }}>
                {floodPrev === floodPrevDefault ? "Historical default" : "Overridden"}
              </Tag>
            </div>

            {!whatIf && riskHistory.length >= 1 && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "8px 12px",
                  background: "var(--color-bg)",
                  borderRadius: 6,
                  marginTop: 10,
                }}
              >
                <Text type="secondary" style={{ fontSize: 11, flexShrink: 0 }}>
                  Live trend:
                </Text>
                <Sparkline values={riskHistory.map((r) => r.probability * 100)} width={110} height={22} />
                <Text type="secondary" style={{ fontSize: 10, marginInlineStart: "auto" }}>
                  {riskHistory.length} reading{riskHistory.length === 1 ? "" : "s"}
                </Text>
              </div>
            )}

            {prediction && (
              <Button
                block
                size="large"
                icon={<ArrowRightOutlined />}
                iconPlacement="end"
                onClick={() => setShowRecommendation(true)}
                style={{
                  marginTop: 14,
                  fontWeight: 700,
                  color: colors[prediction.risk_tier],
                  borderColor: colors[prediction.risk_tier],
                  background: `${colors[prediction.risk_tier]}12`,
                }}
              >
                View Recommendations
              </Button>
            )}
          </Card>
        </Col>
      </Row>

      <Modal
        open={!!prediction && showRecommendation}
        onCancel={() => setShowRecommendation(false)}
        footer={null}
        width={620}
        destroyOnHidden
        title={`Recommendation — ${selected}`}
      >
        {prediction && (
          <RecommendationView
            county={selected}
            tier={prediction.risk_tier}
            probability={prediction.probability}
            historicalTier={counties.find((c) => c.county === selected)?.risk_tier}
            historicalRank={detail?.rank}
            nCounties={detail?.n_counties}
          />
        )}
      </Modal>

      <div style={{ marginBottom: 16 }}>
        <WeatherPanel county={selected} />
      </div>

      {/* Month-ahead outlook + the compiled multi-month rollout. */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        {prediction?.outlook_probability != null && (
          <Col xs={24} lg={9}>
            <Card
              style={{ height: "100%" }}
              title={
                <Text
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    color: "var(--color-primary)",
                  }}
                >
                  Month-Ahead Outlook
                </Text>
              }
            >
              <Space align="baseline" size={10}>
                <span className="tabular" style={{ fontSize: 30, fontWeight: 700 }}>
                  {(prediction.outlook_probability * 100).toFixed(1)}%
                </span>
                {prediction.outlook_risk_tier && <RiskTag tier={prediction.outlook_risk_tier} />}
              </Space>
              <Caption>{prediction.outlook_model_name} — uses only history before this month, no live inputs.</Caption>
              <Divider style={{ margin: "12px 0" }} />
              <Text style={{ fontSize: 13, lineHeight: 1.6 }}>
                <b>Recommended action — next month: </b>
                {prediction.outlook_risk_tier && OUTLOOK_ADVICE[prediction.outlook_risk_tier]}
              </Text>
            </Card>
          </Col>
        )}

        <Col xs={24} lg={prediction?.outlook_probability != null ? 15 : 24}>
          <Card
            style={{ height: "100%" }}
            title={
              <Text
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: "var(--color-primary)",
                }}
              >
                Compiled Outlook — {targetYear}
              </Text>
            }
            extra={
              <Space size={8}>
                {multistepLoading && (
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    updating…
                  </Text>
                )}
                <Select
                  size="small"
                  value={targetYear}
                  onChange={setTargetYear}
                  style={{ width: 130 }}
                  options={YEAR_OPTIONS.map((y) => ({
                    value: y,
                    label: y === CURRENT_YEAR ? `${y} (partial)` : String(y),
                  }))}
                />
              </Space>
            }
          >
            {multistep.length > 0 && (
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={multistep} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke={chart.grid} vertical={false} />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: chart.axis }} stroke={chart.grid} />
                  <YAxis
                    tick={{ fontSize: 10, fill: chart.axis }}
                    stroke={chart.grid}
                    domain={[0, 1]}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  />
                  <Tooltip
                    formatter={(v, _n, p) =>
                      [
                        `${(Number(v) * 100).toFixed(1)}% (confidence ${((p?.payload?.confidence ?? 0) * 100).toFixed(0)}%)`,
                        "Probability",
                      ] as [string, string]
                    }
                    contentStyle={chart.tooltip}
                  />
                  <ReferenceLine
                    y={prediction?.threshold ?? 0.5}
                    stroke={colors.Critical}
                    strokeDasharray="3 3"
                    strokeOpacity={0.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="probability"
                    stroke="#7c3aed"
                    strokeWidth={2.2}
                    strokeOpacity={0.9}
                    dot={(props: { cx?: number; cy?: number; payload?: { confidence: number }; index?: number }) => (
                      <circle
                        key={`dot-${props.index}`}
                        cx={props.cx}
                        cy={props.cy}
                        r={4}
                        fill="#7c3aed"
                        fillOpacity={Math.max(0.25, props.payload?.confidence ?? 1)}
                        stroke={chart.surface}
                        strokeWidth={1}
                      />
                    )}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
            {multistep.length === 0 && !multistepLoading && (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={`No forecast data for ${targetYear}.`} />
            )}
            <Caption>
              Recursive GRU forecast — each month's prediction feeds the next step, so uncertainty compounds further
              out (fainter dots = lower confidence).{" "}
              {targetYear > CURRENT_YEAR + 1 &&
                "Years this far out are a directional trend only, not a precise forecast."}
            </Caption>
            <Explainer>
              A neural network (GRU) trained on 15 years of monthly history produces this outlook by predicting one
              month, then feeding that prediction back in as an input to predict the next month, and so on — like
              forecasting weather a week out gets less certain day by day, each extra month compounds a bit more
              uncertainty. That's why dots fade further into the future: it's the model itself signalling "less sure
              here", not a display glitch.
            </Explainer>
          </Card>
        </Col>
      </Row>

      {/* Seasonal risk pattern across a chosen month window. */}
      <Card
        style={{ marginBottom: 16 }}
        title={
          <Text
            type="secondary"
            style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
          >
            Seasonal Risk Pattern — {selected}
          </Text>
        }
        extra={
          <Space size={8} wrap>
            {trendLoading && (
              <Text type="secondary" style={{ fontSize: 11 }}>
                updating…
              </Text>
            )}
            <Text type="secondary" style={{ fontSize: 12 }}>
              From
            </Text>
            <Select
              size="small"
              value={rangeFrom}
              onChange={setRangeFrom}
              style={{ width: 110 }}
              options={MONTHS.map((m, i) => ({ value: i + 1, label: m }))}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              to
            </Text>
            <Select
              size="small"
              value={rangeTo}
              onChange={setRangeTo}
              style={{ width: 110 }}
              options={MONTHS.map((m, i) => ({ value: i + 1, label: m }))}
            />
          </Space>
        }
      >
        {seasonalTrend.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={seasonalTrend} margin={{ top: 6, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid stroke={chart.grid} vertical={false} />
              <XAxis dataKey="month" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
              <YAxis
                tick={{ fontSize: 11, fill: chart.axis }}
                stroke={chart.grid}
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
              />
              <Tooltip
                formatter={tooltipValue((n) => `${(n * 100).toFixed(1)}%`, "Probability")}
                contentStyle={chart.tooltip}
              />
              <Line type="monotone" dataKey="probability" stroke={chart.accent} strokeWidth={2} dot={{ r: 2 }} />
              {seasonalTrend.find((d) => d.m === month) && (
                <ReferenceDot
                  x={MONTH_SHORT[month - 1]}
                  y={seasonalTrend.find((d) => d.m === month)!.probability}
                  r={5}
                  fill={chart.accent}
                  stroke={chart.surface}
                  strokeWidth={2}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Skeleton active paragraph={{ rows: 3 }} title={false} />
        )}
        <Caption>
          Uses this county's historical climate medians per month (nowcast model) — a seasonal profile, not a live
          forecast.
        </Caption>
      </Card>

      {/* Sensitivity sweep + how today compares to this county's own normal. */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} xl={13}>
          <Card
            style={{ height: "100%" }}
            title={
              <Text
                type="secondary"
                style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
              >
                Sensitivity Simulation
              </Text>
            }
            extra={
              <Space size={8}>
                {sensitivityLoading && (
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    simulating…
                  </Text>
                )}
                <Select
                  size="small"
                  value={sensitivityField}
                  onChange={setSensitivityField}
                  style={{ width: 180 }}
                  options={CLIMATE_FIELDS.map((f) => ({ value: f.key, label: f.label }))}
                />
              </Space>
            }
          >
            {sensitivityData.length > 0 && sensitivityFieldDef ? (
              <ResponsiveContainer width="100%" height={210}>
                <ComposedChart data={sensitivityData} margin={{ top: 6, right: 8, left: -10, bottom: 8 }}>
                  <defs>
                    <linearGradient id="sensGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#7c3aed" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={chart.grid} vertical={false} />
                  <XAxis
                    dataKey="x"
                    tickFormatter={(v) => v.toFixed(sensitivityFieldDef.step < 1 ? 1 : 0)}
                    tick={{ fontSize: 10, fill: chart.axis }}
                    stroke={chart.grid}
                    label={{
                      value: `${sensitivityFieldDef.label} (${sensitivityFieldDef.unit || "unitless"})`,
                      position: "insideBottom",
                      offset: -4,
                      fontSize: 10,
                      fill: chart.axis,
                    }}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                    tick={{ fontSize: 10, fill: chart.axis }}
                    stroke={chart.grid}
                  />
                  <Tooltip
                    formatter={tooltipValue((n) => `${(n * 100).toFixed(1)}%`, "Probability")}
                    labelFormatter={(v) =>
                      `${sensitivityFieldDef.label}: ${Number(v).toFixed(2)} ${sensitivityFieldDef.unit}`
                    }
                    contentStyle={chart.tooltip}
                  />
                  {prediction?.threshold != null && (
                    <ReferenceLine
                      y={prediction.threshold}
                      stroke={colors.Critical}
                      strokeDasharray="3 3"
                      strokeOpacity={0.45}
                    />
                  )}
                  {effectiveInputs[sensitivityField] != null && (
                    <ReferenceLine
                      x={effectiveInputs[sensitivityField]}
                      stroke={chart.accent}
                      strokeDasharray="4 3"
                      label={{ value: "current", position: "top", fontSize: 9, fill: chart.accent }}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="probability"
                    stroke="#7c3aed"
                    strokeWidth={2.2}
                    fill="url(#sensGrad)"
                    dot={false}
                    animationDuration={400}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <Skeleton active paragraph={{ rows: 4 }} title={false} />
            )}
            <Caption>
              Every other input stays fixed at its current value while{" "}
              {sensitivityFieldDef?.label.toLowerCase()} sweeps its full 2011–2025 observed range — isolating how much
              this one variable alone is driving today's number.
            </Caption>
            <Explainer>
              This is a "what if" experiment, not a forecast: it re-runs the model dozens of times, changing only the
              variable you pick while freezing everything else exactly as it is right now for {selected} today. The
              dashed vertical line marks where conditions actually are today; the dashed horizontal line is the alert
              threshold. A steep curve means the prediction is very sensitive to that variable; a flat curve means it
              barely matters here.
            </Explainer>
          </Card>
        </Col>

        <Col xs={24} xl={11}>
          <Card
            style={{ height: "100%" }}
            title={
              <Text
                type="secondary"
                style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
              >
                Situation vs. Historical Normal — {selected}
              </Text>
            }
          >
            {climatePercentiles ? (
              CLIMATE_FIELDS.map((f) => (
                <ClimatePercentileBar
                  key={f.key}
                  label={f.label}
                  unit={f.unit ? ` ${f.unit}` : ""}
                  value={effectiveInputs[f.key]}
                  stats={climatePercentiles[f.key]}
                  decimals={f.step < 1 ? 2 : 1}
                />
              ))
            ) : (
              <Skeleton active paragraph={{ rows: 6 }} title={false} />
            )}
            <Caption>
              Green band = the middle 80% of {selected}'s own 2011–2025 history for that variable (10th–90th
              percentile). The dot is {useLive ? "today's live reading" : "your manual input"} — sitting outside the
              band means genuinely unusual for this specific county, not just different from a national average.
            </Caption>
            <Explainer>
              Each bar is this county's own 15-year history for one variable, stretched from its lowest ever recorded
              value (left) to its highest (right). The shaded middle section is where readings normally fall — about 8
              months out of 10, historically. The dot marks where things stand right now. A dot near the far left or
              right edge, outside the shaded band, means conditions are genuinely unusual for {selected} specifically.
            </Explainer>
          </Card>
        </Col>
      </Row>

      {/* Model inputs. */}
      <Card
        title={
          <Space>
            <ExperimentOutlined />
            <Text
              type="secondary"
              style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
            >
              Model Inputs {predictLoading && "· updating"}
            </Text>
          </Space>
        }
      >
        <Row gutter={[32, 16]}>
          <Col xs={24} lg={12}>
            <Text strong style={{ fontSize: 13, display: "block", marginBottom: 4 }}>
              Climate
            </Text>
            {useLive && (
              <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 14, lineHeight: 1.5 }}>
                These come from live climate data for {selected} and can't be adjusted here — turn off "Live climate
                data" above to set them manually for a what-if scenario instead.
              </Text>
            )}
            {CLIMATE_FIELDS.map((f) => (
              <Slider
                key={f.key}
                label={f.label}
                unit={f.unit}
                min={f.min}
                max={f.max}
                step={f.step}
                value={effectiveInputs[f.key] ?? f.min}
                lockBadge={useLive ? (liveFieldSet.has(f.key) ? "live" : "fallback") : undefined}
                disabled={useLive}
                onChange={(v) => setManual((m) => ({ ...m, [f.key]: v }))}
              />
            ))}
          </Col>
          <Col xs={24} lg={12}>
            <Text strong style={{ fontSize: 13, display: "block", marginBottom: 14 }}>
              Terrain (static)
            </Text>
            {TERRAIN_FIELDS.map((f) => (
              <Slider
                key={f.key}
                label={f.label}
                unit={f.unit}
                min={f.min}
                max={f.max}
                step={f.step}
                value={effectiveInputs[f.key] ?? f.min}
                onChange={(v) => setManual((m) => ({ ...m, [f.key]: v }))}
              />
            ))}
          </Col>
        </Row>
      </Card>
    </>
  );
}

const QUANTILE_POINTS: Array<{ q: number; key: "min" | "p10" | "p25" | "median" | "p75" | "p90" | "max" }> = [
  { q: 0, key: "min" }, { q: 10, key: "p10" }, { q: 25, key: "p25" },
  { q: 50, key: "median" }, { q: 75, key: "p75" }, { q: 90, key: "p90" }, { q: 100, key: "max" },
];

/** Where `value` falls in the county's own historical distribution, as a
 * percentile 0-100, via linear interpolation between the known quantile
 * points (min/p10/p25/median/p75/p90/max) — not a normal-distribution
 * assumption, just a piecewise-linear read of the real empirical spread. */
function percentileRank(value: number, stats: Record<string, number>): number {
  const pts = QUANTILE_POINTS.map((p) => ({ q: p.q, v: stats[p.key] }));
  if (value <= pts[0].v) return 0;
  if (value >= pts[pts.length - 1].v) return 100;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    if (value >= a.v && value <= b.v) {
      const span = b.v - a.v;
      const frac = span === 0 ? 0 : (value - a.v) / span;
      return Math.round(a.q + frac * (b.q - a.q));
    }
  }
  return 50;
}

function ClimatePercentileBar({
  label,
  unit,
  value,
  stats,
  decimals,
}: {
  label: string;
  unit: string;
  value: number | undefined;
  stats: { min: number; p10: number; p25: number; median: number; p75: number; p90: number; max: number } | undefined;
  decimals: number;
}) {
  const colors = useRiskColors();

  if (!stats || value == null || !Number.isFinite(value)) {
    return (
      <Text type="secondary" style={{ fontSize: 13, display: "block", padding: "6px 0" }}>
        {label}: no data
      </Text>
    );
  }

  const { min, p10, median, p90, max } = stats;
  const range = max - min || 1;
  const pctOf = (v: number) => Math.max(0, Math.min(100, ((v - min) / range) * 100));
  const rank = percentileRank(value, stats);
  const status =
    rank < 10 ? "unusually low"
      : rank > 90 ? "unusually high"
        : rank > 75 ? "above normal"
          : rank < 25 ? "below normal"
            : "normal";
  const markerColor =
    rank < 10 || rank > 90 ? colors.Critical : rank > 75 || rank < 25 ? colors.Moderate : colors.Low;

  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 8,
          flexWrap: "wrap",
          marginBottom: 6,
        }}
      >
        <Text strong style={{ fontSize: 13 }}>
          {label}
        </Text>
        <Text style={{ fontSize: 12, textAlign: "right" }}>
          <b className="tabular">
            {value.toFixed(decimals)}
            {unit}
          </b>{" "}
          <span className="tabular" style={{ color: markerColor, fontWeight: 700 }}>
            {rank}th pct.
          </span>{" "}
          <Text type="secondary" style={{ fontSize: 12 }}>
            ({status})
          </Text>
        </Text>
      </div>

      <div style={{ position: "relative", height: 12, background: "var(--color-border)", borderRadius: 6 }}>
        {/* 10th-90th percentile band — this county's own "normal" range. */}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `${pctOf(p10)}%`,
            width: `${Math.max(0, pctOf(p90) - pctOf(p10))}%`,
            background: `${colors.Low}33`,
            borderRadius: 6,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: -2,
            bottom: -2,
            left: `${pctOf(median)}%`,
            width: 2,
            background: "var(--color-text-muted)",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: `${pctOf(value)}%`,
            width: 13,
            height: 13,
            borderRadius: "50%",
            background: markerColor,
            border: "2px solid var(--color-surface)",
            transform: "translate(-50%, -50%)",
            boxShadow: "0 1px 3px rgba(16,24,40,0.35)",
          }}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        <Text type="secondary" className="tabular" style={{ fontSize: 10 }}>
          {min.toFixed(decimals)}
        </Text>
        <Text type="secondary" className="tabular" style={{ fontSize: 10 }}>
          med. {median.toFixed(decimals)}
        </Text>
        <Text type="secondary" className="tabular" style={{ fontSize: 10 }}>
          {max.toFixed(decimals)}
        </Text>
      </div>
    </div>
  );
}
