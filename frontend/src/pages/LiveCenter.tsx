import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  Card, Row, Col, Input, Select, Segmented, Button, Typography, Space, Badge, Tag, Alert, Empty, Skeleton, Statistic,
} from "antd";
import { SearchOutlined, ExpandOutlined, CompressOutlined } from "@ant-design/icons";
import {
  ComposedChart, Area, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from "recharts";
import { api } from "../api";
import type { LiveWeather, RiskTier, CountyDetail, LiveData } from "../types";
import AnimatedNumber from "../components/AnimatedNumber";
import WeatherIcon from "../components/WeatherIcon";
import Sparkline from "../components/Sparkline";
import { PageHeader, Caption, RiskTag, useRiskColors, useChartTheme, tooltipValue } from "../ui";

const { Text, Title } = Typography;

const OW_TICKER_REFRESH_SECONDS = 90;
const OW_REFRESH_SECONDS = 90;
const RISK_REFRESH_SECONDS = 60;
const RAIN_TICKS = [0, 0.25, 0.5, 0.75, 1];

/* CSS custom property rather than a literal hex, so tier colors here follow
   the theme toggle the same way ui.tsx's palette does. */
function riskVar(tier: RiskTier): string {
  return `var(--risk-${tier.toLowerCase()})`;
}

const RISK_ORDER: RiskTier[] = ["Critical", "High", "Moderate", "Low"];
type Mode = "weather" | "risk";

type Station = { key: string; name: string; lat: number; lon: number };
type Reading = { t: number; temp: number; wind: number; icon: string; description: string; humidity: number };
type SortMode = "name" | "value_desc" | "value_asc";

function stationKey(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function localHM(unixS: number, tzOffsetS: number): string {
  const d = new Date((unixS + tzOffsetS) * 1000);
  return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
}

type FeedStatus = "loading" | "ok" | "error";

/** Reads the Live Updates feed from our own backend (/api/weather-ticker),
 * which does the OpenWeatherMap fetching server-side, once per interval, for
 * all 79 counties combined. Deliberately OpenWeatherMap rather than
 * Open-Meteo: this app's OpenWeatherMap quota is tied to its own API key, not
 * shared across every anonymous caller on whatever network the server
 * happens to run on — the thing that made the earlier Open-Meteo-backed
 * ticker vulnerable to rate-limiting. */
function useLiveReadings() {
  const [history, setHistory] = useState<Record<string, Reading[]>>({});
  const [status, setStatus] = useState<FeedStatus>("loading");

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api.weatherTicker().then((rows) => {
        if (cancelled) return;
        const next: Record<string, Reading[]> = {};
        for (const row of rows) {
          next[stationKey(row.county)] = row.history.map((h) => ({
            t: h.t * 1000, temp: h.temp_c, wind: h.wind_speed_ms, icon: h.icon, description: h.description, humidity: h.humidity_pct,
          }));
        }
        setHistory(next);
        setStatus("ok");
      }).catch(() => { if (!cancelled) setStatus("error"); });
    };
    load();
    const t = setInterval(load, OW_TICKER_REFRESH_SECONDS * 1000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  return { history, status };
}

type RiskReading = { t: number; probability: number; tier: RiskTier };

/** Reads the flood-risk trend from our own backend (/api/risk-trend), which
 * computes it server-side on a timer and shares it across every client — so
 * the trend is already flowing, hours deep, the moment this page opens,
 * rather than resetting to empty per browser session. */
function useRiskHistory() {
  const [history, setHistory] = useState<Record<string, RiskReading[]>>({});

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api.riskTrend().then((rows) => {
        if (cancelled) return;
        const next: Record<string, RiskReading[]> = {};
        for (const row of rows) {
          next[stationKey(row.county)] = row.history.map((h) => ({ t: h.t * 1000, probability: h.probability, tier: h.tier }));
        }
        setHistory(next);
      }).catch(console.error);
    };
    load();
    const t = setInterval(load, RISK_REFRESH_SECONDS * 1000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  return history;
}


export default function LiveCenter() {
  const [mode, setMode] = useState<Mode>("weather");
  const [stations, setStations] = useState<Station[]>([]);
  const [omCountdown, setOmCountdown] = useState(OW_TICKER_REFRESH_SECONDS);
  const [riskCountdown, setRiskCountdown] = useState(RISK_REFRESH_SECONDS);
  const [search, setSearch] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("name");
  const [selected, setSelected] = useState("Malakal");
  const [expanded, setExpanded] = useState(false);

  const [detail, setDetail] = useState<LiveWeather | null>(null);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);
  const [owCountdown, setOwCountdown] = useState(OW_REFRESH_SECONDS);
  const [unavailable, setUnavailable] = useState(false);
  const [countyDetail, setCountyDetail] = useState<CountyDetail | null>(null);
  const [liveInputs, setLiveInputs] = useState<LiveData | null>(null);

  useEffect(() => {
    api.counties().then((cs) => {
      setStations(cs.map((c) => ({ key: stationKey(c.county), name: c.county, lat: c.lat, lon: c.lon })));
    }).catch(console.error);
  }, []);

  const { history, status: omStatus } = useLiveReadings();
  const riskHistory = useRiskHistory();

  useEffect(() => {
    const t = setInterval(() => {
      setOmCountdown((c) => (c <= 1 ? OW_TICKER_REFRESH_SECONDS : c - 1));
      setRiskCountdown((c) => (c <= 1 ? RISK_REFRESH_SECONDS : c - 1));
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // OpenWeatherMap detail for whichever county is selected — refreshes independently.
  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setSelectedDay(null);
    const load = (first: boolean) => {
      api.weather(selected).then((w) => {
        if (cancelled) return;
        setDetail(w);
        if (first) setSelectedDay(w.daily[0]?.day_key ?? null);
      }).catch(() => { if (!cancelled) setUnavailable(true); });
    };
    load(true);
    const t = setInterval(() => load(false), OW_REFRESH_SECONDS * 1000);
    return () => { cancelled = true; clearInterval(t); };
  }, [selected]);

  useEffect(() => {
    const t = setInterval(() => setOwCountdown((c) => (c <= 1 ? OW_REFRESH_SECONDS : c - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  // Flood-risk supporting context for whichever county is selected — historical
  // baseline/rank plus the live climate inputs actually feeding the nowcast,
  // so the risk panel can explain *why* the number is what it is, not just show it.
  useEffect(() => {
    let cancelled = false;
    setCountyDetail(null);
    setLiveInputs(null);
    api.countyDetail(selected).then((d) => { if (!cancelled) setCountyDetail(d); }).catch(console.error);
    api.live(selected).then((d) => { if (!cancelled) setLiveInputs(d); }).catch(console.error);
    return () => { cancelled = true; };
  }, [selected]);

  const visible = useMemo(() => {
    let list = stations.filter((s) => s.name.toLowerCase().includes(search.toLowerCase()));
    list = [...list].sort((a, b) => {
      if (sortMode === "name") return a.name.localeCompare(b.name);
      const va = mode === "risk" ? (riskHistory[a.key]?.at(-1)?.probability ?? -1) : (history[a.key]?.at(-1)?.temp ?? -999);
      const vb = mode === "risk" ? (riskHistory[b.key]?.at(-1)?.probability ?? -1) : (history[b.key]?.at(-1)?.temp ?? -999);
      return sortMode === "value_desc" ? vb - va : va - vb;
    });
    return list;
  }, [stations, search, sortMode, history, riskHistory, mode]);

  // Flood-risk ticker categorised by tier — a flat A–Z list of 79 rows buries
  // the counties that actually need attention; grouping mirrors the Alerts
  // bulletin's Critical/High/Moderate/Low structure so the terminal reads the
  // same way the rest of the app does.
  const riskGroups = useMemo(() => {
    if (mode !== "risk") return null;
    const groups: Record<RiskTier, Station[]> = { Critical: [], High: [], Moderate: [], Low: [] };
    const unrated: Station[] = [];
    for (const s of visible) {
      const latest = riskHistory[s.key]?.at(-1);
      if (!latest) { unrated.push(s); continue; }
      groups[latest.tier].push(s);
    }
    return { groups, unrated };
  }, [visible, riskHistory, mode]);

  const loadedCount = stations.filter((s) => (mode === "risk" ? riskHistory[s.key]?.length : history[s.key]?.length)).length;
  const selectedHistory = history[stationKey(selected)] ?? [];
  const selLatest = selectedHistory.at(-1);
  const selectedRiskHistory = riskHistory[stationKey(selected)] ?? [];
  const selRiskLatest = selectedRiskHistory.at(-1);

  const dayForecast = useMemo(() => {
    if (!detail || !selectedDay) return [];
    return detail.forecast.filter((f) => f.day_key === selectedDay);
  }, [detail, selectedDay]);
  const isToday = detail && detail.daily.length > 0 && selectedDay === detail.daily[0].day_key;
  const nowDt = isToday ? Date.now() / 1000 : null;

  const chart = useChartTheme();
  const colors = useRiskColors();

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
      {!expanded && (
        <PageHeader
          eyebrow="Live Monitoring"
          title="Live Updates"
          subtitle="Current weather and flood-risk readings across all 79 counties, refreshed automatically."
        />
      )}

      {/* Terminal card — a bounded, scrollable list in the same idiom as the
          county table on Overview, not a separate dark app bolted onto this
          one. Expand takes it full-viewport. */}
      <Card
        styles={{ body: { padding: 0, display: "flex", flexDirection: "column", minHeight: 0, flex: 1 } }}
        style={{
          display: "flex",
          flexDirection: "column",
          minHeight: 0,
          overflow: "hidden",
          marginBottom: expanded ? 0 : 20,
          height: expanded ? "calc(100vh - 120px)" : 480,
        }}
        title={
          <Space orientation="vertical" size={2} style={{ padding: "4px 0" }}>
            <Text
              style={{
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--color-primary)",
              }}
            >
              Live Updates Terminal
            </Text>
            <Space size={6}>
              <Badge status="processing" />
              <Text type="secondary" className="tabular" style={{ fontSize: 12 }}>
                {loadedCount}/{stations.length} live · next tick in {mode === "risk" ? riskCountdown : omCountdown}s
              </Text>
            </Space>
          </Space>
        }
        extra={
          <Button
            size="small"
            icon={expanded ? <CompressOutlined /> : <ExpandOutlined />}
            onClick={() => setExpanded((e) => !e)}
          >
            {expanded ? "Collapse" : "Expand"}
          </Button>
        }
      >
        <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--color-border)", flexShrink: 0 }}>
          {mode === "weather" && omStatus === "error" && (
            <Alert
              type="error"
              showIcon
              style={{ marginBottom: 12 }}
              title="Couldn't reach the live updates feed."
              description="Retrying automatically — the county detail below refreshes independently and should still be working."
            />
          )}

          <div className="filter-bar">
            <div className="filter-bar-inner">
              <Segmented
                value={mode}
                onChange={(m) => setMode(m as Mode)}
                options={[
                  { label: "Weather", value: "weather" },
                  { label: "Flood Risk", value: "risk" },
                ]}
              />
              <Input
                allowClear
                prefix={<SearchOutlined style={{ opacity: 0.45 }} />}
                placeholder="Search county…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ width: 180, marginInlineStart: "auto" }}
              />
              <Select
                value={sortMode}
                onChange={(v) => setSortMode(v as SortMode)}
                style={{ width: 150 }}
                options={[
                  { value: "name", label: "A–Z" },
                  { value: "value_desc", label: mode === "risk" ? "Highest risk" : "Warmest" },
                  { value: "value_asc", label: mode === "risk" ? "Lowest risk" : "Coolest" },
                ]}
              />
            </div>
          </div>
        </div>

        <div style={{ overflowY: "auto", flex: 1, minHeight: 0 }}>
          {stations.length === 0 && (
            <div style={{ padding: 20 }}>
              <Skeleton active paragraph={{ rows: 4 }} />
            </div>
          )}

          {mode === "weather" && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))",
                gap: 10,
                padding: "14px 16px",
              }}
            >
              {visible.map((s) => {
                const readings = history[s.key] ?? [];
                const latest = readings.at(-1);
                const prev = readings.at(-2);
                const delta = latest && prev ? latest.temp - prev.temp : 0;
                const arrowColor = delta > 0.05 ? colors.Low : delta < -0.05 ? colors.Critical : chart.axis;
                const arrow = delta > 0.05 ? "▲" : delta < -0.05 ? "▼" : "•";
                if (latest && s.name === selected) {
                  return (
                    <FeaturedWeatherCard
                      key={s.key}
                      name={s.name}
                      temp={latest.temp}
                      description={latest.description}
                      readings={readings}
                      onClick={() => setSelected(s.name)}
                    />
                  );
                }
                return (
                  <TickerEntry
                    key={s.key}
                    name={s.name}
                    valueLabel={latest ? `${latest.temp.toFixed(1)}°` : "—"}
                    valueColor="var(--color-text)"
                    values={readings.map((r) => r.temp)}
                    arrow={latest ? arrow : ""}
                    arrowColor={arrowColor}
                    accentColor="var(--color-primary)"
                    selected={s.name === selected}
                    faded={!latest}
                    onClick={() => setSelected(s.name)}
                  />
                );
              })}
            </div>
          )}

          {mode === "risk" && riskGroups && (
            <div style={{ display: "flex", flexDirection: "column" }}>
              {RISK_ORDER.map((tier) => {
                const entries = riskGroups.groups[tier];
                if (entries.length === 0) return null;
                return (
                  <TierGroup key={tier} tier={tier} count={entries.length}>
                    {entries.map((s) => {
                      const readings = riskHistory[s.key] ?? [];
                      const latest = readings.at(-1)!;
                      const prev = readings.at(-2);
                      const delta = prev ? latest.probability - prev.probability : 0;
                      const arrowColor =
                        delta > 0.002 ? colors.Critical : delta < -0.002 ? colors.Low : chart.axis;
                      const arrow = delta > 0.002 ? "▲" : delta < -0.002 ? "▼" : "•";
                      if (s.name === selected) {
                        return (
                          <FeaturedRiskCard
                            key={s.key}
                            name={s.name}
                            tier={latest.tier}
                            probability={latest.probability}
                            readings={readings}
                            historicalPct={countyDetail ? countyDetail.flood_rate * 100 : null}
                            onClick={() => setSelected(s.name)}
                          />
                        );
                      }
                      return (
                        <TickerEntry
                          key={s.key}
                          name={s.name}
                          valueLabel={`${(latest.probability * 100).toFixed(0)}%`}
                          valueColor={riskVar(latest.tier)}
                          values={readings.map((r) => r.probability * 100)}
                          arrow={arrow}
                          arrowColor={arrowColor}
                          accentColor={riskVar(latest.tier)}
                          selected={s.name === selected}
                          faded={false}
                          onClick={() => setSelected(s.name)}
                        />
                      );
                    })}
                  </TierGroup>
                );
              })}
              {riskGroups.unrated.length > 0 && (
                <TierGroup tier={null} label="Awaiting First Reading" count={riskGroups.unrated.length}>
                  {riskGroups.unrated.map((s) => (
                    <TickerEntry
                      key={s.key}
                      name={s.name}
                      valueLabel="—"
                      valueColor="var(--color-text-muted)"
                      values={[]}
                      arrow=""
                      arrowColor={chart.axis}
                      accentColor="var(--color-primary)"
                      selected={s.name === selected}
                      faded
                      onClick={() => setSelected(s.name)}
                    />
                  ))}
                </TierGroup>
              )}
            </div>
          )}

          {stations.length > 0 && visible.length === 0 && (
            <div style={{ padding: 24 }}>
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={`No counties match "${search}".`} />
            </div>
          )}
        </div>

        <div style={{ padding: "10px 16px", borderTop: "1px solid var(--color-border)", flexShrink: 0 }}>
          <Text type="secondary" style={{ fontSize: 11, lineHeight: 1.5 }}>
            {mode === "weather"
              ? "OpenWeatherMap. Independent of the flood model's own climate inputs."
              : "Deployed nowcast model, live climate inputs. Click a county to open its full prediction."}
          </Text>
        </div>
      </Card>

      {/* Detail for the selected county, in the page's normal flow below the
          terminal — one scrollbar for the page rather than several nested ones
          fighting for the mouse wheel. Hidden while the terminal is expanded. */}
      {!expanded && (
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 10,
              marginBottom: 16,
            }}
          >
            <div>
              <Title level={4} style={{ margin: 0 }}>
                {selected}
              </Title>
              <Text type="secondary" style={{ fontSize: 13 }}>
                Pick a county above for its live detail.
              </Text>
            </div>
            <Space size={6}>
              <Badge status="processing" />
              <Text type="secondary" style={{ fontSize: 12 }}>
                live · next refresh in {mode === "risk" ? riskCountdown : owCountdown}s
              </Text>
            </Space>
          </div>

          {mode === "risk" && (
            <Card
              title={
                <Text
                  type="secondary"
                  style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
                >
                  Live Flood Risk — {selected}{" "}
                  <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>
                    (deployed nowcast model)
                  </span>
                </Text>
              }
            >
              {!selRiskLatest && <Skeleton active paragraph={{ rows: 4 }} />}
              {selRiskLatest &&
                (() => {
                  const prevReading =
                    selectedRiskHistory.length >= 2 ? selectedRiskHistory[selectedRiskHistory.length - 2] : null;
                  const riskDelta = prevReading ? selRiskLatest.probability - prevReading.probability : 0;
                  const deltaColor = riskDelta > 0.002 ? colors.Critical : riskDelta < -0.002 ? colors.Low : chart.axis;
                  const deltaArrow = riskDelta > 0.002 ? "▲" : riskDelta < -0.002 ? "▼" : "—";
                  const chartData = selectedRiskHistory.map((r) => ({ t: r.t, pct: r.probability * 100 }));
                  const historicalPct = countyDetail ? countyDetail.flood_rate * 100 : null;

                  return (
                    <>
                      <Row gutter={[16, 16]} justify="space-between" align="top">
                        <Col>
                          <Space align="baseline" size={12} wrap>
                            <span
                              className="tabular"
                              style={{
                                fontSize: 44,
                                fontWeight: 800,
                                color: colors[selRiskLatest.tier],
                                lineHeight: 1,
                              }}
                            >
                              {(selRiskLatest.probability * 100).toFixed(1)}%
                            </span>
                            <RiskTag tier={selRiskLatest.tier} />
                            {prevReading && (
                              <Text strong className="tabular" style={{ fontSize: 13, color: deltaColor }}>
                                {deltaArrow} {Math.abs(riskDelta * 100).toFixed(1)}pp
                              </Text>
                            )}
                          </Space>
                          <Text type="secondary" style={{ fontSize: 13, display: "block", marginTop: 6 }}>
                            Flood probability under current live climate conditions for {selected}.
                          </Text>
                        </Col>
                        {liveInputs?.live_data_available && (
                          <Col style={{ textAlign: "right" }}>
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              Climate inputs live as of
                              <br />
                              <b className="tabular">{liveInputs.last_updated}</b>
                              {liveInputs.source === "nasa-power" && <div>via NASA POWER</div>}
                            </Text>
                          </Col>
                        )}
                      </Row>

                      <div style={{ marginTop: 20 }}>
                        <div style={{ display: "flex", alignItems: "center", marginBottom: 6 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            Live trend
                          </Text>
                          <Text type="secondary" style={{ fontSize: 12, marginInlineStart: "auto" }}>
                            updates every {RISK_REFRESH_SECONDS}s
                          </Text>
                        </div>
                        {chartData.length >= 2 ? (
                          <ResponsiveContainer width="100%" height={180}>
                            <ComposedChart data={chartData} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                              <defs>
                                <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="0%" stopColor={colors[selRiskLatest.tier]} stopOpacity={0.35} />
                                  <stop offset="100%" stopColor={colors[selRiskLatest.tier]} stopOpacity={0} />
                                </linearGradient>
                              </defs>
                              <CartesianGrid stroke={chart.grid} vertical={false} />
                              <XAxis
                                dataKey="t"
                                tickFormatter={(t) =>
                                  new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                                }
                                tick={{ fontSize: 10, fill: chart.axis }}
                                axisLine={{ stroke: chart.grid }}
                                tickLine={false}
                              />
                              <YAxis
                                domain={[
                                  0,
                                  (dataMax: number) =>
                                    Math.max(
                                      20,
                                      Math.ceil((historicalPct ?? 0) / 10) * 10,
                                      Math.ceil(dataMax / 10) * 10
                                    ),
                                ]}
                                tickFormatter={(v) => `${v}%`}
                                tick={{ fontSize: 10, fill: chart.axis }}
                                axisLine={false}
                                tickLine={false}
                                width={34}
                              />
                              <Tooltip
                                labelFormatter={(l) =>
                                  new Date(l as number).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                                }
                                formatter={tooltipValue((n) => `${n.toFixed(1)}%`, "Flood probability")}
                                contentStyle={chart.tooltip}
                              />
                              {historicalPct !== null && (
                                <ReferenceLine
                                  y={historicalPct}
                                  stroke={chart.axis}
                                  strokeDasharray="4 3"
                                  label={{
                                    value: "15-yr avg",
                                    position: "insideTopRight",
                                    fontSize: 9,
                                    fill: chart.axis,
                                  }}
                                />
                              )}
                              <Area
                                type="monotone"
                                dataKey="pct"
                                stroke={colors[selRiskLatest.tier]}
                                strokeWidth={2.4}
                                fill="url(#riskGrad)"
                                dot={{ r: 2.5, fill: colors[selRiskLatest.tier], strokeWidth: 0 }}
                                animationDuration={500}
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        ) : (
                          <Empty
                            image={Empty.PRESENTED_IMAGE_SIMPLE}
                            description="Building trend from live readings — check back shortly."
                          />
                        )}
                      </div>

                      {liveInputs && (
                        <div style={{ marginTop: 20 }}>
                          <Text
                            type="secondary"
                            style={{
                              fontSize: 11,
                              fontWeight: 700,
                              letterSpacing: "0.05em",
                              textTransform: "uppercase",
                              display: "block",
                              marginBottom: 10,
                            }}
                          >
                            What's Driving This Prediction
                          </Text>
                          <Row gutter={[10, 10]}>
                            {(
                              [
                                ["Rainfall (MTD)", "rainfall_mm", "mm"],
                                ["Soil Moisture", "soil_moisture_mm", "mm"],
                                ["Max Temp", "max_temperature_celsius", "°C"],
                                ["Min Temp", "min_temperature_celsius", "°C"],
                                ["VPD", "vapor_pressure_deficit_kPa", "kPa"],
                              ] as [string, string, string][]
                            ).map(([label, key, unit]) => (
                              <Col key={key} xs={12} sm={8} lg={4}>
                                <InputStat
                                  label={label}
                                  value={liveInputs.inputs[key]}
                                  unit={unit}
                                  live={liveInputs.field_source[key] === "live"}
                                />
                              </Col>
                            ))}
                          </Row>
                        </div>
                      )}

                      {countyDetail && (
                        <div
                          style={{
                            marginTop: 20,
                            padding: "14px 16px",
                            background: "var(--color-bg)",
                            borderRadius: 6,
                            display: "flex",
                            alignItems: "center",
                            gap: 28,
                            flexWrap: "wrap",
                          }}
                        >
                          <Statistic
                            title="15-yr historical rate"
                            value={countyDetail.flood_rate * 100}
                            precision={1}
                            suffix="%"
                            styles={{ content: { fontSize: 18, fontWeight: 700 } }}
                          />
                          <Statistic
                            title="Flood events"
                            value={countyDetail.flood_events}
                            styles={{ content: { fontSize: 18, fontWeight: 700 } }}
                          />
                          <Statistic
                            title="Rank"
                            value={countyDetail.rank}
                            prefix="#"
                            suffix={`of ${countyDetail.n_counties}`}
                            styles={{ content: { fontSize: 18, fontWeight: 700 } }}
                          />
                          {countyDetail.flood_rate > 0 && (
                            <Text
                              style={{
                                marginInlineStart: "auto",
                                fontSize: 13,
                                maxWidth: 280,
                                textAlign: "right",
                              }}
                            >
                              Current live probability is{" "}
                              <b style={{ color: colors[selRiskLatest.tier] }}>
                                {(selRiskLatest.probability / countyDetail.flood_rate).toFixed(1)}×
                              </b>{" "}
                              this county's historical average.
                            </Text>
                          )}
                        </div>
                      )}

                      <Link to={`/prediction/${encodeURIComponent(selected)}`}>
                        <Button type="primary" style={{ marginTop: 20 }}>
                          View Full Prediction &amp; Recommendations →
                        </Button>
                      </Link>
                    </>
                  );
                })()}
            </Card>
          )}

          {mode === "weather" && !detail && !selLatest && (
            <Card>
              <Skeleton active paragraph={{ rows: 5 }} />
            </Card>
          )}

          {mode === "weather" && (detail || selLatest) && (
            <Card
              title={
                <Text
                  type="secondary"
                  style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
                >
                  Live Detail — {selected}{" "}
                  <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(OpenWeatherMap)</span>
                </Text>
              }
              extra={
                detail && (
                  <Space size={16}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Sunrise <b className="tabular">{localHM(detail.sunrise, detail.timezone_offset_s)}</b>
                    </Text>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Sunset <b className="tabular">{localHM(detail.sunset, detail.timezone_offset_s)}</b>
                    </Text>
                  </Space>
                )
              }
            >
              <div
                style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 16, flexWrap: "wrap" }}
              >
                {detail ? (
                  <>
                    <WeatherIcon code={detail.icon} size={68} />
                    <div>
                      <AnimatedNumber
                        value={detail.temp_c}
                        decimals={1}
                        suffix="°C"
                        style={{ fontSize: 34, fontWeight: 800, lineHeight: 1 }}
                      />
                      <Text type="secondary" style={{ fontSize: 14, textTransform: "capitalize", display: "block" }}>
                        {detail.description}
                      </Text>
                    </div>
                    <Row gutter={[16, 12]} style={{ flex: 1, minWidth: 220 }}>
                      {(
                        [
                          ["Feels Like", `${detail.feels_like_c.toFixed(1)}°C`],
                          ["Humidity", `${detail.humidity_pct}%`],
                          ["Wind", `${detail.wind_speed_ms.toFixed(1)} m/s`],
                          ["Pressure", `${detail.pressure_hpa} hPa`],
                        ] as [string, string][]
                      ).map(([label, value]) => (
                        <Col key={label} xs={12} md={6}>
                          <Stat label={label} value={value} />
                        </Col>
                      ))}
                    </Row>
                  </>
                ) : (
                  selLatest && (
                    <>
                      <WeatherIcon code={selLatest.icon} size={68} />
                      <div>
                        <AnimatedNumber
                          value={selLatest.temp}
                          decimals={1}
                          suffix="°C"
                          style={{ fontSize: 34, fontWeight: 800, lineHeight: 1 }}
                        />
                        <Text type="secondary" style={{ fontSize: 14, textTransform: "capitalize", display: "block" }}>
                          {selLatest.description}
                        </Text>
                      </div>
                      <Row gutter={[16, 12]} style={{ flex: 1, minWidth: 180 }}>
                        <Col xs={12} md={6}>
                          <Stat label="Humidity" value={`${selLatest.humidity}%`} />
                        </Col>
                        <Col xs={12} md={6}>
                          <Stat label="Wind" value={`${selLatest.wind.toFixed(1)} m/s`} />
                        </Col>
                      </Row>
                    </>
                  )
                )}
              </div>

              {selectedHistory.length >= 2 && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "8px 12px",
                    background: "var(--color-bg)",
                    borderRadius: 6,
                    marginBottom: 20,
                  }}
                >
                  <Text type="secondary" style={{ fontSize: 12, flexShrink: 0 }}>
                    Trend, last ~18h:
                  </Text>
                  <Sparkline values={selectedHistory.map((r) => r.temp)} width={120} height={24} />
                  <Text type="secondary" style={{ fontSize: 11, marginInlineStart: "auto" }}>
                    updates every {OW_TICKER_REFRESH_SECONDS}s
                  </Text>
                </div>
              )}

              <Text
                type="secondary"
                style={{ fontSize: 12, fontWeight: 600, display: "block", marginBottom: 10 }}
              >
                5-DAY FORECAST OUTLOOK{" "}
                <span style={{ fontWeight: 400 }}>
                  (a separate forward-looking forecast, not today's current reading above)
                </span>
              </Text>

              {!detail && (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={
                    unavailable ? (
                      <>
                        Forecast requires <Text code>OPENWEATHER_API_KEY</Text> to be configured on the server.
                      </>
                    ) : (
                      "Loading forecast…"
                    )
                  }
                />
              )}

              {detail && (
                <>
                  <Row gutter={[10, 10]} style={{ marginBottom: 20 }}>
                    {detail.daily.map((d, i) => (
                      <Col key={d.day_key} xs={12} sm={12} md={8} lg={4}>
                        <Card
                          hoverable
                          size="small"
                          onClick={() => setSelectedDay(d.day_key)}
                          style={{
                            textAlign: "center",
                            cursor: "pointer",
                            borderColor: selectedDay === d.day_key ? "var(--color-primary)" : undefined,
                            background: selectedDay === d.day_key ? "rgba(21,101,192,0.08)" : undefined,
                          }}
                          styles={{ body: { padding: "10px 6px" } }}
                        >
                          <Text
                            strong
                            style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.03em" }}
                          >
                            {i === 0 ? "Today" : d.label}
                          </Text>
                          <div style={{ display: "flex", justifyContent: "center", margin: "4px 0" }}>
                            <WeatherIcon code={d.icon} size={34} />
                          </div>
                          <div className="tabular" style={{ fontSize: 13 }}>
                            <b>{Math.round(d.temp_max)}°</b>{" "}
                            <Text type="secondary">{Math.round(d.temp_min)}°</Text>
                          </div>
                          <Text style={{ fontSize: 11, color: "var(--color-primary)" }}>
                            💧 {Math.round(d.rain_probability_max * 100)}%
                          </Text>
                        </Card>
                      </Col>
                    ))}
                  </Row>

                  <Text strong style={{ fontSize: 13, display: "block", marginBottom: 8 }}>
                    Hourly — {isToday ? "Today" : detail.daily.find((d) => d.day_key === selectedDay)?.label}
                    <Text type="secondary" style={{ fontWeight: 400, marginInlineStart: 8 }}>
                      temperature &amp; rain chance
                    </Text>
                  </Text>

                  <ResponsiveContainer width="100%" height={240}>
                    <ComposedChart data={dayForecast} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
                      <defs>
                        <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={chart.accent} stopOpacity={0.32} />
                          <stop offset="100%" stopColor={chart.accent} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke={chart.grid} vertical={false} />
                      <XAxis
                        dataKey="dt"
                        tickFormatter={(t) => localHM(t, detail.timezone_offset_s)}
                        tick={{ fontSize: 10, fill: chart.axis }}
                        axisLine={{ stroke: chart.grid }}
                        tickLine={false}
                      />
                      <YAxis
                        yAxisId="rain"
                        orientation="right"
                        domain={[0, 1]}
                        ticks={RAIN_TICKS}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                        tick={{ fontSize: 10, fill: "#93c5fd" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        yAxisId="temp"
                        domain={["dataMin - 3", "dataMax + 3"]}
                        tickFormatter={(v) => `${Math.round(v)}°`}
                        tick={{ fontSize: 10, fill: chart.axis }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        labelFormatter={(l) => localHM(l as number, detail.timezone_offset_s)}
                        formatter={(v, name) =>
                          (name === "Rain chance"
                            ? [`${(Number(v) * 100).toFixed(0)}%`, name]
                            : [`${Number(v).toFixed(1)}°C`, name]) as [string, string]
                        }
                        contentStyle={chart.tooltip}
                      />
                      {nowDt && (
                        <ReferenceLine
                          yAxisId="temp"
                          x={nowDt}
                          stroke={chart.accent}
                          strokeDasharray="4 3"
                          label={{ value: "now", position: "top", fontSize: 10, fill: chart.accent }}
                        />
                      )}
                      <Bar
                        yAxisId="rain"
                        dataKey="rain_probability"
                        name="Rain chance"
                        fill="#93c5fd"
                        radius={[4, 4, 0, 0]}
                        barSize={16}
                        animationDuration={500}
                      />
                      <Area
                        yAxisId="temp"
                        type="monotone"
                        dataKey="temp_c"
                        name="Temperature"
                        stroke={chart.accent}
                        strokeWidth={2.4}
                        fill="url(#tempGrad)"
                        dot={{ r: 2.5, fill: chart.accent, strokeWidth: 0 }}
                        animationDuration={500}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>

                  <Caption>
                    5-day/hourly forecast: OpenWeatherMap, refreshed every {OW_REFRESH_SECONDS}s. Situational
                    awareness only — the flood model's own nowcast/outlook uses separately aggregated climate data,
                    not this forecast directly.
                  </Caption>
                </>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

/** Tier band inside the risk terminal — a sticky label above that tier's
 * grid, mirroring the Critical/High/Moderate/Low structure of the Alerts
 * bulletin so the terminal reads the same way the rest of the app does. */
function TierGroup({
  tier,
  label,
  count,
  children,
}: {
  tier: RiskTier | null;
  label?: string;
  count: number;
  children: ReactNode;
}) {
  const color = tier ? riskVar(tier) : "var(--color-text-muted)";
  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color,
          padding: "10px 16px",
          position: "sticky",
          top: 0,
          background: "var(--color-surface)",
          zIndex: 1,
        }}
      >
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0 }} />
        {label ?? tier}
        <Text type="secondary" style={{ fontWeight: 500, fontSize: 11 }}>
          ({count})
        </Text>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))",
          gap: 10,
          padding: "0 16px 12px",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function TickerEntry({
  name,
  valueLabel,
  valueColor,
  values,
  arrow,
  arrowColor,
  accentColor,
  selected,
  faded,
  onClick,
}: {
  name: string;
  valueLabel: string;
  valueColor: string;
  values: number[];
  arrow: string;
  arrowColor: string;
  accentColor: string;
  selected: boolean;
  faded: boolean;
  onClick: () => void;
}) {
  return (
    <Card
      hoverable
      size="small"
      onClick={onClick}
      style={{
        cursor: "pointer",
        opacity: faded ? 0.5 : 1,
        borderColor: selected ? accentColor : undefined,
        background: selected ? "rgba(21,101,192,0.08)" : undefined,
      }}
      styles={{ body: { padding: "12px 14px" } }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <Text
          strong
          style={{ fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        >
          {name}
        </Text>
        <span className="tabular" style={{ fontSize: 12, fontWeight: 700, color: arrowColor, flexShrink: 0 }}>
          {arrow}
        </span>
      </div>
      <div className="tabular" style={{ fontSize: 24, fontWeight: 800, color: valueColor, marginTop: 6 }}>
        {valueLabel}
      </div>
      <div style={{ color: valueColor, marginTop: 6 }}>
        {values.length > 0 && <Sparkline values={values} width={190} height={36} />}
      </div>
    </Card>
  );
}

/** The clicked card in the terminal grid — spans two columns and swaps the
 * tiny sparkline for a real axis'd, gridded area chart that mounts fresh
 * (recharts animates the draw-in) each time a different county is clicked, so
 * clicking actually does something rather than just toggling a border. */
function FeaturedRiskCard({
  name,
  tier,
  probability,
  readings,
  historicalPct,
  onClick,
}: {
  name: string;
  tier: RiskTier;
  probability: number;
  readings: RiskReading[];
  historicalPct: number | null;
  onClick: () => void;
}) {
  const chart = useChartTheme();
  const colors = useRiskColors();
  const chartData = readings.map((r) => ({ t: r.t, pct: r.probability * 100 }));
  const gradId = `feat-risk-${name.replace(/[^a-zA-Z0-9]/g, "")}`;

  return (
    <Card
      size="small"
      onClick={onClick}
      style={{ gridColumn: "span 2", cursor: "pointer", borderColor: colors[tier] }}
      styles={{ body: { padding: "14px 16px" } }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <Text strong style={{ fontSize: 15 }}>
          {name}
        </Text>
        <span className="tabular" style={{ fontSize: 26, fontWeight: 800, color: colors[tier] }}>
          {(probability * 100).toFixed(0)}%
        </span>
      </div>
      {chartData.length >= 2 ? (
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={colors[tier]} stopOpacity={0.4} />
                <stop offset="100%" stopColor={colors[tier]} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={chart.grid} vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 9, fill: chart.axis }}
              tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 9, fill: chart.axis }}
              width={30}
              axisLine={false}
              tickLine={false}
              domain={[
                0,
                (max: number) =>
                  Math.max(20, Math.ceil((historicalPct ?? 0) / 10) * 10, Math.ceil(max / 10) * 10),
              ]}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              formatter={tooltipValue((n) => `${n.toFixed(1)}%`, "Flood probability")}
              labelFormatter={(l) =>
                new Date(l as number).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
              }
              contentStyle={chart.tooltip}
            />
            {historicalPct != null && (
              <ReferenceLine
                y={historicalPct}
                stroke={chart.axis}
                strokeDasharray="3 3"
                label={{ value: "15-yr avg", position: "insideTopRight", fontSize: 8, fill: chart.axis }}
              />
            )}
            <Area
              type="monotone"
              dataKey="pct"
              stroke={colors[tier]}
              strokeWidth={2}
              fill={`url(#${gradId})`}
              animationDuration={650}
              animationEasing="ease-out"
              dot={{ r: 2, fill: colors[tier], strokeWidth: 0 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="Building trend from live readings — check back shortly."
        />
      )}
      <Link
        to={`/prediction/${encodeURIComponent(name)}`}
        onClick={(e) => e.stopPropagation()}
        style={{ fontSize: 13, color: colors[tier], fontWeight: 700 }}
      >
        View Full Prediction →
      </Link>
    </Card>
  );
}

function FeaturedWeatherCard({
  name,
  temp,
  description,
  readings,
  onClick,
}: {
  name: string;
  temp: number;
  description: string;
  readings: Reading[];
  onClick: () => void;
}) {
  const chart = useChartTheme();
  const colors = useRiskColors();
  const chartData = readings.map((r) => ({ t: r.t, temp: r.temp }));
  const up = readings.length >= 2 ? readings[readings.length - 1].temp >= readings[0].temp : true;
  const color = up ? colors.Low : colors.Critical;
  const gradId = `feat-wx-${name.replace(/[^a-zA-Z0-9]/g, "")}`;

  return (
    <Card
      size="small"
      onClick={onClick}
      style={{ gridColumn: "span 2", cursor: "pointer", borderColor: "var(--color-primary)" }}
      styles={{ body: { padding: "14px 16px" } }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <Text strong style={{ fontSize: 15 }}>
          {name}
        </Text>
        <span className="tabular" style={{ fontSize: 26, fontWeight: 800 }}>
          {temp.toFixed(1)}°C
        </span>
      </div>
      <Text type="secondary" style={{ fontSize: 13, textTransform: "capitalize", display: "block" }}>
        {description}
      </Text>
      {chartData.length >= 2 ? (
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 4, left: -14, bottom: 0 }}>
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.4} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={chart.grid} vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 9, fill: chart.axis }}
              tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 9, fill: chart.axis }}
              width={30}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}°`}
              domain={["dataMin - 1", "dataMax + 1"]}
            />
            <Tooltip
              formatter={tooltipValue((n) => `${n.toFixed(1)}°C`, "Temperature")}
              labelFormatter={(l) =>
                new Date(l as number).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
              }
              contentStyle={chart.tooltip}
            />
            <Area
              type="monotone"
              dataKey="temp"
              stroke={color}
              strokeWidth={2}
              fill={`url(#${gradId})`}
              animationDuration={650}
              animationEasing="ease-out"
              dot={{ r: 2, fill: color, strokeWidth: 0 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="Building trend from live readings — check back shortly."
        />
      )}
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    // Left-aligned: these sit in a wrapping grid now, not flush against the
    // card's right edge, so right-aligning them detaches each value from its
    // own label once the row wraps.
    <div>
      <Text type="secondary" style={{ fontSize: 10, letterSpacing: "0.04em", display: "block" }}>
        {label.toUpperCase()}
      </Text>
      <Text strong className="tabular" style={{ fontSize: 15 }}>
        {value}
      </Text>
    </div>
  );
}

/** One live model input, badged by whether it came from the live feed or fell
 * back to this county's historical median. */
function InputStat({ label, value, unit, live }: { label: string; value: number; unit: string; live: boolean }) {
  return (
    <Card size="small" styles={{ body: { padding: "10px 12px" } }} style={{ background: "var(--color-bg)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
        <Text type="secondary" style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.04em" }}>
          {label}
        </Text>
        <Tag
          color={live ? "green" : "default"}
          style={{ marginInlineEnd: 0, fontSize: 9, lineHeight: "14px", paddingInline: 4 }}
        >
          {live ? "Live" : "Hist."}
        </Tag>
      </div>
      <Text strong className="tabular" style={{ fontSize: 16 }}>
        {Number.isFinite(value) ? value.toFixed(1) : "—"}
        <Text type="secondary" style={{ fontSize: 11, fontWeight: 500 }}>
          {" "}
          {unit}
        </Text>
      </Text>
    </Card>
  );
}
