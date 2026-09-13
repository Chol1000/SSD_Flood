import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Card,
  Row,
  Col,
  Table,
  Input,
  Button,
  Typography,
  Space,
  Tag,
  Badge,
  Empty,
  Spin,
  Skeleton,
  Tooltip as AntTooltip,
} from "antd";
import {
  DownloadOutlined,
  RiseOutlined,
  EnvironmentOutlined,
  AimOutlined,
  CloudOutlined,
  SearchOutlined,
  ArrowRightOutlined,
} from "@ant-design/icons";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip as ReTooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { api } from "../api";
import type { OverviewData, RiskTier, CountyListItem, ScanResult, ModelInfo } from "../types";
import {
  PageHeader, StatCard, StatRow, RiskTag, Caption, useRiskColors, useChartTheme, tooltipValue, RISK_ORDER,
} from "../ui";

const { Text } = Typography;

const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTH_FULL = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];
const LIVE_REFRESH_SECONDS = 300;

/* Calendar heatmap ramp: pale blue-grey → amber → deep red, saturating at a
   50% monthly rate (above that, counties are flooding at a rate where finer
   gradation stops telling a responder anything new). */
const HEAT_STOPS: [number, string][] = [
  [0, "#eaf1fb"],
  [0.35, "#f5c97a"],
  [0.7, "#e0722e"],
  [1, "#b91c1c"],
];

function lerpColor(a: string, b: string, f: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const p = pa.map((v, i) => Math.round(v + (pb[i] - v) * f));
  return `rgb(${p[0]},${p[1]},${p[2]})`;
}

function heatColor(rate: number): string {
  if (rate <= 0) return HEAT_STOPS[0][1];
  const t = Math.min(1, rate / 0.5);
  for (let i = 0; i < HEAT_STOPS.length - 1; i++) {
    const [t0, c0] = HEAT_STOPS[i];
    const [t1, c1] = HEAT_STOPS[i + 1];
    if (t >= t0 && t <= t1) return lerpColor(c0, c1, (t - t0) / (t1 - t0));
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

export default function Overview() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [allCounties, setAllCounties] = useState<CountyListItem[]>([]);
  const [live, setLive] = useState<ScanResult[] | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [search, setSearch] = useState("");
  const [countdown, setCountdown] = useState(LIVE_REFRESH_SECONDS);

  const riskColors = useRiskColors();
  const chart = useChartTheme();

  useEffect(() => {
    api.overview().then(setData).catch(console.error);
    api.counties().then(setAllCounties).catch(console.error);
    api.modelInfo().then(setModelInfo).catch(console.error);
  }, []);

  useEffect(() => {
    const loadLive = () => {
      api
        .scan({ month: new Date().getMonth() + 1, use_live: true })
        .then((r) => {
          setLive(r);
          setCountdown(LIVE_REFRESH_SECONDS);
        })
        .catch(console.error);
    };
    loadLive();
    const t = setInterval(loadLive, LIVE_REFRESH_SECONDS * 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setCountdown((c) => (c <= 1 ? LIVE_REFRESH_SECONDS : c - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  const liveTierCounts = useMemo(() => {
    const counts: Record<RiskTier, number> = { Critical: 0, High: 0, Moderate: 0, Low: 0 };
    if (live) for (const r of live) counts[r.risk_tier]++;
    return counts;
  }, [live]);

  const liveTop5 = useMemo(
    () => (live ? [...live].sort((a, b) => b.probability - a.probability).slice(0, 5) : []),
    [live]
  );

  // Counties whose live nowcast tier reads worse than their historical
  // baseline tier — the anomalies actually worth a human's attention, rather
  // than a repeat of the static ranking in the table below.
  const riskMovers = useMemo(() => {
    if (!live || allCounties.length === 0) return [];
    const rank: Record<RiskTier, number> = { Low: 0, Moderate: 1, High: 2, Critical: 3 };
    const histByCounty: Record<string, RiskTier> = {};
    for (const c of allCounties) histByCounty[c.county] = c.risk_tier;
    return live
      .map((r) => ({ ...r, historical_tier: histByCounty[r.county] }))
      .filter((r) => r.historical_tier && rank[r.risk_tier] > rank[r.historical_tier])
      .sort(
        (a, b) =>
          rank[b.risk_tier] - rank[b.historical_tier] - (rank[a.risk_tier] - rank[a.historical_tier])
      )
      .slice(0, 6);
  }, [live, allCounties]);

  const rows = useMemo(
    () => allCounties.filter((c) => c.county.toLowerCase().includes(search.toLowerCase())),
    [allCounties, search]
  );

  if (!data) {
    return (
      <>
        <Skeleton active paragraph={{ rows: 2 }} style={{ marginBottom: 24 }} />
        <Row gutter={[16, 16]}>
          {[0, 1, 2, 3].map((i) => (
            <Col key={i} xs={24} sm={12} xl={6}>
              <Card size="small">
                <Skeleton active paragraph={{ rows: 1 }} title={false} />
              </Card>
            </Col>
          ))}
        </Row>
      </>
    );
  }

  const years = Array.from(
    { length: data.period.year_max - data.period.year_min + 1 },
    (_, i) => data.period.year_min + i
  );
  const calByYearMonth: Record<string, number> = {};
  for (const c of data.calendar) calByYearMonth[`${c.year}-${c.month}`] = c.flood;

  const donutData = RISK_ORDER.map((tier) => ({ name: tier, value: data.risk_tier_counts[tier] }));
  const modelPrecision = modelInfo?.test_metrics?.[modelInfo.best_model_name]?.precision;
  const now = new Date();

  // National annual trend — one number per year, rolled up from the same
  // calendar data driving the heatmap, so a reader gets the headline
  // direction without having to parse a 15x12 grid.
  const yearlyTrend = years.map((y) => {
    const vals = MONTH_SHORT.map((_, mi) => calByYearMonth[`${y}-${mi + 1}`]).filter(
      (v) => v != null
    ) as number[];
    return {
      year: y,
      rate: vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) * 100 : 0,
    };
  });
  const nationalMeanPct = data.national_mean_rate * 100;

  return (
    <>
      <PageHeader
        eyebrow="South Sudan Flood Early Warning System"
        title="National Overview"
        subtitle={`${data.n_counties} counties · training record ${data.period.year_min}–${data.period.year_max} · predictions forecast forward from today, not from this window`}
        extra={
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            href={api.nationalReportUrl()}
            target="_blank"
            rel="noreferrer"
          >
            National Report
          </Button>
        }
      />

      {/* Live nowcast snapshot — ties the landing page to today's actual
          model run, not just the historical record below it. */}
      <Card
        style={{ marginBottom: 20, borderInlineStart: "3px solid var(--color-primary)" }}
        styles={{ body: { padding: "18px 22px" } }}
        title={
          <Space wrap size={12} style={{ padding: "4px 0" }}>
            <Badge status="processing" />
            <Text strong style={{ fontSize: 13, letterSpacing: "0.04em", textTransform: "uppercase" }}>
              Live Nowcast — {MONTH_FULL[now.getMonth()]} {now.getFullYear()}
            </Text>
          </Space>
        }
        extra={
          live ? (
            <Text type="secondary" style={{ fontSize: 12 }} className="tabular">
              updated {LIVE_REFRESH_SECONDS - countdown}s ago · next in {Math.floor(countdown / 60)}:
              {String(countdown % 60).padStart(2, "0")}
            </Text>
          ) : (
            <Spin size="small" />
          )
        }
      >
        <Row gutter={[16, 16]} className="compact-stat-row">
          {RISK_ORDER.map((tier) => (
            <Col key={tier} xs={12} sm={6}>
              <div style={{ textAlign: "center" }}>
                <div
                  className="tabular"
                  style={{ fontSize: 30, fontWeight: 800, lineHeight: 1.2, color: riskColors[tier] }}
                >
                  {live ? liveTierCounts[tier] : "–"}
                </div>
                <Text type="secondary" style={{ fontSize: 12, fontWeight: 600 }}>
                  {tier} now
                </Text>
              </div>
            </Col>
          ))}
        </Row>

        {live && liveTop5.length > 0 && (
          <div style={{ marginTop: 18, paddingTop: 14, borderTop: "1px solid var(--color-border)" }}>
            <Space wrap size={[8, 8]}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Highest live risk right now:
              </Text>
              {liveTop5.map((r) => (
                <Link key={r.county} to={`/prediction/${encodeURIComponent(r.county)}`}>
                  <Tag
                    style={{
                      marginInlineEnd: 0,
                      cursor: "pointer",
                      borderColor: riskColors[r.risk_tier],
                      color: "var(--color-text)",
                      background: "transparent",
                    }}
                  >
                    <span
                      style={{
                        display: "inline-block",
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        background: riskColors[r.risk_tier],
                        marginInlineEnd: 6,
                      }}
                    />
                    {r.county} <b className="tabular">{(r.probability * 100).toFixed(0)}%</b>
                  </Tag>
                </Link>
              ))}
            </Space>
          </div>
        )}
      </Card>

      <Text
        type="secondary"
        style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}
      >
        Historical baseline — {data.period.year_min}–{data.period.year_max}
      </Text>
      <div style={{ height: 10 }} />

      <StatRow>
        <StatCard
          label="Total Flood Events"
          value={data.total_flood_events}
          icon={<CloudOutlined />}
          hint="County-months recorded as flooded across the full training record."
        />
        <StatCard
          label="National Mean Rate"
          value={data.national_mean_rate * 100}
          precision={1}
          suffix="%"
          icon={<RiseOutlined />}
          hint="Average share of counties recording a flood in any given month."
        />
        <StatCard label="Highest-Risk County" value={data.highest_risk_county} icon={<EnvironmentOutlined />} />
        <StatCard
          label={`Model Precision — ${modelInfo?.best_model_name ?? "…"}`}
          value={modelPrecision != null ? modelPrecision * 100 : "…"}
          precision={modelPrecision != null ? 1 : undefined}
          suffix={modelPrecision != null ? "%" : ""}
          icon={<AimOutlined />}
          color="var(--color-primary)"
          hint="Share of flood warnings that were correct, on the held-out test set."
        />
      </StatRow>

      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        <Col xs={24} xl={14}>
          <Card title="Flood Calendar — National Rate by Year & Month" style={{ height: "100%" }}>
            <div className="scroll-x">
              <table style={{ borderCollapse: "collapse", fontSize: 11, minWidth: 420 }}>
                <thead>
                  <tr>
                    <th />
                    {MONTH_SHORT.map((m) => (
                      <th key={m} style={{ padding: "2px 4px", color: "var(--color-text-muted)", fontWeight: 500 }}>
                        {m}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {years.map((y) => (
                    <tr key={y}>
                      <td className="tabular" style={{ padding: "2px 8px 2px 0", color: "var(--color-text-muted)" }}>
                        {y}
                      </td>
                      {MONTH_SHORT.map((m, mi) => {
                        const rate = calByYearMonth[`${y}-${mi + 1}`];
                        return (
                          <td key={mi} style={{ padding: 1 }}>
                            <AntTooltip
                              title={rate != null ? `${m} ${y} — ${(rate * 100).toFixed(1)}% of counties` : "No data"}
                            >
                              <div
                                style={{
                                  width: 26,
                                  height: 18,
                                  borderRadius: 3,
                                  background: rate != null ? heatColor(rate) : "var(--color-bg)",
                                  border: "1px solid rgba(16,24,40,0.06)",
                                }}
                              />
                            </AntTooltip>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Space size={6} style={{ marginTop: 12 }}>
              <Text type="secondary" style={{ fontSize: 11 }}>
                Low
              </Text>
              <div style={{ display: "flex", height: 8, width: 120, borderRadius: 4, overflow: "hidden" }}>
                {HEAT_STOPS.map(([, c]) => (
                  <div key={c} style={{ flex: 1, background: c }} />
                ))}
              </div>
              <Text type="secondary" style={{ fontSize: 11 }}>
                High
              </Text>
            </Space>

            <Caption>
              Darker = higher share of counties flooded that month. 2022 onward may undercount — see the Data Quality
              note on any County Profile. Use <Link to="/historical">Historical</Link> to filter by county.
            </Caption>
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Space orientation="vertical" size={16} style={{ display: "flex" }}>
            <Card
              title="Historical Risk Distribution"
              extra={
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {data.period.year_min}–{data.period.year_max} baseline
                </Text>
              }
            >
              <div style={{ position: "relative", height: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={donutData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={50}
                      outerRadius={76}
                      paddingAngle={2}
                      animationDuration={600}
                    >
                      {donutData.map((d) => (
                        <Cell
                          key={d.name}
                          fill={riskColors[d.name as RiskTier]}
                          stroke={chart.surface}
                          strokeWidth={2}
                        />
                      ))}
                    </Pie>
                    <ReTooltip
                      formatter={tooltipValue((n) => `${n} counties`)}
                      contentStyle={chart.tooltip}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    pointerEvents: "none",
                  }}
                >
                  <div className="tabular" style={{ fontSize: 24, fontWeight: 800 }}>
                    {data.n_counties}
                  </div>
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    counties
                  </Text>
                </div>
              </div>
              <Space wrap size={[14, 6]} style={{ justifyContent: "center", width: "100%", marginTop: 8 }}>
                {RISK_ORDER.map((tier) => (
                  <Space key={tier} size={5}>
                    <span
                      style={{ width: 8, height: 8, borderRadius: 2, background: riskColors[tier], display: "block" }}
                    />
                    <Text style={{ fontSize: 12 }}>{tier}</Text>
                    <b className="tabular">{data.risk_tier_counts[tier]}</b>
                  </Space>
                ))}
              </Space>
              <Caption>
                Tiers here reflect the 2011–2025 record — they differ from the live counts above, which reflect today's
                climate.
              </Caption>
            </Card>

            <Card title="Risk Movers — Live vs Historical Baseline">
              {!live && <Skeleton active paragraph={{ rows: 3 }} title={false} />}
              {live && riskMovers.length === 0 && (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No county is currently running above its historical baseline tier."
                />
              )}
              {live && riskMovers.length > 0 && (
                <Space orientation="vertical" size={10} style={{ display: "flex" }}>
                  {riskMovers.map((r) => (
                    <Link
                      key={r.county}
                      to={`/prediction/${encodeURIComponent(r.county)}`}
                      style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}
                    >
                      <Text strong style={{ fontSize: 13 }}>
                        {r.county}
                      </Text>
                      <Space size={6}>
                        <RiskTag tier={r.historical_tier} dot />
                        <ArrowRightOutlined style={{ fontSize: 10, color: "var(--color-text-muted)" }} />
                        <RiskTag tier={r.risk_tier} dot />
                      </Space>
                    </Link>
                  ))}
                </Space>
              )}
              <Caption>Compares today's live nowcast tier against each county's 2011–2025 historical tier.</Caption>
            </Card>
          </Space>
        </Col>
      </Row>

      <Card
        title={`National Annual Flood Trend — ${data.period.year_min}–${data.period.year_max}`}
        style={{ marginBottom: 20 }}
      >
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={yearlyTrend} margin={{ top: 6, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid stroke={chart.grid} vertical={false} />
            <XAxis dataKey="year" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
            <YAxis unit="%" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
            <ReferenceLine
              y={nationalMeanPct}
              stroke={chart.axis}
              strokeDasharray="3 3"
              label={{ value: "15-yr mean", position: "insideTopRight", fontSize: 10, fill: chart.axis }}
            />
            <ReTooltip
              formatter={tooltipValue((n) => `${n.toFixed(1)}%`, "National rate")}
              labelFormatter={(y) => (Number(y) >= 2022 ? `${y} (data gap — see note)` : String(y))}
              contentStyle={chart.tooltip}
            />
            <Line
              type="monotone"
              dataKey="rate"
              stroke={chart.accent}
              strokeWidth={2.2}
              dot={{ r: 3 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        <Caption>
          Share of counties recording a flood month, averaged across the year. The 2022–2025 drop coincides with the
          satellite data gap noted throughout this app, not a real decline.
        </Caption>
      </Card>

      <Card
        title={
          <Space>
            <span>All Counties</span>
            <Tag>{rows.length}</Tag>
          </Space>
        }
        extra={
          <Input
            allowClear
            prefix={<SearchOutlined style={{ opacity: 0.45 }} />}
            placeholder="Search county…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: 200 }}
          />
        }
      >
        <Table<CountyListItem>
          dataSource={rows}
          rowKey="county"
          size="middle"
          scroll={{ x: "max-content" }}
          pagination={{ pageSize: 15, showSizeChanger: false, hideOnSinglePage: true }}
          columns={[
            {
              title: "County",
              dataIndex: "county",
              sorter: (a, b) => a.county.localeCompare(b.county),
              render: (county: string, row) => (
                <Space size={8}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 3,
                      height: 18,
                      borderRadius: 2,
                      background: riskColors[row.risk_tier],
                    }}
                  />
                  <Link to={`/county/${encodeURIComponent(county)}`}>
                    <Text strong>{county}</Text>
                  </Link>
                </Space>
              ),
            },
            {
              title: "Historical Rate",
              dataIndex: "flood_rate",
              align: "right",
              defaultSortOrder: "descend",
              sorter: (a, b) => a.flood_rate - b.flood_rate,
              render: (v: number) => <span className="tabular">{(v * 100).toFixed(1)}%</span>,
            },
            {
              title: "Events",
              dataIndex: "flood_events",
              align: "right",
              sorter: (a, b) => a.flood_events - b.flood_events,
              render: (v: number) => <span className="tabular">{v}</span>,
            },
            {
              title: "Risk",
              dataIndex: "risk_tier",
              align: "right",
              filters: RISK_ORDER.map((t) => ({ text: t, value: t })),
              onFilter: (value, row) => row.risk_tier === value,
              render: (tier: RiskTier) => <RiskTag tier={tier} />,
            },
            {
              title: "",
              key: "actions",
              align: "right",
              width: 96,
              render: (_, row) => <Link to={`/county/${encodeURIComponent(row.county)}`}>Profile →</Link>,
            },
          ]}
        />
      </Card>
    </>
  );
}
