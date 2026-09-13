import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Card, Row, Col, Select, Slider as SliderRange, Tag, Button, Table, Typography, Space, Empty, Tooltip,
} from "antd";
import { WarningOutlined } from "@ant-design/icons";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, Legend, ResponsiveContainer, Cell,
} from "recharts";
import { api } from "../api";
import type { CountyListItem } from "../types";
import Explainer from "../components/Explainer";
import { PageHeader, Caption, useChartTheme, useRiskColors, tooltipValue } from "../ui";

const { Text } = Typography;

const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/* Categorical series palette — distinct hues that stay separable for up to
   eight simultaneously-selected counties on one line chart. */
const PALETTE = ["#1565c0", "#e0a62e", "#2bae66", "#e14b4b", "#b18ae0", "#4fd1c5", "#f97316", "#0ea5e9"];
const HEATMAP_CAP = 8;
const COOCCURRENCE_MIN = 2;
const COOCCURRENCE_MAX = 12;


function lerpColor(a: string, b: string, f: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const p = pa.map((v, i) => Math.round(v + (pb[i] - v) * f));
  return `rgb(${p[0]},${p[1]},${p[2]})`;
}

function heatColor(rate: number): string {
  if (rate <= 0) return "#EEF1F5";
  const t = Math.min(1, rate);
  const stops: [number, string][] = [[0, "#EAF1FB"], [0.35, "#F5C97A"], [0.7, "#E0722E"], [1, "#B91C1C"]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) return lerpColor(c0, c1, (t - t0) / (t1 - t0));
  }
  return stops[stops.length - 1][1];
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n === 0) return 0;
  const meanA = a.reduce((s, v) => s + v, 0) / n;
  const meanB = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA, db = b[i] - meanB;
    num += da * db; denA += da * da; denB += db * db;
  }
  const den = Math.sqrt(denA * denB);
  return den === 0 ? 0 : num / den;
}

/** Diverging ramp for the co-occurrence matrix: red for counties that flood
 * together, blue for those that rarely coincide, near-neutral at r ≈ 0. */
function corrColor(r: number): string {
  if (r >= 0) return lerpColor("#f3f4f6", "#b91c1c", Math.min(1, r));
  return lerpColor("#f3f4f6", "#1565c0", Math.min(1, -r));
}

export default function Historical() {
  const [counties, setCounties] = useState<CountyListItem[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [yearRange, setYearRange] = useState<[number, number]>([2011, 2025]);
  const [months, setMonths] = useState<number[]>([...Array(12)].map((_, i) => i + 1));
  const [records, setRecords] = useState<any[]>([]);
  const [national, setNational] = useState<{ year: number; flood: number }[]>([]);

  useEffect(() => {
    api.counties().then((cs) => {
      setCounties(cs);
      const topByRisk = [...cs].sort((a, b) => b.flood_rate - a.flood_rate).slice(0, 5).map((c) => c.county);
      setSelected(topByRisk);
    });
  }, []);

  useEffect(() => {
    if (selected.length === 0 || months.length === 0) return;
    api.historical(selected, yearRange[0], yearRange[1], months).then((res) => {
      setRecords(res.records);
      setNational(res.national);
    });
  }, [selected, yearRange, months]);

  const annualByCounty = useMemo(() => {
    const byYear: Record<number, any> = {};
    for (let y = yearRange[0]; y <= yearRange[1]; y++) byYear[y] = { year: y };
    for (const c of selected) {
      const rows = records.filter((r) => r.county === c);
      const byY: Record<number, number[]> = {};
      for (const r of rows) { (byY[r.year] ??= []).push(r.flood); }
      for (const y of Object.keys(byYear).map(Number)) {
        const vals = byY[y];
        byYear[y][c] = vals ? (vals.reduce((a, b) => a + b, 0) / vals.length) * 100 : 0;
      }
    }
    const nat: Record<number, number> = {};
    for (const n of national) nat[n.year] = n.flood * 100;
    for (const y of Object.keys(byYear).map(Number)) byYear[y]["National"] = nat[y] ?? 0;
    return Object.values(byYear);
  }, [records, national, selected, yearRange]);

  const seasonality = useMemo(() => {
    const byMonth: Record<number, number[]> = {};
    for (const r of records) (byMonth[r.month] ??= []).push(r.flood);
    return months.sort((a, b) => a - b).map((m) => ({
      month: MONTH_SHORT[m - 1],
      rate: byMonth[m] ? (byMonth[m].reduce((a, b) => a + b, 0) / byMonth[m].length) * 100 : 0,
    }));
  }, [records, months]);

  // Per-county summary for the current filter window — mean rate, total events, and
  // where that stacks up against the county's all-time national rank.
  const summary = useMemo(() => {
    const rankByCounty: Record<string, number> = {};
    const allTimeRate: Record<string, number> = {};
    const ranked = [...counties].sort((a, b) => b.flood_rate - a.flood_rate);
    ranked.forEach((c, i) => { rankByCounty[c.county] = i + 1; allTimeRate[c.county] = c.flood_rate; });

    return selected.map((c) => {
      const rows = records.filter((r) => r.county === c);
      const events = rows.reduce((s, r) => s + r.flood, 0);
      const rate = rows.length ? (events / rows.length) * 100 : 0;
      return { county: c, rate, events, months: rows.length, rank: rankByCounty[c] ?? null };
    }).sort((a, b) => b.rate - a.rate);
  }, [records, selected, counties]);

  // Running total of flood-months per county, year over year — complements the
  // annual RATE chart above (which resets every year) by showing accumulated
  // burden building up over the window, county by county.
  const cumulativeByCounty = useMemo(() => {
    const byYear: Record<number, any> = {};
    for (let y = yearRange[0]; y <= yearRange[1]; y++) byYear[y] = { year: y };
    const running: Record<string, number> = {};
    for (let y = yearRange[0]; y <= yearRange[1]; y++) {
      for (const c of selected) {
        const eventsThisYear = records.filter((r) => r.county === c && r.year === y).reduce((s, r) => s + r.flood, 0);
        running[c] = (running[c] ?? 0) + eventsThisYear;
        byYear[y][c] = running[c];
      }
    }
    return Object.values(byYear);
  }, [records, selected, yearRange]);

  // Per-county year x month calendar, small multiples — same visual language as
  // the national heatmap on Overview, scoped to whichever counties are selected
  // here and reactive to the same year/month filters as everything else on this page.
  const perCountyHeatmaps = useMemo(() => {
    const sortedMonths = [...months].sort((a, b) => a - b);
    return selected.slice(0, HEATMAP_CAP).map((c) => {
      const byYM: Record<string, number> = {};
      for (const r of records) if (r.county === c) byYM[`${r.year}-${r.month}`] = r.flood;
      return { county: c, sortedMonths, byYM };
    });
  }, [records, selected, months]);

  // Cross-county co-occurrence: Pearson correlation between counties' monthly
  // flood series over the current filter window. Positive = tend to flood in
  // the same months (shared hydrology/regional rainfall); negative = rarely
  // coincide. Capped to a readable matrix size.
  const coOccurrence = useMemo(() => {
    if (selected.length < COOCCURRENCE_MIN || selected.length > COOCCURRENCE_MAX) return null;
    const keys = Array.from(new Set(records.map((r) => `${r.year}-${r.month}`))).sort();
    if (keys.length === 0) return null;
    const vectors: Record<string, number[]> = {};
    for (const c of selected) {
      const byKey: Record<string, number> = {};
      for (const r of records) if (r.county === c) byKey[`${r.year}-${r.month}`] = r.flood;
      vectors[c] = keys.map((k) => byKey[k] ?? 0);
    }
    const matrix = selected.map((a) => selected.map((b) => pearson(vectors[a], vectors[b])));
    return { matrix };
  }, [records, selected]);

  const toggleMonth = (m: number) => {
    setMonths((ms) => (ms.includes(m) ? ms.filter((x) => x !== m) : [...ms, m]));
  };

  const chart = useChartTheme();
  const colors = useRiskColors();
  const showDataQualityNote = yearRange[1] >= 2022;

  const emptyNote = (msg: string) => <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={msg} />;

  return (
    <>
      <PageHeader
        eyebrow="2011–2025 Training Record"
        title="Historical Flood Record & Trends"
        subtitle="Filter by county, year range, and month — every chart below is driven from one selection."
      />

      {/* One filter panel drives the whole page. */}
      <Card style={{ marginBottom: 20 }}>
        <Row gutter={[24, 20]}>
          <Col xs={24} lg={10}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <Text
                type="secondary"
                style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}
              >
                Counties — {selected.length} of {counties.length}
              </Text>
              <Space size={4}>
                <Button type="link" size="small" onClick={() => setSelected(counties.map((c) => c.county))}>
                  All
                </Button>
                <Button type="link" size="small" onClick={() => setSelected([])}>
                  None
                </Button>
              </Space>
            </div>
            <Select
              mode="multiple"
              allowClear
              value={selected}
              onChange={setSelected}
              options={counties.map((c) => ({ value: c.county, label: c.county }))}
              placeholder="Search and select counties…"
              maxTagCount="responsive"
              style={{ width: "100%" }}
            />
          </Col>

          <Col xs={24} lg={7}>
            <Text
              type="secondary"
              style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}
            >
              Year Range
            </Text>
            <SliderRange
              range
              min={2011}
              max={2025}
              value={yearRange}
              onChange={(v) => setYearRange(v as [number, number])}
              marks={{ 2011: "2011", 2025: "2025" }}
            />
            {showDataQualityNote && (
              <Text style={{ fontSize: 11, color: colors.High, lineHeight: 1.5, display: "block", marginTop: 18 }}>
                <WarningOutlined /> 2022 onward may undercount (satellite data gap) — see the Data Quality note on any
                County Profile.
              </Text>
            )}
          </Col>

          <Col xs={24} lg={7}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <Text
                type="secondary"
                style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}
              >
                Months
              </Text>
              <Space size={4}>
                <Button type="link" size="small" onClick={() => setMonths([...Array(12)].map((_, i) => i + 1))}>
                  All
                </Button>
                <Button type="link" size="small" onClick={() => setMonths([])}>
                  None
                </Button>
              </Space>
            </div>
            <Space size={[4, 6]} wrap>
              {MONTH_SHORT.map((m, i) => (
                <Tag.CheckableTag
                  key={m}
                  checked={months.includes(i + 1)}
                  onChange={() => toggleMonth(i + 1)}
                  style={{ fontSize: 12, paddingInline: 8, border: "1px solid var(--color-border)" }}
                >
                  {m}
                </Tag.CheckableTag>
              ))}
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} xl={12}>
          <Card title="Annual Flood Rate — Selected Counties vs National" style={{ height: "100%" }}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={annualByCounty} margin={{ top: 6, right: 12, left: -14, bottom: 0 }}>
                <CartesianGrid stroke={chart.grid} vertical={false} />
                <XAxis dataKey="year" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
                <YAxis tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} unit="%" />
                <ReTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => `${n.toFixed(1)}%`)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="National" stroke={chart.axis} strokeDasharray="4 3" dot={false} />
                {selected.map((c, i) => (
                  <Line key={c} type="monotone" dataKey={c} stroke={PALETTE[i % PALETTE.length]} dot={{ r: 2 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title="Monthly Seasonality (Selected Counties)" style={{ height: "100%" }}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={seasonality} margin={{ top: 6, right: 12, left: -14, bottom: 0 }}>
                <CartesianGrid stroke={chart.grid} vertical={false} />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
                <YAxis tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} unit="%" />
                <ReTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => `${n.toFixed(1)}%`, "Flood rate")} />
                <Bar dataKey="rate" fill={chart.accent} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} xl={11}>
          <Card title={`Ranked Comparison — ${yearRange[0]}–${yearRange[1]}`} style={{ height: "100%" }}>
            {summary.length === 0 ? (
              emptyNote("Select at least one county.")
            ) : (
              <ResponsiveContainer width="100%" height={Math.max(160, summary.length * 34)}>
                <BarChart data={summary} layout="vertical" margin={{ left: 8, right: 12 }}>
                  <CartesianGrid stroke={chart.grid} horizontal={false} />
                  <XAxis type="number" unit="%" tick={{ fontSize: 10, fill: chart.axis }} stroke={chart.grid} />
                  <YAxis
                    type="category"
                    dataKey="county"
                    width={110}
                    tick={{ fontSize: 11, fill: chart.text }}
                    stroke={chart.grid}
                  />
                  <ReTooltip
                    formatter={tooltipValue((n) => `${n.toFixed(1)}%`, "Rate in window")}
                    contentStyle={chart.tooltip}
                  />
                  <Bar dataKey="rate" radius={[0, 4, 4, 0]} barSize={16}>
                    {summary.map((s, i) => (
                      <Cell
                        key={s.county}
                        fill={PALETTE[selected.indexOf(s.county) % PALETTE.length] ?? PALETTE[i % PALETTE.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>
        </Col>

        <Col xs={24} xl={13}>
          <Card title="Window Summary" style={{ height: "100%" }}>
            <Table
              dataSource={summary}
              rowKey="county"
              size="small"
              pagination={false}
              scroll={{ x: "max-content", y: 300 }}
              locale={{ emptyText: "No counties selected." }}
              columns={[
                { title: "County", dataIndex: "county", render: (c: string) => <Text strong>{c}</Text> },
                {
                  title: "Rate (window)",
                  dataIndex: "rate",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(1)}%</span>,
                },
                {
                  title: "Events",
                  dataIndex: "events",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v}</span>,
                },
                {
                  title: "All-time rank",
                  dataIndex: "rank",
                  align: "right",
                  render: (r: number | null) => (
                    <span className="tabular">{r ? `#${r} of ${counties.length}` : "—"}</span>
                  ),
                },
                {
                  title: "",
                  key: "link",
                  align: "right",
                  render: (_, s) => <Link to={`/county/${encodeURIComponent(s.county)}`}>Profile →</Link>,
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <Card title={`Cumulative Flood-Months — ${yearRange[0]}–${yearRange[1]}`} style={{ marginBottom: 16 }}>
        {selected.length === 0 ? (
          emptyNote("Select at least one county.")
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={cumulativeByCounty} margin={{ top: 6, right: 12, left: -20, bottom: 0 }}>
              <CartesianGrid stroke={chart.grid} vertical={false} />
              <XAxis dataKey="year" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
              <YAxis tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} allowDecimals={false} />
              <ReTooltip contentStyle={chart.tooltip} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {selected.map((c, i) => (
                <Line key={c} type="monotone" dataKey={c} stroke={PALETTE[i % PALETTE.length]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
        <Caption>
          Running total of recorded flood-months per county — a steepening line means events are accelerating, not
          just recurring at a steady rate.
        </Caption>
        <Explainer>
          Each line only ever goes up (it's a running total, never resets), so a straight diagonal means a county
          floods at a steady pace year after year, while a line that bends upward and gets steeper means flooding is
          happening more often lately than it used to.
        </Explainer>
      </Card>

      <Card
        title={`Monthly Calendar by County${selected.length > HEATMAP_CAP ? ` — first ${HEATMAP_CAP} of ${selected.length} selected` : ""}`}
        style={{ marginBottom: 16 }}
      >
        {perCountyHeatmaps.length === 0 ? (
          emptyNote("Select at least one county.")
        ) : (
          <Row gutter={[20, 20]}>
            {perCountyHeatmaps.map(({ county, sortedMonths, byYM }) => (
              <Col key={county} xs={24} sm={12} lg={8} xxl={6}>
                <Text strong style={{ fontSize: 13, display: "block", marginBottom: 8 }}>
                  {county}
                </Text>
                <div className="scroll-x">
                  <table style={{ borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th />
                        {sortedMonths.map((m) => (
                          <th
                            key={m}
                            style={{ fontSize: 9, fontWeight: 500, color: chart.axis, padding: "0 2px" }}
                          >
                            {MONTH_SHORT[m - 1]}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: yearRange[1] - yearRange[0] + 1 }, (_, i) => yearRange[0] + i).map((y) => (
                        <tr key={y}>
                          <td className="tabular" style={{ fontSize: 10, color: chart.axis, padding: "1px 4px 1px 0" }}>
                            {y}
                          </td>
                          {sortedMonths.map((m) => {
                            const v = byYM[`${y}-${m}`];
                            return (
                              <td key={m}>
                                <Tooltip
                                  title={`${county} ${MONTH_SHORT[m - 1]} ${y}: ${v != null ? (v ? "flood" : "no flood") : "no data"}`}
                                >
                                  <div
                                    style={{
                                      width: 13,
                                      height: 12,
                                      borderRadius: 2,
                                      margin: 1,
                                      background: v != null ? heatColor(v) : "var(--color-bg)",
                                      border: "1px solid rgba(16,24,40,0.06)",
                                    }}
                                  />
                                </Tooltip>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Col>
            ))}
          </Row>
        )}
        <Explainer>
          Each small square is one month. Darker/redder means a flood was recorded that county-month; pale means no
          data yet. Reading across a row lets you spot a county's own seasonal pattern (e.g. always red in
          August–October); reading down a column shows whether a bad year hit every selected county or just one.
        </Explainer>
      </Card>

      <Card title="Cross-County Co-occurrence — Do These Counties Flood Together?">
        {selected.length < COOCCURRENCE_MIN && emptyNote("Select at least 2 counties.")}
        {selected.length > COOCCURRENCE_MAX &&
          emptyNote(
            `Select ${COOCCURRENCE_MAX} or fewer counties to see a readable co-occurrence matrix (${selected.length} selected).`
          )}
        {coOccurrence && (
          <>
            <div className="scroll-x">
              <table style={{ borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th />
                    {selected.map((c) => (
                      <th
                        key={c}
                        title={c}
                        style={{
                          fontSize: 10,
                          fontWeight: 500,
                          color: chart.axis,
                          padding: "0 3px",
                          maxWidth: 46,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {c.length > 6 ? `${c.slice(0, 5)}…` : c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selected.map((rowC, ri) => (
                    <tr key={rowC}>
                      <td style={{ fontSize: 12, fontWeight: 600, paddingRight: 8, whiteSpace: "nowrap" }}>{rowC}</td>
                      {selected.map((colC, ci) => {
                        const r = coOccurrence.matrix[ri][ci];
                        return (
                          <td key={colC}>
                            <Tooltip title={`${rowC} vs ${colC}: r = ${r.toFixed(2)}`}>
                              <div
                                className="tabular"
                                style={{
                                  width: 40,
                                  height: 28,
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  background: corrColor(ri === ci ? 0 : r),
                                  borderRadius: 3,
                                  margin: 1,
                                  fontSize: 10,
                                  fontWeight: 600,
                                  color: Math.abs(r) > 0.55 ? "#ffffff" : "#4b5563",
                                }}
                              >
                                {ri === ci ? "—" : r.toFixed(2)}
                              </div>
                            </Tooltip>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Caption>
              Pearson correlation between counties' monthly flood series in the current window. Red = tend to flood
              together (shared rainfall/hydrology); blue = rarely coincide. Diagonal omitted (always self-correlated).
            </Caption>
            <Explainer>
              Pick any two counties: the number where their row and column meet ranges from -1 to +1. Close to +1
              means when one floods, the other usually does too in the same month (they likely share a river basin or
              a regional storm system). Close to -1 would mean they almost never flood together. Close to 0 means no
              real relationship either way.
            </Explainer>
          </>
        )}
      </Card>
    </>
  );
}
