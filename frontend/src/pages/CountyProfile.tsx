import { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  Card, Row, Col, Table, Button, Dropdown, Typography, Space, Skeleton, Breadcrumb, Descriptions, Alert, Tooltip,
} from "antd";
import {
  DownloadOutlined, DownOutlined, FileExcelOutlined, CodeOutlined, ThunderboltOutlined,
} from "@ant-design/icons";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer, Legend } from "recharts";
import { api } from "../api";
import type { CountyDetail, CountyListItem, RiskTier } from "../types";
import MapView from "../components/MapView";
import ReportGenerator from "../components/ReportGenerator";
import Sparkline from "../components/Sparkline";
import { PageHeader, StatCard, RiskTag, Caption, useRiskColors, useChartTheme, tooltipValue } from "../ui";

const { Text, Title, Paragraph } = Typography;

const RISK_POLL_SECONDS = 60;

function hrTier(rate: number): RiskTier {
  if (rate >= 0.12) return "Critical";
  if (rate >= 0.06) return "High";
  if (rate >= 0.03) return "Moderate";
  return "Low";
}

const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

const FEATURE_LABELS: Record<string, string> = {
  rainfall_mm: "Rainfall", soil_moisture_mm: "Soil Moisture",
  max_temperature_celsius: "Max Temperature", min_temperature_celsius: "Min Temperature",
  vapor_pressure_deficit_kPa: "Vapour Pressure Deficit", wetland_fraction: "Wetland Fraction",
  elevation_m: "Elevation", slope_deg: "Slope", ndvi: "NDVI (Vegetation)",
  flood_count_last_12mo: "Floods, Trailing 12mo (median)",
};
const FEATURE_UNITS: Record<string, string> = {
  rainfall_mm: "mm", soil_moisture_mm: "mm", max_temperature_celsius: "°C", min_temperature_celsius: "°C",
  vapor_pressure_deficit_kPa: "kPa", wetland_fraction: "", elevation_m: "m", slope_deg: "°", ndvi: "", flood_count_last_12mo: "",
};

function heatColor(rate: number): string {
  if (rate <= 0) return "#EEF1F5";
  const t = Math.min(1, rate / 0.5);
  const stops: [number, string][] = [[0, "#EAF1FB"], [0.35, "#F5C97A"], [0.7, "#E0722E"], [1, "#B91C1C"]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      const pa = [1, 3, 5].map((i2) => parseInt(c0.slice(i2, i2 + 2), 16));
      const pb = [1, 3, 5].map((i2) => parseInt(c1.slice(i2, i2 + 2), 16));
      const p = pa.map((v, i2) => Math.round(v + (pb[i2] - v) * f));
      return `rgb(${p[0]},${p[1]},${p[2]})`;
    }
  }
  return stops[stops.length - 1][1];
}

function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function CountyProfile() {
  const { county = "" } = useParams();
  const [detail, setDetail] = useState<CountyDetail | null>(null);
  const [counties, setCounties] = useState<CountyListItem[]>([]);
  const [records, setRecords] = useState<any[]>([]);
  const [national, setNational] = useState<{ year: number; flood: number }[]>([]);
  const [riskHistory, setRiskHistory] = useState<{ t: number; probability: number; tier: RiskTier }[]>([]);

  useEffect(() => {
    // Clear immediately — otherwise the map (keyed to `county`) remounts
    // fresh while `detail` still holds the PREVIOUS county's coordinates,
    // flying to the wrong place under the new county's name.
    setDetail(null);
    api.countyDetail(county).then(setDetail).catch(console.error);
    api.counties().then(setCounties).catch(console.error);
  }, [county]);

  // Live flood-risk trend for this county — computed server-side and shared
  // across every client (see /api/risk-trend), so it's already flowing, hours
  // deep, the moment this profile opens rather than starting from empty.
  useEffect(() => {
    if (!county) return;
    let cancelled = false;
    const load = () => {
      api.riskTrend().then((rows) => {
        if (cancelled) return;
        const row = rows.find((r) => r.county === county);
        if (!row) return;
        setRiskHistory(row.history.map((h) => ({ t: h.t * 1000, probability: h.probability, tier: h.tier })));
      }).catch(console.error);
    };
    load();
    const t = setInterval(load, RISK_POLL_SECONDS * 1000);
    return () => { cancelled = true; clearInterval(t); };
  }, [county]);

  useEffect(() => {
    if (!county) return;
    api.historical([county], 2011, 2025, Array.from({ length: 12 }, (_, i) => i + 1)).then((res) => {
      setRecords(res.records);
      setNational(res.national);
    });
  }, [county]);

  const annual = useMemo(() => {
    const byY: Record<number, number[]> = {};
    for (const r of records) (byY[r.year] ??= []).push(r.flood);
    const nat: Record<number, number> = {};
    for (const n of national) nat[n.year] = n.flood * 100;
    return Array.from({ length: 2025 - 2011 + 1 }, (_, i) => 2011 + i).map((y) => ({
      year: y,
      rate: byY[y] ? (byY[y].reduce((a, b) => a + b, 0) / byY[y].length) * 100 : 0,
      national: nat[y] ?? 0,
      dataQualityFlag: y >= 2022,
    }));
  }, [records, national]);

  const seasonality = useMemo(() => {
    const byM: Record<number, number[]> = {};
    for (const r of records) (byM[r.month] ??= []).push(r.flood);
    return MONTH_SHORT.map((m, i) => ({
      month: m, monthFull: MONTH_FULL[i],
      rate: byM[i + 1] ? (byM[i + 1].reduce((a, b) => a + b, 0) / byM[i + 1].length) : 0,
    }));
  }, [records]);

  const totalEvents = useMemo(() => records.reduce((s, r) => s + r.flood, 0), [records]);

  const similar = useMemo(() => {
    if (!detail) return [];
    return [...counties]
      .filter((c) => c.county !== county)
      .sort((a, b) => Math.abs(a.flood_rate - detail.flood_rate) - Math.abs(b.flood_rate - detail.flood_rate))
      .slice(0, 6);
  }, [counties, detail, county]);

  const percentile = useMemo(() => {
    if (!detail || counties.length === 0) return null;
    const below = counties.filter((c) => c.flood_rate <= detail.flood_rate).length;
    return Math.round((below / counties.length) * 100);
  }, [counties, detail]);

  const downloadCSV = () => {
    const rows = ["county,year,month,flood"];
    for (const r of records) rows.push(`${r.county},${r.year},${r.month},${r.flood}`);
    downloadBlob(`${county}_monthly_record.csv`, rows.join("\n"), "text/csv");
  };
  const downloadJSON = () => {
    downloadBlob(`${county}_profile.json`, JSON.stringify({ county, detail, seasonality, annual, similar }, null, 2), "application/json");
  };

  const colors = useRiskColors();
  const chart = useChartTheme();

  if (!detail) {
    return (
      <>
        <Skeleton active paragraph={{ rows: 2 }} style={{ marginBottom: 24 }} />
        <Skeleton active paragraph={{ rows: 6 }} />
      </>
    );
  }

  const tier = hrTier(detail.flood_rate);
  const latest = riskHistory.at(-1);

  return (
    <>
      <Breadcrumb
        style={{ marginBottom: 12 }}
        items={[{ title: <Link to="/">Overview</Link> }, { title: county }]}
      />

      <PageHeader
        eyebrow="County Risk Profile"
        title={county}
        subtitle={
          <Space size={8} wrap>
            <RiskTag tier={tier} />
            <span>historical risk · ranked #{detail.rank} of {detail.n_counties} nationally</span>
          </Space>
        }
        extra={
          <Space wrap>
            <Link to={`/prediction/${encodeURIComponent(county)}`}>
              <Button type="primary" ghost icon={<ThunderboltOutlined />}>
                Live Prediction
              </Button>
            </Link>
            <Dropdown
              menu={{
                items: [
                  { key: "csv", icon: <FileExcelOutlined />, label: "CSV — full monthly record", onClick: downloadCSV },
                  { key: "json", icon: <CodeOutlined />, label: "JSON — full profile", onClick: downloadJSON },
                ],
              }}
            >
              <Button icon={<DownloadOutlined />}>
                Export Data <DownOutlined style={{ fontSize: 10 }} />
              </Button>
            </Dropdown>
            <ReportGenerator county={county} />
          </Space>
        }
      />

      {/* Locator + the four headline historical numbers. */}
      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        <Col xs={24} md={8} xl={6}>
          <Card styles={{ body: { padding: 0, height: "100%" } }} style={{ overflow: "hidden", height: 200 }}>
            {detail.lat != null && (
              <MapView
                key={county}
                points={[{ county, lat: detail.lat, lon: detail.lon, value: detail.flood_rate, tier }]}
                selected={county}
                onSelect={() => {}}
                focusZoom={9.2}
              />
            )}
          </Card>
        </Col>
        <Col xs={24} md={16} xl={18}>
          <Row gutter={[16, 16]}>
            <Col xs={12} xl={6}>
              <StatCard
                label="Historical Flood Rate"
                value={detail.flood_rate * 100}
                precision={1}
                suffix="%"
                hint="Share of months in the record where this county registered a flood."
              />
            </Col>
            <Col xs={12} xl={6}>
              <StatCard label="Flood Events" value={detail.flood_events} suffix=" / 180 mo" />
            </Col>
            <Col xs={12} xl={6}>
              <StatCard
                label="National Percentile"
                value={percentile != null ? `${percentile}th` : "—"}
                hint="Percentage of counties with an equal or lower historical flood rate."
              />
            </Col>
            <Col xs={12} xl={6}>
              <StatCard label="National Rank" value={`#${detail.rank}`} suffix={` of ${detail.n_counties}`} />
            </Col>
          </Row>
        </Col>
      </Row>

      {/* Live nowcast reading, polled while this profile is open. */}
      <Card
        style={{ marginBottom: 24 }}
        title={
          <Text
            type="secondary"
            style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
          >
            Live Flood Risk Trend{" "}
            <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(deployed nowcast model)</span>
          </Text>
        }
        extra={
          <Link to={`/prediction/${encodeURIComponent(county)}`}>Full prediction &amp; recommendations →</Link>
        }
      >
        {!latest ? (
          <Skeleton active paragraph={{ rows: 1 }} title={false} />
        ) : (
          <Space size={16} wrap style={{ width: "100%" }}>
            <span
              className="tabular"
              style={{ fontSize: 32, fontWeight: 800, color: colors[latest.tier], lineHeight: 1 }}
            >
              {(latest.probability * 100).toFixed(1)}%
            </span>
            <RiskTag tier={latest.tier} />
            <Sparkline values={riskHistory.map((r) => r.probability * 100)} width={140} height={30} />
            <Text type="secondary" style={{ fontSize: 11 }}>
              updates every {RISK_POLL_SECONDS}s while this page is open
            </Text>
          </Space>
        )}
      </Card>

      <SectionHeading>Historical Flood Record</SectionHeading>

      <Card title={`Annual Flood Rate vs. National Average (2011–2025)`} style={{ marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={annual} margin={{ top: 6, right: 12, left: -14, bottom: 0 }}>
            <CartesianGrid stroke={chart.grid} vertical={false} />
            <XAxis dataKey="year" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} />
            <YAxis tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} unit="%" />
            <ReTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => `${n.toFixed(1)}%`)} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="national"
              name="National average"
              stroke={chart.axis}
              strokeDasharray="4 3"
              dot={false}
            />
            <Line type="monotone" dataKey="rate" name={county} stroke={chart.accent} strokeWidth={2.2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 12 }}
          title="2022–2025 coincide with a known gap in the underlying satellite water-extent source — read the Data Quality note below before comparing those years directly against 2011–2021."
        />
      </Card>

      <Card title={`Monthly Seasonality — ${totalEvents} recorded events, 2011–2025`} style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {seasonality.map((s) => (
            <Tooltip key={s.month} title={`${s.monthFull}: ${(s.rate * 100).toFixed(1)}% of months flooded`}>
              <div style={{ flex: "1 1 56px", textAlign: "center", minWidth: 44 }}>
                <div
                  style={{
                    height: 56,
                    borderRadius: 6,
                    background: heatColor(s.rate),
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 13,
                    fontWeight: 700,
                    color: s.rate > 0.25 ? "#ffffff" : "#1f2937",
                    border: "1px solid rgba(16,24,40,0.06)",
                  }}
                >
                  {(s.rate * 100).toFixed(0)}%
                </div>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {s.month}
                </Text>
              </div>
            </Tooltip>
          ))}
        </div>
        {totalEvents === 0 && (
          <Caption>
            No flood events recorded for {county} in the 2011–2025 dataset — this reflects the recorded data (see the
            Data Quality note below), not necessarily zero real flood risk.
          </Caption>
        )}
      </Card>

      <SectionHeading>Climate, Terrain &amp; Comparative Context</SectionHeading>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Historical Climate &amp; Terrain Medians" style={{ height: "100%" }}>
            <Descriptions column={1} size="small" bordered>
              {Object.entries(FEATURE_LABELS).map(([key, label]) => (
                <Descriptions.Item key={key} label={label}>
                  <span className="tabular">
                    {detail.defaults[key] != null
                      ? `${detail.defaults[key].toFixed(key.includes("fraction") || key === "ndvi" ? 3 : 1)} ${FEATURE_UNITS[key]}`.trim()
                      : "—"}
                  </span>
                </Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Counties with Similar Historical Risk" style={{ height: "100%" }}>
            <Table
              dataSource={similar}
              rowKey="county"
              size="small"
              pagination={false}
              scroll={{ x: "max-content" }}
              columns={[
                {
                  title: "County",
                  dataIndex: "county",
                  render: (c: string) => <Link to={`/county/${encodeURIComponent(c)}`}>{c}</Link>,
                },
                {
                  title: "Rate",
                  dataIndex: "flood_rate",
                  align: "right",
                  render: (v: number) => <span className="tabular">{(v * 100).toFixed(1)}%</span>,
                },
                {
                  title: "Risk",
                  dataIndex: "risk_tier",
                  align: "right",
                  render: (t: RiskTier) => <RiskTag tier={t} />,
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <SectionHeading>Data Quality &amp; Methodology</SectionHeading>

      <Card style={{ marginBottom: 24 }}>
        <Paragraph style={{ fontSize: 13, lineHeight: 1.8 }}>
          <b>Source period:</b> 2011–2025, county-month observations derived from satellite climate and terrain
          products (CHIRPS, TerraClimate, MODIS, SRTM, ESA WorldCover, JRC Global Surface Water). This is the
          training/historical record only — live predictions forecast forward from today, not from this window.
        </Paragraph>
        <Paragraph style={{ fontSize: 13, lineHeight: 1.8 }}>
          <b>Known data quality gap, 2022–2025:</b> the satellite water-extent signal underlying the flood label has a
          growing rate of missing values in these years (roughly 4.6% missing in 2022 rising to 7.9% by 2025, versus
          0% missing every year from 2011–2021), while rainfall and other climate variables show no corresponding
          anomaly. This is consistent with a processing-lag gap in the satellite source rather than a genuine drop in
          flooding, and means flood counts for 2022–2025 above likely <i>undercount</i> real events. Treat direct
          year-over-year comparisons involving 2022–2025 with caution until re-verified against an updated source.
        </Paragraph>
        <Paragraph style={{ fontSize: 13, lineHeight: 1.8, marginBottom: 0 }}>
          <b>Label definition:</b> a county-month is recorded as a flood event when satellite-derived{" "}
          <Text code>water_fraction</Text> ≥ 0.01. That field is excluded from the predictive model itself (it
          perfectly encodes the label), but the raw variable is what drives whether a historical month counts as a
          flood above.
        </Paragraph>
      </Card>
    </>
  );
}

/** Section rule used to break this profile into its three documented parts. */
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <Title
      level={5}
      style={{
        margin: "0 0 12px",
        paddingBottom: 8,
        borderBottom: "2px solid var(--color-text)",
      }}
    >
      {children}
    </Title>
  );
}
