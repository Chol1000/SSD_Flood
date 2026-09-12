import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Card,
  Typography,
  Space,
  Button,
  Input,
  Modal,
  Badge,
  Skeleton,
  Divider,
  Row,
  Col,
  Segmented,
  Empty,
  Tooltip,
} from "antd";
import { SearchOutlined, FileProtectOutlined } from "@ant-design/icons";
import { api } from "../api";
import type { RiskTier, ScanResult, CountyListItem } from "../types";
import RecommendationView from "../components/RecommendationView";
import { PageHeader, RiskTag, useRiskColors, RISK_ORDER } from "../ui";

const { Text, Title, Paragraph } = Typography;

const SECTION_NOTE: Record<RiskTier, string> = {
  Critical: "Immediate action recommended for all counties in this tier.",
  High: "Elevated risk — activate response readiness.",
  Moderate: "Above baseline — preparedness review recommended.",
  Low: "",
};
const REFRESH_SECONDS = 300;

export default function Alerts() {
  const [live, setLive] = useState<ScanResult[] | null>(null);
  const [issuedAt, setIssuedAt] = useState<Date | null>(null);
  const [counties, setCounties] = useState<CountyListItem[]>([]);
  const [search, setSearch] = useState("");
  const [lowView, setLowView] = useState<"list" | "probabilities">("list");
  const [countdown, setCountdown] = useState(REFRESH_SECONDS);
  const [recCounty, setRecCounty] = useState<ScanResult | null>(null);

  const colors = useRiskColors();

  useEffect(() => {
    api.counties().then(setCounties).catch(console.error);
  }, []);

  useEffect(() => {
    const load = () => {
      api
        .scan({ month: new Date().getMonth() + 1, use_live: true })
        .then((r) => {
          setLive(r);
          setIssuedAt(new Date());
          setCountdown(REFRESH_SECONDS);
        })
        .catch(console.error);
    };
    load();
    const t = setInterval(load, REFRESH_SECONDS * 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setCountdown((c) => (c <= 1 ? REFRESH_SECONDS : c - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  const histByCounty = useMemo(() => {
    const m: Record<string, RiskTier> = {};
    for (const c of counties) m[c.county] = c.risk_tier;
    return m;
  }, [counties]);

  const rankByCounty = useMemo(() => {
    const ranked = [...counties].sort((a, b) => b.flood_rate - a.flood_rate);
    const m: Record<string, number> = {};
    ranked.forEach((c, i) => {
      m[c.county] = i + 1;
    });
    return m;
  }, [counties]);

  const byTier = useMemo(() => {
    const groups: Record<RiskTier, ScanResult[]> = { Critical: [], High: [], Moderate: [], Low: [] };
    if (live) {
      for (const r of live) groups[r.risk_tier].push(r);
      for (const tier of RISK_ORDER) groups[tier].sort((a, b) => b.probability - a.probability);
    }
    return groups;
  }, [live]);

  const filteredLow = useMemo(() => {
    if (!search) return byTier.Low;
    return byTier.Low.filter((r) => r.county.toLowerCase().includes(search.toLowerCase()));
  }, [byTier, search]);

  const total = live?.length ?? 0;
  const activeAlerts = total - byTier.Low.length;
  const urgent = [...byTier.Critical, ...byTier.High].slice(0, 3);
  const maxUrgentPct = urgent.length ? Math.round(Math.max(...urgent.map((u) => u.probability)) * 100) : 0;

  const summarySentence = (() => {
    if (!live || !issuedAt) return "Assessing current national flood risk from live climate data…";
    const dateStr = issuedAt.toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });
    if (activeAlerts === 0) {
      return `As of ${dateStr}, all ${total} monitored counties are within normal seasonal flood risk parameters based on live climate conditions. No advisories are currently active.`;
    }
    const names = urgent.map((u) => u.county);
    const nameList =
      names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : names[0];
    return (
      `As of ${dateStr}, ${activeAlerts} of ${total} monitored counties show elevated flood risk under the deployed nowcast model. ` +
      `${nameList} ${names.length > 1 ? "require" : "requires"} the closest attention, with live flood probabilities reaching ${maxUrgentPct}%. ` +
      `The remaining ${byTier.Low.length} counties remain within normal seasonal parameters.`
    );
  })();

  return (
    <div style={{ maxWidth: 1040 }}>
      <PageHeader
        eyebrow="South Sudan Flood Early Warning System"
        title="National Early Warning Bulletin"
        subtitle={
          issuedAt ? (
            <Space size={8} wrap>
              <Badge status="processing" />
              <span>
                Issued{" "}
                {issuedAt.toLocaleDateString(undefined, {
                  weekday: "long",
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                })}{" "}
                at {issuedAt.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
              </span>
              <span className="tabular">
                · next refresh in {Math.floor(countdown / 60)}:{String(countdown % 60).padStart(2, "0")}
              </span>
            </Space>
          ) : (
            "Loading bulletin…"
          )
        }
      />

      {/* Executive summary — written prose, the way a real bulletin opens. */}
      <Card style={{ marginBottom: 20 }} styles={{ body: { padding: "20px 24px" } }}>
        <Paragraph style={{ fontSize: 16, lineHeight: 1.7, margin: 0 }}>{summarySentence}</Paragraph>
      </Card>

      {/* Severity meter — one segmented bar rather than four generic tiles. */}
      <Card
        style={{ marginBottom: 24 }}
        title={
          <Text
            type="secondary"
            style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
          >
            National Risk Composition
          </Text>
        }
        extra={
          <Text type="secondary" style={{ fontSize: 12 }}>
            {total} counties assessed
          </Text>
        }
      >
        <div
          style={{
            display: "flex",
            height: 14,
            borderRadius: 7,
            overflow: "hidden",
            background: "var(--color-border)",
          }}
        >
          {RISK_ORDER.map((tier) => {
            const n = byTier[tier].length;
            const pct = total ? (n / total) * 100 : 0;
            return pct > 0 ? (
              <Tooltip key={tier} title={`${tier}: ${n} ${n === 1 ? "county" : "counties"}`}>
                <div style={{ width: `${pct}%`, background: colors[tier] }} />
              </Tooltip>
            ) : null;
          })}
        </div>
        <Row gutter={[16, 8]} style={{ marginTop: 14 }} className="compact-stat-row">
          {RISK_ORDER.map((tier) => (
            <Col key={tier} xs={12} sm={6}>
              <Space size={6}>
                <span
                  style={{ display: "block", width: 9, height: 9, borderRadius: 2, background: colors[tier] }}
                />
                <Text type="secondary" style={{ fontSize: 13 }}>
                  {tier}
                </Text>
                <b className="tabular">{live ? byTier[tier].length : "–"}</b>
              </Space>
            </Col>
          ))}
        </Row>
      </Card>

      {!live && <Skeleton active paragraph={{ rows: 6 }} />}

      {/* Advisory sections — only tiers that actually have something to report. */}
      {live &&
        (["Critical", "High", "Moderate"] as RiskTier[]).map((tier) => {
          const entries = byTier[tier];
          if (entries.length === 0) return null;
          return (
            <div key={tier} style={{ marginBottom: 24 }}>
              <Space
                align="baseline"
                wrap
                style={{
                  width: "100%",
                  justifyContent: "space-between",
                  borderBottom: `2px solid ${colors[tier]}`,
                  paddingBottom: 6,
                  marginBottom: 12,
                }}
              >
                <Space align="baseline" size={8}>
                  <span
                    style={{
                      display: "block",
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: colors[tier],
                    }}
                  />
                  <Title level={5} style={{ margin: 0, color: colors[tier] }}>
                    {tier} Advisories
                  </Title>
                  <Text type="secondary">({entries.length})</Text>
                </Space>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {SECTION_NOTE[tier]}
                </Text>
              </Space>

              <Card styles={{ body: { padding: 0 } }}>
                {entries.map((r, i) => {
                  const hist = histByCounty[r.county];
                  return (
                    <div
                      key={r.county}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: 12,
                        flexWrap: "wrap",
                        padding: "12px 18px",
                        borderBottom: i < entries.length - 1 ? "1px solid var(--color-border)" : "none",
                      }}
                    >
                      <Space size={10} wrap style={{ minWidth: 0 }}>
                        <span
                          style={{
                            display: "block",
                            width: 7,
                            height: 7,
                            borderRadius: "50%",
                            background: colors[r.risk_tier],
                          }}
                        />
                        <Link to={`/prediction/${encodeURIComponent(r.county)}`}>
                          <Text strong style={{ fontSize: 15 }}>
                            {r.county}
                          </Text>
                        </Link>
                        {hist && hist !== r.risk_tier && (
                          <Space size={4}>
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              was
                            </Text>
                            <RiskTag tier={hist} dot />
                          </Space>
                        )}
                      </Space>

                      <Space size={16}>
                        <div style={{ textAlign: "right" }}>
                          <div
                            className="tabular"
                            style={{ fontSize: 18, fontWeight: 800, color: colors[r.risk_tier], lineHeight: 1.2 }}
                          >
                            {(r.probability * 100).toFixed(0)}%
                          </div>
                          <Text type="secondary" style={{ fontSize: 10 }}>
                            flood probability
                          </Text>
                        </div>
                        <Button type="link" size="small" onClick={() => setRecCounty(r)} style={{ padding: 0 }}>
                          Recommendations →
                        </Button>
                      </Space>
                    </div>
                  );
                })}
              </Card>
            </div>
          );
        })}

      {/* Low-risk appendix — a written list, not 74 repeated "all clear" cards. */}
      {live && (
        <div style={{ marginBottom: 24 }}>
          <Space
            align="baseline"
            wrap
            style={{
              width: "100%",
              justifyContent: "space-between",
              borderBottom: "2px solid var(--color-border)",
              paddingBottom: 6,
              marginBottom: 12,
            }}
          >
            <Space align="baseline" size={8}>
              <span
                style={{ display: "block", width: 10, height: 10, borderRadius: "50%", background: colors.Low }}
              />
              <Title level={5} style={{ margin: 0 }}>
                Low-Risk Counties
              </Title>
              <Text type="secondary">({byTier.Low.length})</Text>
            </Space>
            <Segmented
              size="small"
              value={lowView}
              onChange={(v) => setLowView(v as "list" | "probabilities")}
              options={[
                { label: "List", value: "list" },
                { label: "Probabilities", value: "probabilities" },
              ]}
            />
          </Space>

          <Card>
            <Input
              allowClear
              prefix={<SearchOutlined style={{ opacity: 0.45 }} />}
              placeholder="Search within low-risk counties…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ width: 260, marginBottom: 16 }}
            />

            {filteredLow.length === 0 && (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No matching low-risk county." />
            )}

            {filteredLow.length > 0 && lowView === "list" && (
              <Paragraph style={{ fontSize: 14, lineHeight: 2.1, margin: 0 }}>
                {filteredLow.map((r, i) => (
                  <span key={r.county}>
                    <Link to={`/prediction/${encodeURIComponent(r.county)}`} style={{ color: "var(--color-text)" }}>
                      {r.county}
                    </Link>
                    {i < filteredLow.length - 1 ? ", " : "."}
                  </span>
                ))}
              </Paragraph>
            )}

            {filteredLow.length > 0 && lowView === "probabilities" && (
              <Row gutter={[16, 4]}>
                {filteredLow.map((r) => (
                  <Col key={r.county} xs={12} sm={8} lg={6}>
                    <Link
                      to={`/prediction/${encodeURIComponent(r.county)}`}
                      style={{ display: "flex", justifyContent: "space-between", gap: 8, padding: "3px 0" }}
                    >
                      <Text style={{ fontSize: 13 }}>{r.county}</Text>
                      <Text type="secondary" strong className="tabular" style={{ fontSize: 13 }}>
                        {(r.probability * 100).toFixed(0)}%
                      </Text>
                    </Link>
                  </Col>
                ))}
              </Row>
            )}
          </Card>
        </div>
      )}

      {/* Methodology footer — reads like a document, not a UI hint. */}
      <Divider />
      <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.8, display: "block" }}>
        <b>Methodology.</b> Advisories are generated from the deployed nowcast model (see{" "}
        <Link to="/model">Model &amp; Validation</Link>) using live climate inputs aggregated for the current month,
        refreshed automatically every five minutes. Historical baseline tiers reflect each county's 2011–2025 flood
        record and may undercount 2022 onward — see the Data Quality note on any County Profile.
        <br />
        <b>Intended audience.</b> County administrations, national disaster management authorities, and humanitarian
        response coordinators. This bulletin is a decision-support tool and is not a substitute for official government
        flood advisories.
      </Text>

      <Modal
        open={!!recCounty}
        onCancel={() => setRecCounty(null)}
        footer={null}
        width={620}
        destroyOnHidden
        title={
          <Space>
            <FileProtectOutlined />
            <span>Recommendation — {recCounty?.county}</span>
          </Space>
        }
      >
        {recCounty && (
          <RecommendationView
            county={recCounty.county}
            tier={recCounty.risk_tier}
            probability={recCounty.probability}
            historicalTier={histByCounty[recCounty.county]}
            historicalRank={rankByCounty[recCounty.county]}
            nCounties={counties.length}
          />
        )}
      </Modal>
    </div>
  );
}
