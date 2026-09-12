import { useEffect, useMemo, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Card, Switch, Typography, Space, Spin, Tag } from "antd";
import { api } from "../api";
import type { ScanResult } from "../types";
import MapView, { type MapPoint } from "../components/MapView";
import { useRiskColors, RISK_ORDER } from "../ui";

const { Text } = Typography;

const TODAY = new Date();
const CURRENT_MONTH = TODAY.getMonth() + 1;
const CURRENT_YEAR = TODAY.getFullYear();
const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

export default function MapPage() {
  const navigate = useNavigate();
  const [useLive, setUseLive] = useState(true);
  const [scan, setScan] = useState<ScanResult[]>([]);
  const [scanLoading, setScanLoading] = useState(false);
  const colors = useRiskColors();

  useEffect(() => {
    setScanLoading(true);
    api
      .scan({ month: CURRENT_MONTH, use_live: useLive })
      .then(setScan)
      .catch(console.error)
      .finally(() => setScanLoading(false));
  }, [useLive]);

  const mapPoints: MapPoint[] = useMemo(
    () => scan.map((s) => ({ county: s.county, lat: s.lat, lon: s.lon, value: s.probability, tier: s.risk_tier })),
    [scan]
  );

  const onSelect = useCallback(
    (c: string) => {
      navigate(`/prediction/${encodeURIComponent(c)}`);
    },
    [navigate]
  );

  const tierCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const s of scan) counts[s.risk_tier] = (counts[s.risk_tier] ?? 0) + 1;
    return counts;
  }, [scan]);

  return (
    <Card
      styles={{ body: { padding: 0, height: "100%" } }}
      style={{ overflow: "hidden", position: "relative", height: "calc(100vh - 160px)", minHeight: 460 }}
    >
      <MapView points={mapPoints} selected={null} onSelect={onSelect} />

      {/* Controls float over the map rather than pushing it down, so the map
          keeps the full height of the page on a laptop screen. */}
      <Card
        size="small"
        style={{ position: "absolute", top: 14, insetInlineStart: 14, maxWidth: 340, zIndex: 1 }}
        styles={{ body: { padding: "12px 14px" } }}
      >
        <Text
          type="secondary"
          style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase" }}
        >
          National Risk Map
        </Text>
        <div style={{ fontSize: 15, fontWeight: 600, marginTop: 2 }}>
          {MONTHS[CURRENT_MONTH - 1]} {CURRENT_YEAR}{" "}
          <Text style={{ color: "var(--color-primary)", fontWeight: 500, fontSize: 13 }}>(today)</Text>
        </div>

        <Space size={8} style={{ marginTop: 10 }}>
          <Switch size="small" checked={useLive} onChange={setUseLive} />
          <Text style={{ fontSize: 13 }}>Live climate data</Text>
          {scanLoading && <Spin size="small" />}
        </Space>

        <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 8, lineHeight: 1.5 }}>
          Hover a county for details, click to open its prediction.
        </Text>
      </Card>

      <Card
        size="small"
        style={{ position: "absolute", bottom: 14, insetInlineStart: 14, zIndex: 1 }}
        styles={{ body: { padding: "10px 14px" } }}
      >
        <Space size={12} wrap>
          {[...RISK_ORDER].reverse().map((t) => (
            <Space key={t} size={5}>
              <span style={{ display: "block", width: 9, height: 9, borderRadius: "50%", background: colors[t] }} />
              <Text style={{ fontSize: 12 }}>{t}</Text>
              {tierCounts[t] != null && (
                <Tag style={{ marginInlineEnd: 0, fontSize: 11, lineHeight: "16px" }}>{tierCounts[t]}</Tag>
              )}
            </Space>
          ))}
        </Space>
      </Card>
    </Card>
  );
}
