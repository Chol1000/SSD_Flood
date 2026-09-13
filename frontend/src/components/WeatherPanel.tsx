import { useEffect, useState } from "react";
import { Card, Typography, Space, Badge, Skeleton, Row, Col, Tooltip, Empty } from "antd";
import { api } from "../api";
import type { LiveWeather } from "../types";
import WeatherIcon from "./WeatherIcon";
import { Caption } from "../ui";

const { Text } = Typography;

function agoText(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Text type="secondary" style={{ fontSize: 10, letterSpacing: "0.05em", display: "block" }}>
        {label}
      </Text>
      <Text strong className="tabular" style={{ fontSize: 14 }}>
        {value}
      </Text>
    </div>
  );
}

export default function WeatherPanel({ county }: { county: string }) {
  const [weather, setWeather] = useState<LiveWeather | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nowTick, setNowTick] = useState(Date.now());

  useEffect(() => {
    let cancelled = false;
    setWeather(null);
    setError(null);
    api
      .weather(county)
      .then((w) => {
        if (!cancelled) setWeather(w);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || "unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [county]);

  useEffect(() => {
    const t = setInterval(() => setNowTick(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  if (error) {
    return (
      <Card size="small">
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={`Live weather unavailable (${error}).`} />
      </Card>
    );
  }
  if (!weather) {
    return (
      <Card size="small" title={`Live Weather — ${county}`}>
        <Skeleton active paragraph={{ rows: 2 }} title={false} />
      </Card>
    );
  }

  const secondsAgo = (nowTick - weather.observed_at * 1000) / 1000;
  const nextHours = weather.forecast.slice(0, 8); // next ~24h at 3h resolution
  const maxRain = Math.max(0.01, ...nextHours.map((f) => f.rain_probability));

  return (
    <Card
      title={
        <Text
          type="secondary"
          style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
        >
          Live Weather Now — {county}
        </Text>
      }
      extra={
        <Space size={6}>
          <Badge status="success" />
          <Text type="secondary" style={{ fontSize: 11 }}>
            {agoText(secondsAgo)} · {weather.source}
          </Text>
        </Space>
      }
    >
      <Row gutter={[16, 16]} align="middle">
        <Col xs={24} md={10}>
          <Space size={14} align="center">
            <WeatherIcon code={weather.icon} size={52} />
            <div>
              <div className="tabular" style={{ fontSize: 30, fontWeight: 700, lineHeight: 1 }}>
                {weather.temp_c.toFixed(1)}°C
              </div>
              <Text type="secondary" style={{ fontSize: 13, textTransform: "capitalize" }}>
                {weather.description}
              </Text>
            </div>
          </Space>
        </Col>
        <Col xs={24} md={14}>
          <Row gutter={[16, 12]}>
            <Col xs={12} sm={8}>
              <Metric label="FEELS LIKE" value={`${weather.feels_like_c.toFixed(1)}°C`} />
            </Col>
            <Col xs={12} sm={8}>
              <Metric label="HUMIDITY" value={`${weather.humidity_pct}%`} />
            </Col>
            <Col xs={12} sm={8}>
              <Metric label="WIND" value={`${weather.wind_speed_ms.toFixed(1)} m/s`} />
            </Col>
          </Row>
        </Col>
      </Row>

      {/* Rain-chance strip for the next 24h. Bars are scaled against the
          window's own maximum rather than a flat 0-100%, so a day that peaks
          at 20% still shows readable variation instead of eight near-empty
          slivers. The printed percentage stays absolute. */}
      <div style={{ display: "flex", gap: 4, marginTop: 18 }}>
        {nextHours.map((f) => (
          <Tooltip
            key={f.time}
            title={`${f.time.slice(11, 16)} — ${(f.rain_probability * 100).toFixed(0)}% rain chance, ${f.description}`}
          >
            <div style={{ flex: 1, textAlign: "center", minWidth: 0 }}>
              <Text type="secondary" style={{ fontSize: 10 }}>
                {f.time.slice(11, 16)}
              </Text>
              <div
                style={{
                  height: 34,
                  borderRadius: 4,
                  marginTop: 3,
                  display: "flex",
                  alignItems: "flex-end",
                  background: "var(--color-border)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: "100%",
                    height: `${Math.max(6, (f.rain_probability / maxRain) * 100)}%`,
                    background: f.rain_probability > 0.5 ? "var(--color-primary)" : "#93c5fd",
                  }}
                />
              </div>
              <Text type="secondary" className="tabular" style={{ fontSize: 10 }}>
                {(f.rain_probability * 100).toFixed(0)}%
              </Text>
            </div>
          </Tooltip>
        ))}
      </div>

      <Caption>
        Rain chance, next 24h (3h steps) — for situational awareness only. The nowcast model uses aggregated climate
        data, not this forecast directly.
      </Caption>
    </Card>
  );
}
