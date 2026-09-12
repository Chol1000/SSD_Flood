import type { ReactNode } from "react";
import { Card, Col, Row, Statistic, Tag, Typography, Space, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useThemeMode } from "./context/ThemeContext";
import type { RiskTier } from "./types";

const { Title, Text } = Typography;

export const RISK_ORDER: RiskTier[] = ["Critical", "High", "Moderate", "Low"];

/* Two palettes rather than one: the light-mode risk colors (deep green
   through deep red) are chosen for contrast against white, and go muddy and
   low-contrast on AntD's #141414 dark surface. The dark set is the same four
   hues lifted in lightness so they stay legible there. Charts need literal
   hex (recharts can't resolve CSS vars), so this is the single source both
   the DOM and the charts read from. */
const RISK_HEX_LIGHT: Record<RiskTier, string> = {
  Low: "#15803d",
  Moderate: "#b7791e",
  High: "#c2410c",
  Critical: "#dc2626",
};
const RISK_HEX_DARK: Record<RiskTier, string> = {
  Low: "#4ade80",
  Moderate: "#fbbf24",
  High: "#fb923c",
  Critical: "#f87171",
};

/** AntD Tag preset per tier, for the places a preset reads better than a
 * literal hex (it carries AntD's own matching border + background tints). */
const RISK_TAG_COLOR: Record<RiskTier, string> = {
  Low: "green",
  Moderate: "gold",
  High: "orange",
  Critical: "red",
};

export function useRiskColors(): Record<RiskTier, string> {
  const { mode } = useThemeMode();
  return mode === "dark" ? RISK_HEX_DARK : RISK_HEX_LIGHT;
}

/** Axis/grid/tooltip colors for recharts, which renders to SVG attributes and
 * so can't inherit any of AntD's theming. Keeps every chart in the app
 * switching with the theme instead of keeping light-mode gridlines on a dark
 * card. */
export function useChartTheme() {
  const { mode } = useThemeMode();
  const dark = mode === "dark";
  return {
    dark,
    axis: dark ? "#8c8c8c" : "#8792a2",
    grid: dark ? "#303030" : "#e8ecf2",
    accent: dark ? "#4a90e8" : "#1565c0",
    surface: dark ? "#1f1f1f" : "#ffffff",
    text: dark ? "#e6e6e6" : "#1c2733",
    tooltip: {
      fontSize: 12,
      borderRadius: 6,
      background: dark ? "#1f1f1f" : "#ffffff",
      border: `1px solid ${dark ? "#303030" : "#dde3ec"}`,
      color: dark ? "#e6e6e6" : "#1c2733",
    } as const,
  };
}

/** Risk indicator. `dot` is the dense-table variant — a colored dot plus
 * muted colored text, which stays readable when the same tier repeats across
 * 79 rows where a filled tag on every row reads as noise. */
export function RiskTag({ tier, dot = false }: { tier: RiskTier; dot?: boolean }) {
  const colors = useRiskColors();
  if (dot) {
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: colors[tier], flexShrink: 0 }} />
        <span style={{ color: colors[tier], fontWeight: 600, fontSize: 13 }}>{tier}</span>
      </span>
    );
  }
  return (
    <Tag color={RISK_TAG_COLOR[tier]} style={{ marginInlineEnd: 0, fontWeight: 600 }}>
      {tier}
    </Tag>
  );
}

/** Page masthead: eyebrow + title + one line of context, with an optional
 * action slot on the right. Every page opens with this so they read as one
 * product rather than fourteen separately-styled screens. */
export function PageHeader({
  eyebrow,
  title,
  subtitle,
  extra,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: ReactNode;
  extra?: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        flexWrap: "wrap",
        gap: 12,
        marginBottom: 20,
      }}
    >
      <div style={{ minWidth: 0 }}>
        {eyebrow && (
          <Text
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--color-primary)",
            }}
          >
            {eyebrow}
          </Text>
        )}
        <Title level={3} style={{ margin: eyebrow ? "6px 0 0" : 0 }}>
          {title}
        </Title>
        {subtitle && (
          <Text type="secondary" style={{ fontSize: 13, display: "block", marginTop: 6 }}>
            {subtitle}
          </Text>
        )}
      </div>
      {/* The action slot must be allowed to shrink and wrap: on a phone a
          fixed-width group here (a county Select plus two buttons) is wider
          than the viewport, and `flex-shrink: 0` would push the whole page
          sideways rather than letting the group wrap onto its own line. */}
      {extra && <div style={{ minWidth: 0, maxWidth: "100%" }}>{extra}</div>}
    </div>
  );
}

/** A single KPI tile. Wraps AntD's Statistic so every stat in the app gets the
 * same card shell, icon treatment, and optional "what is this" tooltip. */
export function StatCard({
  label,
  value,
  suffix,
  precision,
  icon,
  color,
  hint,
}: {
  label: string;
  value: number | string;
  suffix?: string;
  precision?: number;
  icon?: ReactNode;
  color?: string;
  hint?: string;
}) {
  return (
    <Card size="small" styles={{ body: { padding: "16px 18px" } }}>
      <Statistic
        title={
          <Space size={4}>
            {icon}
            <span style={{ fontSize: 12 }}>{label}</span>
            {hint && (
              <Tooltip title={hint}>
                <InfoCircleOutlined style={{ fontSize: 11, opacity: 0.5 }} />
              </Tooltip>
            )}
          </Space>
        }
        value={value}
        suffix={suffix}
        precision={precision}
        styles={{ content: { fontSize: 26, fontWeight: 700, color } }}
      />
    </Card>
  );
}

/** Row of StatCards that stacks 4 → 2 → 1 across breakpoints. */
export function StatRow({ children }: { children: ReactNode[] }) {
  return (
    <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
      {children.map((child, i) => (
        <Col key={i} xs={24} sm={12} xl={6}>
          {child}
        </Col>
      ))}
    </Row>
  );
}

/** recharts types a Tooltip `formatter`'s first argument as the full
 * ValueType union (number | string | array) even on charts that only ever
 * carry numbers, so a plain `(v: number) => …` no longer type-checks in
 * recharts 3.x. Taking `unknown` and coercing once here keeps every call site
 * a one-liner instead of repeating the widening cast on every chart. */
export function tooltipValue(
  format: (n: number) => string,
  label?: string
): (v: unknown, name?: unknown) => [string, string] {
  return (v, name) => [format(Number(v)), label ?? String(name ?? "")];
}

/** Small muted caption under a chart or table — the "how to read this /
 * what's the caveat" line. */
export function Caption({ children }: { children: ReactNode }) {
  return (
    <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 10, lineHeight: 1.6 }}>
      {children}
    </Text>
  );
}
