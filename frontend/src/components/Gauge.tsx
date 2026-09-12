import type { RiskTier } from "../types";
import { useRiskColors } from "../ui";

/** Half-circle probability dial. Kept as hand-drawn SVG rather than AntD's
 * Progress `type="dashboard"` because the arc needs to read as a gauge with
 * its own scale, and the tier label sits inside the arc. */
export default function Gauge({
  probability,
  tier,
  size = 160,
}: {
  probability: number;
  tier: RiskTier;
  size?: number;
}) {
  const colors = useRiskColors();
  const pct = Math.max(0, Math.min(1, probability));
  const r = size / 2 - 10;
  const cx = size / 2;
  const cy = size / 2;
  const startAngle = Math.PI; // 180deg
  const endAngle = Math.PI + Math.PI * pct;
  const arcPoint = (angle: number) => [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  const [x0, y0] = arcPoint(startAngle);
  const [x1, y1] = arcPoint(endAngle);
  const largeArc = pct > 0.5 ? 1 : 0;
  const color = colors[tier];

  return (
    <svg width={size} height={size / 2 + 22} viewBox={`0 0 ${size} ${size / 2 + 22}`} style={{ flexShrink: 0 }}>
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
        fill="none"
        stroke="var(--color-border)"
        strokeWidth={10}
        strokeLinecap="round"
      />
      <path
        d={`M ${x0} ${y0} A ${r} ${r} 0 ${largeArc} 1 ${x1} ${y1}`}
        fill="none"
        stroke={color}
        strokeWidth={10}
        strokeLinecap="round"
      />
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize={size * 0.19} fontWeight={700} fill={color}>
        {(pct * 100).toFixed(1)}%
      </text>
      <text
        x={cx}
        y={cy + 16}
        textAnchor="middle"
        fontSize={11}
        fill="var(--color-text-muted)"
        letterSpacing="0.05em"
      >
        {tier.toUpperCase()} RISK
      </text>
    </svg>
  );
}
