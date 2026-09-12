import { useId } from "react";

/** Smooth line through a set of points using quadratic Beziers anchored at
 * each segment's midpoint — a cheap, stable curve (no oscillation risk like
 * Catmull-Rom) that reads as a proper trading-terminal mini chart rather than
 * a jagged polyline. */
function smoothPath(pts: [number, number][]): string {
  let d = `M ${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[i + 1];
    d += ` Q ${x0.toFixed(1)},${y0.toFixed(1)} ${((x0 + x1) / 2).toFixed(1)},${((y0 + y1) / 2).toFixed(1)}`;
  }
  const [lx, ly] = pts[pts.length - 1];
  d += ` L ${lx.toFixed(1)},${ly.toFixed(1)}`;
  return d;
}

/** A real reading history plotted as a filled, gradient area chart with a
 * glowing live endpoint — the way a trading terminal shows a mini price
 * chart next to each ticker row, instead of a bare polyline. */
export default function Sparkline({ values, width = 72, height = 26 }: { values: number[]; width?: number; height?: number }) {
  const gradId = useId();
  const pad = 3;
  const h = height - pad * 2;

  if (values.length === 0) return <svg width={width} height={height} />;

  if (values.length === 1) {
    const y = height / 2;
    return (
      <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
        <line x1={0} y1={y} x2={width - 6} y2={y} stroke="currentColor" strokeWidth={1.3} strokeDasharray="1.5 3" opacity={0.4} />
        <circle cx={width - 6} cy={y} r={2.4} fill="currentColor" opacity={0.65} className="spark-pulse" />
      </svg>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);
  const pts: [number, number][] = values.map((v, i) => [i * step, pad + h - ((v - min) / range) * h]);
  const [lastX, lastY] = pts[pts.length - 1];

  const up = values[values.length - 1] >= values[0];
  const color = up ? "#22C55E" : "#EF4444";
  const linePath = smoothPath(pts);
  const areaPath = `${linePath} L ${width},${height} L 0,${height} Z`;

  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.42} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#${gradId})`} stroke="none" />
      <path d={linePath} fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={lastX} cy={lastY} r={2.8} fill={color} />
      <circle cx={lastX} cy={lastY} r={2.8} fill={color} className="spark-pulse" />
    </svg>
  );
}
