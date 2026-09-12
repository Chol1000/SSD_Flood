/** Self-hosted weather icon set (no external image requests) mapping
 * OpenWeatherMap condition codes to a small inline-SVG language: warm gold
 * for clear-day, indigo/moon for clear-night, grey for cloud cover, blue for
 * rain, a bolt for storms, pale blue for snow, horizontal bands for mist —
 * so the visual reads as "day vs night" and "dry vs wet" at a glance. */

const SUN = "#F5A623";
const MOON = "#CBD5E1";
const NIGHT_SKY = "#3B4863";
const CLOUD_LIGHT = "#CBD3DE";
const CLOUD_DARK = "#8B95A5";
const RAIN_BLUE = "#4A90D9";
const STORM_GREY = "#5B6472";
const BOLT = "#F5C518";
const SNOW = "#DCE8F5";
const MIST = "#AEB8C4";

function Cloud({ fill, cx = 32, cy = 34, scale = 1 }: { fill: string; cx?: number; cy?: number; scale?: number }) {
  return (
    <g transform={`translate(${cx} ${cy}) scale(${scale})`}>
      <ellipse cx="-10" cy="2" rx="11" ry="9" fill={fill} />
      <ellipse cx="6" cy="-4" rx="14" ry="12" fill={fill} />
      <ellipse cx="16" cy="4" rx="10" ry="8" fill={fill} />
      <rect x="-20" y="2" width="46" height="12" rx="6" fill={fill} />
    </g>
  );
}

function Sun({ cx = 26, cy = 26, r = 12 }: { cx?: number; cy?: number; r?: number }) {
  const rays = Array.from({ length: 8 }, (_, i) => (i * Math.PI) / 4);
  return (
    <g>
      {rays.map((a, i) => {
        const x1 = cx + Math.cos(a) * (r + 3);
        const y1 = cy + Math.sin(a) * (r + 3);
        const x2 = cx + Math.cos(a) * (r + 8);
        const y2 = cy + Math.sin(a) * (r + 8);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={SUN} strokeWidth={2.5} strokeLinecap="round" />;
      })}
      <circle cx={cx} cy={cy} r={r} fill={SUN} />
    </g>
  );
}

function Moon({ cx = 26, cy = 26, r = 11 }: { cx?: number; cy?: number; r?: number }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill={MOON} />
      <circle cx={cx + 5} cy={cy - 4} r={r} fill={NIGHT_SKY} />
      <circle cx={cx - 14} cy={cy - 10} r={1.1} fill={MOON} opacity={0.8} />
      <circle cx={cx - 8} cy={cy - 16} r={0.8} fill={MOON} opacity={0.6} />
    </g>
  );
}

function RainDrops({ y = 46, color = RAIN_BLUE, n = 3 }: { y?: number; color?: string; n?: number }) {
  const xs = Array.from({ length: n }, (_, i) => 16 + i * 12);
  return (
    <g>
      {xs.map((x, i) => (
        <path
          key={i} className="wi-raindrop" d={`M${x} ${y} q-3 6 0 10 q3 -2 0 -10`} fill={color} opacity={0.9}
          style={{ animationDelay: `${i * 0.22}s` }}
        />
      ))}
    </g>
  );
}

function Bolt() {
  return <path className="wi-bolt" d="M30 42 L23 54 L29 54 L26 64 L37 49 L30 49 Z" fill={BOLT} stroke="#C99A0A" strokeWidth={0.6} />;
}

function Mist() {
  const ys = [20, 28, 36, 44, 52];
  return (
    <g>
      {ys.map((y, i) => (
        <line key={i} x1={i % 2 === 0 ? 8 : 14} y1={y} x2={i % 2 === 0 ? 56 : 50} y2={y} stroke={MIST} strokeWidth={3.4} strokeLinecap="round" />
      ))}
    </g>
  );
}

function SnowDots() {
  const pts = [[16, 46], [26, 52], [36, 46], [46, 52]];
  return (
    <g>
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={2.1} fill={SNOW} stroke="#9FB6CE" strokeWidth={0.6} />
      ))}
    </g>
  );
}

/** code: OpenWeatherMap icon string, e.g. "01d" / "10n". */
export default function WeatherIcon({ code, size = 40 }: { code: string; size?: number }) {
  const group = code.slice(0, 2);
  const isNight = code.endsWith("n");

  let content: React.ReactNode;
  switch (group) {
    case "01": // clear sky
      content = isNight ? <Moon /> : <Sun />;
      break;
    case "02": // few clouds
      content = (
        <>
          {isNight ? <Moon cx={20} cy={20} r={8} /> : <Sun cx={20} cy={20} r={9} />}
          <Cloud fill={CLOUD_LIGHT} cx={34} cy={38} scale={0.9} />
        </>
      );
      break;
    case "03": // scattered clouds
      content = <Cloud fill={CLOUD_LIGHT} />;
      break;
    case "04": // broken / overcast clouds
      content = (
        <>
          <Cloud fill={CLOUD_LIGHT} cx={26} cy={30} scale={0.85} />
          <Cloud fill={CLOUD_DARK} cx={38} cy={40} scale={0.95} />
        </>
      );
      break;
    case "09": // shower rain
      content = (
        <>
          <Cloud fill={CLOUD_DARK} />
          <RainDrops n={4} />
        </>
      );
      break;
    case "10": // rain
      content = (
        <>
          {isNight ? <Moon cx={18} cy={18} r={7} /> : <Sun cx={18} cy={18} r={8} />}
          <Cloud fill={CLOUD_DARK} cx={34} cy={36} scale={0.95} />
          <RainDrops n={3} />
        </>
      );
      break;
    case "11": // thunderstorm
      content = (
        <>
          <Cloud fill={STORM_GREY} />
          <Bolt />
        </>
      );
      break;
    case "13": // snow
      content = (
        <>
          <Cloud fill={CLOUD_LIGHT} />
          <SnowDots />
        </>
      );
      break;
    case "50": // mist / haze
      content = <Mist />;
      break;
    default:
      content = <Cloud fill={CLOUD_LIGHT} />;
  }

  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      {content}
    </svg>
  );
}
