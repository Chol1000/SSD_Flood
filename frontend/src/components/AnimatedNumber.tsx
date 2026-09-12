import { useEffect, useRef, useState } from "react";

/** Crossfades + flashes green/red briefly when the displayed value changes —
 * the "ticking price" feel from trading terminals, applied to a number. */
export default function AnimatedNumber({
  value, decimals = 1, suffix = "", style,
}: {
  value: number; decimals?: number; suffix?: string; style?: React.CSSProperties;
}) {
  const [flash, setFlash] = useState<"up" | "down" | null>(null);
  const prev = useRef(value);

  useEffect(() => {
    if (prev.current !== value) {
      setFlash(value > prev.current ? "up" : "down");
      prev.current = value;
      const t = setTimeout(() => setFlash(null), 700);
      return () => clearTimeout(t);
    }
  }, [value]);

  return (
    <span
      className="tabular"
      style={{
        transition: "color 400ms ease",
        color: flash === "up" ? "var(--risk-critical)" : flash === "down" ? "var(--color-primary)" : undefined,
        ...style,
      }}
    >
      {value.toFixed(decimals)}{suffix}
    </span>
  );
}
