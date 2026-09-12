import { Slider as AntSlider, Typography, Space, Tag } from "antd";

const { Text } = Typography;

/** Labelled slider row used by the prediction inputs panel. Wraps AntD's
 * Slider so the label, live-lock badge and current value read the same way
 * across both the climate and terrain input columns. */
export default function Slider({
  label,
  unit,
  value,
  min,
  max,
  step,
  onChange,
  lockBadge,
  disabled,
}: {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  lockBadge?: "live" | "fallback";
  disabled?: boolean;
}) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <Space size={6}>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {label}
          </Text>
          {lockBadge === "live" && (
            <Tag color="blue" style={{ fontSize: 10, lineHeight: "16px", marginInlineEnd: 0 }}>
              LIVE · LOCKED
            </Tag>
          )}
          {lockBadge === "fallback" && (
            <Tag style={{ fontSize: 10, lineHeight: "16px", marginInlineEnd: 0 }}>LOCKED · FALLBACK</Tag>
          )}
        </Space>
        <Text strong className="tabular" style={{ fontSize: 13 }}>
          {value.toFixed(step < 1 ? 2 : 0)} {unit}
        </Text>
      </div>
      <AntSlider
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={onChange}
        tooltip={{ formatter: (v) => `${v} ${unit}`.trim() }}
        style={{ marginBlock: 2 }}
      />
    </div>
  );
}
