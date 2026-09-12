import type { ReactNode } from "react";
import { Collapse, Typography } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

const { Text } = Typography;

/** A collapsed-by-default "what does this mean?" toggle for readers who
 * don't work with charts or statistics day to day. Deliberately opt-in — it
 * adds nothing to the default look of a page until someone asks for it. */
export default function Explainer({ children }: { children: ReactNode }) {
  return (
    <Collapse
      ghost
      size="small"
      style={{ marginTop: 8 }}
      items={[
        {
          key: "explain",
          label: (
            <Text style={{ fontSize: 12, fontWeight: 600, color: "var(--color-primary)" }}>
              <InfoCircleOutlined style={{ marginInlineEnd: 6 }} />
              What does this mean?
            </Text>
          ),
          children: (
            <Text type="secondary" style={{ fontSize: 13, lineHeight: 1.7 }}>
              {children}
            </Text>
          ),
        },
      ]}
    />
  );
}
