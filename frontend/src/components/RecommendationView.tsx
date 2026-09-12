import { Typography, Space, Alert, Divider, List } from "antd";
import type { RiskTier } from "../types";
import { buildRecommendation } from "../riskAdvice";
import { RiskTag, useRiskColors } from "../ui";

const { Text, Paragraph } = Typography;

/** Full recommendation content, shown inside an AntD Modal — a dedicated,
 * spacious view rather than crammed under a gauge. */
export default function RecommendationView({
  county,
  tier,
  probability,
  historicalTier,
  historicalRank,
  nCounties,
}: {
  county: string;
  tier: RiskTier;
  probability: number;
  historicalTier?: RiskTier;
  historicalRank?: number;
  nCounties?: number;
}) {
  const colors = useRiskColors();
  const rec = buildRecommendation({ county, tier, probability, historicalTier, historicalRank, nCounties });

  return (
    <div>
      <Space align="baseline" size={12} wrap>
        <span className="tabular" style={{ fontSize: 34, fontWeight: 800, color: colors[tier], lineHeight: 1 }}>
          {(probability * 100).toFixed(0)}%
        </span>
        <RiskTag tier={tier} />
        <Text type="secondary" style={{ fontSize: 12 }}>
          live flood probability
        </Text>
      </Space>

      <Paragraph style={{ fontSize: 14, lineHeight: 1.7, marginTop: 16 }}>{rec.summary}</Paragraph>

      <Divider style={{ margin: "16px 0 12px" }} titlePlacement="start">
        <Text
          type="secondary"
          style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}
        >
          Recommended Actions
        </Text>
      </Divider>

      <List
        size="small"
        dataSource={rec.actions}
        split={false}
        renderItem={(a, i) => (
          <List.Item style={{ paddingInline: 0, alignItems: "flex-start" }}>
            <Space align="start" size={10}>
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 20,
                  height: 20,
                  borderRadius: "50%",
                  background: colors[tier],
                  color: "#fff",
                  fontSize: 11,
                  fontWeight: 700,
                  flexShrink: 0,
                }}
              >
                {i + 1}
              </span>
              <Text style={{ fontSize: 13, lineHeight: 1.7 }}>{a}</Text>
            </Space>
          </List.Item>
        )}
      />

      {rec.infrastructure && (
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 16 }}
          title="Local infrastructure context"
          description={<Text style={{ fontSize: 13, lineHeight: 1.6 }}>{rec.infrastructure}</Text>}
        />
      )}

      <Divider style={{ margin: "16px 0 10px" }} />
      <Text type="secondary" style={{ fontSize: 11, lineHeight: 1.7 }}>
        Generated live from the deployed nowcast model and this county's 2011–2025 historical record — not a static,
        pre-written notice. Re-opening this after the next refresh reflects the latest reading.
      </Text>
    </div>
  );
}
