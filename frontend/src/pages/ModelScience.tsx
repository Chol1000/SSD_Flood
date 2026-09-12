import { useEffect, useState } from "react";
import {
  Card, Row, Col, Table, Typography, Tag, Skeleton, Alert, Descriptions, Statistic,
} from "antd";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ErrorBar, ResponsiveContainer, Cell,
} from "recharts";
import { api } from "../api";
import type { ModelInfo, AblationRow, OnsetRow, SignificanceRow } from "../types";
import Explainer from "../components/Explainer";
import { PageHeader, Caption, useChartTheme, useRiskColors } from "../ui";

const { Text, Paragraph } = Typography;

export default function ModelScience() {
  const [meta, setMeta] = useState<ModelInfo | null>(null);
  const chart = useChartTheme();
  const colors = useRiskColors();

  useEffect(() => {
    api.modelInfo().then(setMeta).catch(console.error);
  }, []);

  if (!meta) {
    return (
      <>
        <Skeleton active paragraph={{ rows: 2 }} style={{ marginBottom: 24 }} />
        <Skeleton active paragraph={{ rows: 8 }} />
      </>
    );
  }

  const modelNames = Object.keys(meta.test_metrics);
  const rankedImportance = Object.entries(meta.feature_importance).sort((a, b) => b[1] - a[1]);
  const importanceData = rankedImportance
    .slice(0, 12)
    .map(([feature, importance]) => ({ feature, importance: importance * 100 }));
  const deployedTest = meta.test_metrics[meta.best_model_name];
  const cvTestData = modelNames.map((name) => ({
    name,
    cv: meta.cv_metrics[name]?.auc_roc_mean,
    cvStd: meta.cv_metrics[name]?.auc_roc_std,
    test: meta.test_metrics[name].auc_roc,
  }));

  const significanceTag = (significant: number) => (
    <Tag color={significant ? "red" : "default"} style={{ marginInlineEnd: 0 }}>
      {significant ? "Yes" : "No"}
    </Tag>
  );

  return (
    <div style={{ maxWidth: 1240 }}>
      <PageHeader eyebrow="Model Science" title="Performance & Validation" subtitle={meta.model_selection_criterion} />

      <Row gutter={[16, 16]} style={{ marginBottom: 20 }} className="compact-stat-row">
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="Decision Threshold"
              value={meta.threshold}
              precision={2}
              styles={{ content: { fontSize: 22, fontWeight: 700 } }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic title="Features Used" value={meta.features.length} styles={{ content: { fontSize: 22, fontWeight: 700 } }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="Excluded (Leakage)"
              value={meta.excluded_features.length}
              styles={{ content: { fontSize: 22, fontWeight: 700 } }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic title="Candidate Models" value={modelNames.length} styles={{ content: { fontSize: 22, fontWeight: 700 } }} />
          </Card>
        </Col>
      </Row>

      {meta.nowcast_vs_outlook && (
        <Alert type="info" showIcon style={{ marginBottom: 20 }} title={meta.nowcast_vs_outlook} />
      )}

      {/* Candidate model scoreboard. */}
      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        {modelNames.map((name) => {
          const m = meta.test_metrics[name];
          const isDeployed = name === meta.best_model_name;
          const isOutlook = name === meta.outlook_model_name;
          return (
            <Col key={name} xs={24} sm={12} lg={8} xl={6}>
              <Card
                size="small"
                style={{ height: "100%", borderColor: isDeployed || isOutlook ? "var(--color-primary)" : undefined }}
              >
                {(isDeployed || isOutlook) && (
                  <Tag color="blue" style={{ fontSize: 10, marginBottom: 6 }}>
                    {isDeployed ? "DEPLOYED · NOWCAST" : "DEPLOYED · OUTLOOK"}
                  </Tag>
                )}
                <Text strong style={{ fontSize: 13, display: "block" }}>
                  {name}
                </Text>
                <div
                  className="tabular"
                  style={{
                    fontSize: 24,
                    fontWeight: 700,
                    marginTop: 4,
                    color: isDeployed ? "var(--color-primary)" : undefined,
                  }}
                >
                  {m.auc_roc.toFixed(4)}
                </div>
                <Text type="secondary" style={{ fontSize: 11, display: "block", marginBottom: 8 }}>
                  AUC-ROC
                </Text>
                {(
                  [
                    ["F1", m.f1],
                    ["Precision", m.precision],
                    ["Recall", m.recall],
                  ] as [string, number][]
                ).map(([label, value]) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0" }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {label}
                    </Text>
                    <Text strong className="tabular" style={{ fontSize: 12 }}>
                      {value.toFixed(4)}
                    </Text>
                  </div>
                ))}
              </Card>
            </Col>
          );
        })}
      </Row>

      <Card title="Feature Importance (Deployed Nowcast Model)" style={{ marginBottom: 20 }}>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={importanceData} layout="vertical" margin={{ left: 40, right: 16 }}>
            <CartesianGrid stroke={chart.grid} horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11, fill: chart.axis }} stroke={chart.grid} unit="%" />
            <YAxis
              type="category"
              dataKey="feature"
              width={170}
              tick={{ fontSize: 11, fill: chart.text }}
              stroke={chart.grid}
            />
            <Tooltip contentStyle={chart.tooltip} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {importanceData.map((_, i) => (
                <Cell key={i} fill={chart.accent} fillOpacity={1 - i * 0.05} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <Explainer>
          Each bar shows how much one input variable moves the model's predictions, on average, across every test
          case. Longer bars matter more. This is <i>not</i> the same as "this variable causes floods" — it just means
          the model leans on it heavily to make its call, which is why lag/rolling rainfall history tends to dominate
          over slow-changing terrain features.
        </Explainer>
      </Card>

      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        <Col xs={24} xl={10}>
          <Card title={`Confusion Matrix — ${meta.best_model_name} (2024–2025 test holdout)`} style={{ height: "100%" }}>
            <ConfusionMatrix tp={deployedTest.tp} fp={deployedTest.fp} fn={deployedTest.fn} tn={deployedTest.tn} />
            <Caption>
              {deployedTest.fn} missed floods (false negatives) vs {deployedTest.fp} false alarms — the
              precision-first selection criterion above explains why the model favours one error type over the other.
            </Caption>
            <Explainer>
              Every prediction the model made on unseen 2024–2025 data falls into exactly one of these four boxes.{" "}
              <b>True Positive</b>: it said "flood" and one happened. <b>True Negative</b>: it said "no flood" and
              none happened — both are correct calls (green). <b>False Negative</b>: it missed a real flood.{" "}
              <b>False Positive</b>: it raised an alarm that didn't happen — both are errors (red). A model tuned to
              avoid false alarms will naturally miss more real events, and vice versa — there's no way to drive both
              to zero at once, only a trade-off to choose.
            </Explainer>
          </Card>
        </Col>

        <Col xs={24} xl={14}>
          <Card title="Cross-Validation vs. Held-Out Test — Generalization Check" style={{ height: "100%" }}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={cvTestData} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={chart.grid} vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: chart.axis }} stroke={chart.grid} />
                <YAxis domain={[0.5, 1]} tick={{ fontSize: 10, fill: chart.axis }} stroke={chart.grid} />
                <Tooltip contentStyle={chart.tooltip} formatter={(v) => Number(v).toFixed(4)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="cv" name="CV AUC (5-fold, ±1 std)" fill={chart.axis} radius={[4, 4, 0, 0]}>
                  <ErrorBar dataKey="cvStd" width={4} strokeWidth={1.4} stroke={chart.text} />
                </Bar>
                <Bar dataKey="test" name="Test AUC (2024–25 holdout)" fill={chart.accent} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <Caption>
              Test AUC tracking close to the cross-validation band means the deployed model isn't overfit to its
              training folds — a real gap here would be a red flag before deployment.
            </Caption>
            <Explainer>
              AUC-ROC scores from 0.5 (no better than a coin flip) to 1.0 (perfect). "CV AUC" is the average score
              across 5 different training/validation splits during development; "Test AUC" is the score on 2024–2025
              data the model never saw at all during training — the real, honest test. If test performance were much
              worse than the CV band, that would mean the model memorized its training data rather than learning
              something that generalizes. Here they track closely, which is the reassuring result.
            </Explainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        <Col xs={24} xl={12}>
          <Card title="Ablation Study — Feature Group Contributions" style={{ height: "100%" }}>
            <Table<AblationRow>
              dataSource={meta.ablation}
              rowKey="Feature Set"
              size="small"
              pagination={false}
              columns={[
                { title: "Feature Set", dataIndex: "Feature Set" },
                {
                  title: "CV AUC",
                  dataIndex: "CV AUC",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(3)}</span>,
                },
                {
                  title: "Test AUC",
                  dataIndex: "Test AUC",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(3)}</span>,
                },
                {
                  title: "Test F1",
                  dataIndex: "Test F1",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(3)}</span>,
                },
              ]}
            />
            <Caption>
              Removing temporal/lag signal drops AUC the most — confirming the model leans on recent flood history,
              not just static geography.
            </Caption>
            <Explainer>
              An ablation study removes one group of input features at a time and re-trains, to see how much each
              group actually contributes. If removing a group barely changes the score, that group wasn't doing much
              work — if it causes a big drop, the model depends on it heavily. This is how we check that a feature
              earns its place rather than being included because it was available.
            </Explainer>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title="Deployed Model vs Persistence Baseline" style={{ height: "100%" }}>
            <Paragraph type="secondary" style={{ fontSize: 13 }}>
              {meta.persistence_baseline.description}
            </Paragraph>
            <Table
              dataSource={[
                {
                  key: "model",
                  name: meta.best_model_name,
                  auc: meta.test_metrics[meta.best_model_name].auc_roc,
                  f1: meta.test_metrics[meta.best_model_name].f1,
                  precision: meta.test_metrics[meta.best_model_name].precision,
                  deployed: true,
                },
                {
                  key: "persistence",
                  name: "Persistence",
                  auc: meta.persistence_baseline.auc_roc,
                  f1: meta.persistence_baseline.f1,
                  precision: meta.persistence_baseline.precision,
                  deployed: false,
                },
              ]}
              size="small"
              pagination={false}
              columns={[
                {
                  title: "",
                  dataIndex: "name",
                  render: (n: string, row) => (row.deployed ? <Text strong>{n}</Text> : <Text type="secondary">{n}</Text>),
                },
                {
                  title: "AUC-ROC",
                  dataIndex: "auc",
                  align: "right",
                  render: (v: number, row) => (
                    <span className="tabular" style={{ color: row.deployed ? "var(--color-primary)" : undefined, fontWeight: row.deployed ? 600 : 400 }}>
                      {v.toFixed(4)}
                    </span>
                  ),
                },
                {
                  title: "F1",
                  dataIndex: "f1",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(4)}</span>,
                },
                {
                  title: "Precision",
                  dataIndex: "precision",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(4)}</span>,
                },
              ]}
            />
            <Caption>{meta.persistence_baseline.note}</Caption>
            <Explainer>
              "Persistence" is the simplest possible forecast — just guess that whatever happened last month will
              happen again this month, no model required. It's the baseline any real model has to beat to prove it's
              adding value rather than restating recent history back at you.
            </Explainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        <Col xs={24} xl={12}>
          <Card title={`Statistical Significance — ${meta.best_model_name} vs Alternatives`} style={{ height: "100%" }}>
            <Caption>{meta.significance_tests.method}</Caption>
            <Table<SignificanceRow>
              dataSource={meta.significance_tests.delong}
              rowKey="comparison"
              size="small"
              pagination={false}
              style={{ marginTop: 10 }}
              columns={[
                { title: "Comparison", dataIndex: "comparison" },
                {
                  title: "ΔAUC",
                  dataIndex: "delta_auc",
                  align: "right",
                  render: (v?: number) => (
                    <span className="tabular">{v != null ? `${v >= 0 ? "+" : ""}${v.toFixed(4)}` : "—"}</span>
                  ),
                },
                {
                  title: "p-value",
                  dataIndex: "p_value",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(4)}</span>,
                },
                {
                  title: "Significant",
                  dataIndex: "significant_05",
                  align: "right",
                  render: significanceTag,
                },
              ]}
            />
            <Explainer>
              Tests whether {meta.best_model_name}'s AUC score is <i>genuinely</i> better than each alternative, or
              whether the difference could just be random luck from which test cases happened to be included. A
              p-value under 0.05 (marked "Yes") means the gap is very unlikely to be chance.
            </Explainer>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title="Onset-Flood Sensitivity" style={{ height: "100%" }}>
            <Paragraph type="secondary" style={{ fontSize: 13 }}>
              Detection of first-month flood onsets (no prior-month signal available) at lower decision thresholds.
            </Paragraph>
            <Table<OnsetRow>
              dataSource={meta.onset_analysis}
              rowKey="threshold"
              size="small"
              pagination={false}
              columns={[
                {
                  title: "Threshold",
                  dataIndex: "threshold",
                  render: (v: number) => <span className="tabular">{v.toFixed(2)}</span>,
                },
                {
                  title: "Onsets detected",
                  key: "onsets",
                  align: "right",
                  render: (_, row) => (
                    <span className="tabular">
                      {row.onset_detected}/{row.onset_total}
                    </span>
                  ),
                },
                {
                  title: "FP",
                  dataIndex: "fp",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v}</span>,
                },
                {
                  title: "Recall",
                  dataIndex: "recall",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(3)}</span>,
                },
              ]}
            />
            <Explainer>
              A flood "onset" is the first month a county floods after a dry spell — the hardest case for this model,
              since the usual strongest signal (did it flood last month?) isn't available yet. This checks how many of
              those first-time events get caught if the alert threshold were lowered, and at what cost in false
              alarms.
            </Explainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        <Col xs={24} xl={14}>
          <Card title="McNemar's Test — Paired Error Disagreement" style={{ height: "100%" }}>
            <Caption>
              b = test cases only {meta.best_model_name} got right · c = cases only the alternative got right. A large,
              significant gap between b and c means the two models fail on genuinely different cases, not just noise.
            </Caption>
            <Table<SignificanceRow>
              dataSource={meta.significance_tests.mcnemar}
              rowKey="comparison"
              size="small"
              pagination={false}
              style={{ marginTop: 10 }}
              columns={[
                { title: "Comparison", dataIndex: "comparison" },
                {
                  title: "b",
                  dataIndex: "b_LR_wins",
                  align: "right",
                  render: (v?: number) => <span className="tabular">{v ?? "—"}</span>,
                },
                {
                  title: "c",
                  dataIndex: "c_other_wins",
                  align: "right",
                  render: (v?: number) => <span className="tabular">{v ?? "—"}</span>,
                },
                {
                  title: "χ²",
                  dataIndex: "chi2_stat",
                  align: "right",
                  render: (v?: number) => <span className="tabular">{v != null ? v.toFixed(3) : "—"}</span>,
                },
                {
                  title: "p-value",
                  dataIndex: "p_value",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v.toFixed(4)}</span>,
                },
                {
                  title: "Significant",
                  dataIndex: "significant_05",
                  align: "right",
                  render: significanceTag,
                },
              ]}
            />
            <Explainer>
              Unlike the AUC comparison above (which looks at overall scores), this looks at individual test cases:
              for each one, did the two models actually agree? b and c count the cases where only one model got it
              right. If those two numbers are close, the models mostly fail on the same cases anyway — a big gap
              (flagged Significant) means one model is genuinely catching things the other misses.
            </Explainer>
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Card title={`Full Ranked Feature Importance — all ${rankedImportance.length}`} style={{ height: "100%" }}>
            <Table
              dataSource={rankedImportance.map(([feature, importance], i) => ({
                key: feature,
                rank: i + 1,
                feature,
                importance,
              }))}
              size="small"
              pagination={false}
              scroll={{ y: 300 }}
              columns={[
                {
                  title: "#",
                  dataIndex: "rank",
                  width: 48,
                  render: (v: number) => (
                    <Text type="secondary" className="tabular">
                      {v}
                    </Text>
                  ),
                },
                { title: "Feature", dataIndex: "feature" },
                {
                  title: "Importance",
                  dataIndex: "importance",
                  align: "right",
                  render: (v: number) => (
                    <Text strong className="tabular">
                      {(v * 100).toFixed(2)}%
                    </Text>
                  ),
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Methodology &amp; Feature Engineering">
        <Row gutter={[32, 20]}>
          <Col xs={24} lg={12}>
            {Object.entries(meta.methodology).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 14 }}>
                <Text
                  style={{
                    fontSize: 11,
                    color: "var(--color-primary)",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    fontWeight: 700,
                    display: "block",
                  }}
                >
                  {k.replace(/_/g, " ")}
                </Text>
                <Text type="secondary" style={{ fontSize: 13, lineHeight: 1.6 }}>
                  {v}
                </Text>
              </div>
            ))}
          </Col>

          <Col xs={24} lg={12}>
            <Text
              style={{
                fontSize: 11,
                color: "var(--color-primary)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                fontWeight: 700,
                display: "block",
                marginBottom: 8,
              }}
            >
              Engineered Features
            </Text>
            <Descriptions column={1} size="small" bordered>
              {Object.entries(meta.feature_engineering).map(([k, v]) => (
                <Descriptions.Item key={k} label={<Text code>{k}</Text>}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {v}
                  </Text>
                </Descriptions.Item>
              ))}
            </Descriptions>

            <Alert
              type="warning"
              style={{ marginTop: 16 }}
              title="Excluded (label leakage)"
              description={
                <Text style={{ fontSize: 13 }}>
                  <Text code>{meta.excluded_features.join(", ")}</Text> — {meta.exclusion_reason}
                </Text>
              }
            />
          </Col>
        </Row>
      </Card>
    </div>
  );

  /** Four-box outcome grid for the deployed model on the test holdout —
   * correct calls green, errors red, so the precision/recall trade-off is
   * visible rather than needing to be read off a table. */
  function ConfusionMatrix({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
    const total = tp + fp + fn + tn;
    const pct = (v: number) => (total ? `${((v / total) * 100).toFixed(1)}%` : "—");

    const Box = ({ label, value, hit }: { label: string; value: number; hit: boolean }) => (
      <div
        style={{
          padding: "14px 10px",
          borderRadius: 8,
          textAlign: "center",
          background: `${hit ? colors.Low : colors.High}14`,
          border: `1px solid ${hit ? colors.Low : colors.High}40`,
        }}
      >
        <div className="tabular" style={{ fontSize: 24, fontWeight: 800, color: hit ? colors.Low : colors.High }}>
          {value}
        </div>
        <Text style={{ fontSize: 11, fontWeight: 600, display: "block" }}>{label}</Text>
        <Text type="secondary" className="tabular" style={{ fontSize: 10 }}>
          {pct(value)}
        </Text>
      </div>
    );

    const axisLabel = (text: string) => (
      <Text
        type="secondary"
        style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.04em", textAlign: "center", display: "block" }}
      >
        {text}
      </Text>
    );

    return (
      <div style={{ marginTop: 8 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 4, marginInlineStart: 78 }}>
          {axisLabel("Predicted: Flood")}
          {axisLabel("Predicted: No Flood")}
        </div>
        {(
          [
            ["Actual: Flood", ["True Positive", tp, true], ["False Negative", fn, false]],
            ["Actual: None", ["False Positive", fp, false], ["True Negative", tn, true]],
          ] as [string, [string, number, boolean], [string, number, boolean]][]
        ).map(([rowLabel, left, right]) => (
          <div key={rowLabel} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <Text
              type="secondary"
              style={{ width: 70, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.04em", flexShrink: 0 }}
            >
              {rowLabel}
            </Text>
            <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <Box label={left[0]} value={left[1]} hit={left[2]} />
              <Box label={right[0]} value={right[1]} hit={right[2]} />
            </div>
          </div>
        ))}
      </div>
    );
  }
}
