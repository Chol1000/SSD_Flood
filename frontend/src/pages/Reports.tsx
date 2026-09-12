import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Card, Table, Input, Button, Typography, Space } from "antd";
import { DownloadOutlined, SearchOutlined } from "@ant-design/icons";
import { api } from "../api";
import type { CountyListItem, RiskTier } from "../types";
import ReportGenerator from "../components/ReportGenerator";
import { PageHeader, RiskTag, Caption, RISK_ORDER } from "../ui";

const { Text } = Typography;

export default function Reports() {
  const [counties, setCounties] = useState<CountyListItem[]>([]);
  const [search, setSearch] = useState("");

  useEffect(() => {
    api.counties().then(setCounties).catch(console.error);
  }, []);

  const visible = useMemo(
    () =>
      [...counties]
        .filter((c) => c.county.toLowerCase().includes(search.toLowerCase()))
        .sort((a, b) => b.flood_rate - a.flood_rate),
    [counties, search]
  );

  return (
    <div style={{ maxWidth: 1100 }}>
      <PageHeader
        eyebrow="South Sudan Flood Early Warning System"
        title="Reports"
        subtitle="Generate a national executive summary, or a detailed per-county report covering historical record, live climate inputs, seasonal risk pattern, compiled outlook and model performance — bounded to any one-year window you choose."
      />

      <Card style={{ marginBottom: 24 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 16,
          }}
        >
          <div style={{ maxWidth: 580 }}>
            <Text strong style={{ fontSize: 15, display: "block" }}>
              National Executive Summary
            </Text>
            <Text type="secondary" style={{ fontSize: 13, lineHeight: 1.6 }}>
              KPIs, risk distribution, flood calendar heatmap, national watchlist and model performance — a
              proposal-ready overview across all 79 counties.
            </Text>
          </div>
          <Button
            type="primary"
            size="large"
            icon={<DownloadOutlined />}
            href={api.nationalReportUrl()}
            target="_blank"
            rel="noreferrer"
          >
            Download National Report
          </Button>
        </div>
      </Card>

      <Card
        title="Per-County Reports"
        extra={
          <Input
            allowClear
            prefix={<SearchOutlined style={{ opacity: 0.45 }} />}
            placeholder="Search county…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: 200 }}
          />
        }
      >
        <Table<CountyListItem>
          dataSource={visible}
          rowKey="county"
          size="middle"
          pagination={{ pageSize: 12, showSizeChanger: false, hideOnSinglePage: true }}
          locale={{ emptyText: `No counties match "${search}".` }}
          columns={[
            {
              title: "County",
              dataIndex: "county",
              sorter: (a, b) => a.county.localeCompare(b.county),
              render: (c: string) => <Text strong>{c}</Text>,
            },
            {
              title: "Historical Rate",
              dataIndex: "flood_rate",
              align: "right",
              defaultSortOrder: "descend",
              sorter: (a, b) => a.flood_rate - b.flood_rate,
              render: (v: number) => <span className="tabular">{(v * 100).toFixed(1)}%</span>,
            },
            {
              title: "Risk",
              dataIndex: "risk_tier",
              align: "right",
              filters: RISK_ORDER.map((t) => ({ text: t, value: t })),
              onFilter: (value, row) => row.risk_tier === value,
              render: (t: RiskTier) => <RiskTag tier={t} />,
            },
            {
              title: "",
              key: "actions",
              align: "right",
              render: (_, c) => (
                <Space>
                  <Link to={`/county/${encodeURIComponent(c.county)}`}>Profile →</Link>
                  <ReportGenerator county={c.county} />
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <Caption>
        Every county report is bounded to a single one-year window (year + from/to month) you choose at generation
        time — see <Link to="/model">Model &amp; Validation</Link> for the methodology behind the figures each report
        contains.
      </Caption>
    </div>
  );
}
