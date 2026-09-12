import { useState } from "react";
import { Button, Popover, Select, Typography, Space, Row, Col } from "antd";
import { FilePdfOutlined, DownOutlined } from "@ant-design/icons";
import { api } from "../api";

const { Text } = Typography;

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const CURRENT_YEAR = new Date().getFullYear();
const YEAR_OPTIONS = [CURRENT_YEAR, CURRENT_YEAR + 1, CURRENT_YEAR + 2, CURRENT_YEAR + 3];

/** Explicit "pick a window, then generate" flow — never a one-click download
 * of whatever some other chart control happens to be set to. The month range
 * is always within a single selected year, so a report can never span more
 * than 12 months. */
export default function ReportGenerator({ county }: { county: string }) {
  const [open, setOpen] = useState(false);
  const [year, setYear] = useState(CURRENT_YEAR);
  const [monthFrom, setMonthFrom] = useState(1);
  const [monthTo, setMonthTo] = useState(12);

  const url = api.reportUrl(county, { year, rangeFrom: monthFrom, rangeTo: monthTo });
  const monthOptions = MONTH_NAMES.map((m, i) => ({ value: i + 1, label: m }));

  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      trigger="click"
      placement="bottomRight"
      title={
        <Text
          type="secondary"
          style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}
        >
          Report Period
        </Text>
      }
      content={
        <div style={{ width: 280 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Year
          </Text>
          <Select
            value={year}
            onChange={setYear}
            style={{ width: "100%", marginBottom: 12, marginTop: 4 }}
            options={YEAR_OPTIONS.map((y) => ({
              value: y,
              label: y === CURRENT_YEAR ? `${y} (partial — year in progress)` : String(y),
            }))}
          />

          <Row gutter={12}>
            <Col span={12}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                From month
              </Text>
              <Select
                value={monthFrom}
                onChange={setMonthFrom}
                style={{ width: "100%", marginTop: 4 }}
                options={monthOptions}
              />
            </Col>
            <Col span={12}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                To month
              </Text>
              <Select
                value={monthTo}
                onChange={setMonthTo}
                style={{ width: "100%", marginTop: 4 }}
                options={monthOptions}
              />
            </Col>
          </Row>

          <Text type="secondary" style={{ fontSize: 11, display: "block", margin: "10px 0 12px" }}>
            Covers {MONTH_NAMES[monthFrom - 1]}–{MONTH_NAMES[monthTo - 1]} {year} (max 12 months, within one year).
          </Text>

          <Button
            type="primary"
            block
            icon={<FilePdfOutlined />}
            href={url}
            target="_blank"
            rel="noreferrer"
            onClick={() => setOpen(false)}
          >
            Generate &amp; Download PDF
          </Button>
        </div>
      }
    >
      <Button type="primary">
        <Space size={6}>
          <FilePdfOutlined />
          Generate Report
          <DownOutlined style={{ fontSize: 10 }} />
        </Space>
      </Button>
    </Popover>
  );
}
