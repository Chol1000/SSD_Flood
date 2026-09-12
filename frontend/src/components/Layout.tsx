import { useEffect, useMemo, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  Layout as AntLayout,
  Menu,
  Select,
  Button,
  Typography,
  Space,
  Drawer,
  Grid,
  Badge,
  Tooltip,
} from "antd";
import {
  DashboardOutlined,
  AlertOutlined,
  ThunderboltOutlined,
  GlobalOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  FileTextOutlined,
  InfoCircleOutlined,
  ReadOutlined,
  PhoneOutlined,
  QuestionCircleOutlined,
  SunOutlined,
  MoonOutlined,
  MenuOutlined,
  SearchOutlined,
  EnvironmentOutlined,
} from "@ant-design/icons";
import { useThemeMode } from "../context/ThemeContext";
import { api } from "../api";
import type { CountyListItem } from "../types";

const { Sider, Header, Content } = AntLayout;
const { Text } = Typography;

const NAV_ITEMS = [
  {
    type: "group" as const,
    label: "Monitoring",
    children: [
      { key: "/", icon: <DashboardOutlined />, label: "National Overview" },
      { key: "/alerts", icon: <AlertOutlined />, label: "Alerts" },
      { key: "/live", icon: <ThunderboltOutlined />, label: "Live Updates" },
      { key: "/map", icon: <GlobalOutlined />, label: "Risk Map" },
    ],
  },
  {
    type: "group" as const,
    label: "Analysis",
    children: [
      { key: "/prediction", icon: <ExperimentOutlined />, label: "Prediction" },
      { key: "/historical", icon: <LineChartOutlined />, label: "Historical" },
      { key: "/model", icon: <ExperimentOutlined />, label: "Model & Validation" },
      { key: "/reports", icon: <FileTextOutlined />, label: "Reports" },
    ],
  },
  {
    type: "group" as const,
    label: "Reference",
    children: [
      { key: "/guide", icon: <QuestionCircleOutlined />, label: "How to Use This" },
      { key: "/glossary", icon: <ReadOutlined />, label: "Glossary" },
      { key: "/about", icon: <InfoCircleOutlined />, label: "About" },
      { key: "/contact", icon: <PhoneOutlined />, label: "Contact & Emergency" },
    ],
  },
];

/** Live heartbeat against the API, shown as a colored dot in the header so a
 * responder can tell at a glance whether what they're looking at is current
 * or a stale render of a backend that went away. */
function SystemStatus({ compact }: { compact: boolean }) {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      api
        .health()
        .then(() => !cancelled && setOnline(true))
        .catch(() => !cancelled && setOnline(false));
    };
    check();
    const t = setInterval(check, 20000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const status = online === null ? "default" : online ? "success" : "error";
  const label = online === null ? "Connecting…" : online ? "System Live" : "Reconnecting…";

  if (compact) {
    return (
      <Tooltip title={label}>
        <Badge status={status} />
      </Tooltip>
    );
  }
  return (
    <Space size={6}>
      <Badge status={status} />
      <Text type="secondary" style={{ fontSize: 13 }}>
        {label}
      </Text>
    </Space>
  );
}

function LiveClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <Text type="secondary" style={{ fontSize: 13, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>
      {now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })} UTC
      {now.getTimezoneOffset() === 0 ? "" : ""}
    </Text>
  );
}

function SidebarNav({
  activeKey,
  onNavigate,
}: {
  activeKey: string;
  onNavigate?: () => void;
}) {
  const navigate = useNavigate();
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "16px 20px",
          borderBottom: "1px solid var(--color-border)",
          flexShrink: 0,
        }}
      >
        <div style={{ minWidth: 0, lineHeight: 1.25 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: "var(--color-primary)" }}>Flood EWS</div>
          <Text type="secondary" style={{ fontSize: 11 }}>
            South Sudan
          </Text>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
        <Menu
          mode="inline"
          selectedKeys={[activeKey]}
          items={NAV_ITEMS}
          onClick={({ key }) => {
            navigate(key);
            onNavigate?.();
          }}
          style={{ borderInlineEnd: 0 }}
        />
      </div>

      <div
        style={{
          flexShrink: 0,
          padding: "12px 20px 16px",
          borderTop: "1px solid var(--color-border)",
          lineHeight: 1.6,
        }}
      >
        <Text type="secondary" style={{ fontSize: 11 }}>
          79 counties · nowcast &amp; month-ahead outlook. Live climate via Open-Meteo with an automatic NASA POWER
          fallback.
        </Text>
      </div>
    </div>
  );
}

export default function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { mode, toggle } = useThemeMode();
  const screens = Grid.useBreakpoint();
  const isMobile = !screens.lg;
  const [menuOpen, setMenuOpen] = useState(false);
  const [counties, setCounties] = useState<CountyListItem[]>([]);

  useEffect(() => {
    api.counties().then(setCounties).catch(() => {});
  }, []);

  // `/county/Bor` and `/prediction/Bor` should both light up their parent nav
  // entry rather than leaving the sidebar with nothing selected.
  const activeKey = useMemo(() => {
    const p = location.pathname;
    if (p === "/") return "/";
    if (p.startsWith("/county/")) return "/";
    if (p.startsWith("/prediction")) return "/prediction";
    return "/" + p.split("/")[1];
  }, [location.pathname]);

  const countyOptions = useMemo(
    () => counties.map((c) => ({ value: c.county, label: c.county })),
    [counties]
  );

  return (
    <AntLayout style={{ minHeight: "100vh" }}>
      {!isMobile && (
        <Sider
          width={248}
          theme="light"
          style={{
            borderInlineEnd: "1px solid var(--color-border)",
            position: "sticky",
            top: 0,
            height: "100vh",
            background: "var(--color-surface)",
          }}
        >
          <SidebarNav activeKey={activeKey} />
        </Sider>
      )}

      {isMobile && (
        <Drawer
          placement="left"
          size={248}
          open={menuOpen}
          onClose={() => setMenuOpen(false)}
          closable={false}
          styles={{ body: { padding: 0, height: "100%" } }}
        >
          <SidebarNav activeKey={activeKey} onNavigate={() => setMenuOpen(false)} />
        </Drawer>
      )}

      <AntLayout>
        <Header
          style={{
            background: "var(--color-surface)",
            borderBottom: "1px solid var(--color-border)",
            display: "flex",
            alignItems: "center",
            gap: isMobile ? 8 : 16,
            padding: isMobile ? "0 12px" : "0 28px",
            height: 60,
            lineHeight: "normal",
            position: "sticky",
            top: 0,
            zIndex: 10,
          }}
        >
          {isMobile && (
            <Button shape="circle" icon={<MenuOutlined />} onClick={() => setMenuOpen(true)} style={{ flexShrink: 0 }} />
          )}

          <Select
            showSearch
            allowClear
            value={null}
            placeholder={isMobile ? "Find a county…" : "Jump to a county…"}
            options={countyOptions}
            onChange={(county) => county && navigate(`/county/${encodeURIComponent(county)}`)}
            suffixIcon={<SearchOutlined />}
            variant="filled"
            style={{ flex: isMobile ? 1 : "0 1 300px", minWidth: 0 }}
            notFoundContent={counties.length ? "No match" : "Loading counties…"}
          />

          <Space size={isMobile ? 8 : 16} style={{ marginInlineStart: "auto", flexShrink: 0 }}>
            {!isMobile && <LiveClock />}
            <SystemStatus compact={isMobile} />
            {!isMobile && (
              <Button icon={<EnvironmentOutlined />} onClick={() => navigate("/map")}>
                Risk Map
              </Button>
            )}
            <Tooltip title={mode === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
              <Button shape="circle" icon={mode === "dark" ? <SunOutlined /> : <MoonOutlined />} onClick={toggle} />
            </Tooltip>
          </Space>
        </Header>

        <Content
          style={{
            padding: isMobile ? "16px 12px 40px" : "28px 32px 56px",
            maxWidth: 1600,
            margin: "0 auto",
            width: "100%",
          }}
        >
          <Outlet />
        </Content>
      </AntLayout>
    </AntLayout>
  );
}
