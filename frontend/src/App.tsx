import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ConfigProvider, theme as antTheme, Spin, App as AntApp } from "antd";
import { ThemeModeProvider, useThemeMode } from "./context/ThemeContext";
import Layout from "./components/Layout";

// Only the landing page loads eagerly. The rest — especially the
// recharts/maplibre-heavy Live Center, Map and Model pages — are split out so
// a responder on a slow connection isn't downloading the whole app before the
// national overview can paint.
const Overview = lazy(() => import("./pages/Overview"));
const MapPage = lazy(() => import("./pages/MapPage"));
const Prediction = lazy(() => import("./pages/Prediction"));
const CountyProfile = lazy(() => import("./pages/CountyProfile"));
const Historical = lazy(() => import("./pages/Historical"));
const ModelScience = lazy(() => import("./pages/ModelScience"));
const Alerts = lazy(() => import("./pages/Alerts"));
const LiveCenter = lazy(() => import("./pages/LiveCenter"));
const Reports = lazy(() => import("./pages/Reports"));
const About = lazy(() => import("./pages/About"));
const HowToUse = lazy(() => import("./pages/HowToUse"));
const Glossary = lazy(() => import("./pages/Glossary"));
const Contact = lazy(() => import("./pages/Contact"));

function RouteFallback() {
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh" }}>
      <Spin size="large" />
    </div>
  );
}

function Themed({ children }: { children: React.ReactNode }) {
  const { mode } = useThemeMode();
  return (
    <ConfigProvider
      theme={{
        algorithm: mode === "dark" ? antTheme.darkAlgorithm : antTheme.defaultAlgorithm,
        token: {
          colorPrimary: "#1565c0",
          colorLink: "#1565c0",
          borderRadius: 6,
          fontFamily: "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
          colorBgLayout: mode === "dark" ? "#141414" : "#f6f8fb",
        },
        components: {
          Menu: {
            itemSelectedBg: mode === "dark" ? "#112a45" : "#e3edfb",
            itemSelectedColor: "#1565c0",
          },
        },
      }}
    >
      <AntApp>{children}</AntApp>
    </ConfigProvider>
  );
}

export default function App() {
  return (
    <ThemeModeProvider>
      <Themed>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route
                index
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <Overview />
                  </Suspense>
                }
              />
              {(
                [
                  ["/alerts", Alerts],
                  ["/live", LiveCenter],
                  ["/map", MapPage],
                  ["/prediction", Prediction],
                  ["/prediction/:county", Prediction],
                  ["/county/:county", CountyProfile],
                  ["/historical", Historical],
                  ["/model", ModelScience],
                  ["/reports", Reports],
                  ["/about", About],
                  ["/guide", HowToUse],
                  ["/glossary", Glossary],
                  ["/contact", Contact],
                ] as [string, React.ComponentType][]
              ).map(([path, Page]) => (
                <Route
                  key={path}
                  path={path}
                  element={
                    <Suspense fallback={<RouteFallback />}>
                      <Page />
                    </Suspense>
                  }
                />
              ))}
            </Route>
          </Routes>
        </BrowserRouter>
      </Themed>
    </ThemeModeProvider>
  );
}
