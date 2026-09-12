import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { RiskTier } from "../types";

const RISK_HEX: Record<RiskTier, string> = {
  Low: "#2BAE66",
  Moderate: "#E0A62E",
  High: "#E0722E",
  Critical: "#E14B4B",
};

export interface MapPoint {
  county: string;
  lat: number;
  lon: number;
  value: number; // probability or historical rate, 0-1
  tier: RiskTier;
}

export default function MapView({
  points, selected, onSelect, focusZoom,
}: {
  points: MapPoint[];
  selected: string | null;
  onSelect: (county: string) => void;
  /** If set, the map flies to and holds this zoom whenever there's a single point (locator use case). */
  focusZoom?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<Record<string, maplibregl.Marker>>({});
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const flownToRef = useRef<string | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          satellite: {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            attribution: "Esri World Imagery",
          },
          // County boundaries for 78 of 79 counties (Abyei's disputed status
          // excludes it from South Sudan's own admin dataset) — sourced from
          // geoBoundaries.org, itself built from South Sudan's IMWG/NBS/OCHA
          // administrative boundary release (CC BY 3.0 IGO).
          counties: { type: "geojson", data: "/geo/ssd_counties.geojson" },
        },
        layers: [
          { id: "satellite", type: "raster", source: "satellite" },
          {
            id: "county-line-halo", type: "line", source: "counties",
            paint: { "line-color": "#0B1220", "line-width": 2.4, "line-opacity": 0.35 },
          },
          {
            id: "county-line", type: "line", source: "counties",
            paint: { "line-color": "#FDE68A", "line-width": 1, "line-opacity": 0.9 },
          },
          {
            id: "county-selected-fill", type: "fill", source: "counties",
            paint: { "fill-color": "#FDE68A", "fill-opacity": 0.16 },
            filter: ["==", ["get", "county"], "__none__"],
          },
        ],
      },
      center: [30.5, 7.2],
      zoom: 5.4,
      minZoom: 5,
      maxZoom: 13,
      // South Sudan's bounding box, lightly padded — this is a national flood
      // dashboard, not a general-purpose map, so panning to another country
      // isn't a feature worth supporting.
      maxBounds: [
        [22.5, 2.5],
        [37.5, 13.0],
      ],
      attributionControl: false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }));
    popupRef.current = new maplibregl.Popup({
      closeButton: false, closeOnClick: false, offset: 14, className: "ssd-map-popup",
    });
    mapRef.current = map;
    flownToRef.current = null; // fresh map instance (StrictMode remounts this in dev) starts unflown
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  // Highlights the selected county's actual polygon, not just its marker dot.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const applyFilter = () => {
      if (map.getLayer("county-selected-fill")) {
        map.setFilter("county-selected-fill", ["==", ["get", "county"], selected ?? "__none__"]);
      }
    };
    if (map.isStyleLoaded()) applyFilter();
    else map.once("load", applyFilter);
  }, [selected]);

  useEffect(() => {
    const map = mapRef.current;
    const popup = popupRef.current;
    if (!map || !popup) return;

    const existing = markersRef.current;
    const seen = new Set<string>();

    for (const p of points) {
      seen.add(p.county);
      const size = 10 + p.value * 22;
      const isSelected = p.county === selected;

      let marker = existing[p.county];
      if (marker) marker.remove();

      // IMPORTANT: MapLibre positions the marker by writing a CSS transform
      // directly onto the element passed to `Marker`. Never touch
      // `el.style.transform` for hover effects — it silently overwrites that
      // positioning and the marker appears to vanish. Hover animation goes on
      // an inner wrapper instead.
      const el = document.createElement("div");
      el.style.cursor = "pointer";
      const dot = document.createElement("div");
      dot.style.width = `${size}px`;
      dot.style.height = `${size}px`;
      dot.style.borderRadius = "50%";
      dot.style.background = RISK_HEX[p.tier];
      dot.style.border = isSelected ? "2.5px solid white" : "1.5px solid rgba(255,255,255,0.6)";
      dot.style.boxShadow = isSelected
        ? `0 0 0 4px ${RISK_HEX[p.tier]}66, 0 2px 8px rgba(0,0,0,0.5)`
        : "0 1px 4px rgba(0,0,0,0.45)";
      dot.style.transition = "transform 120ms ease";
      dot.style.transformOrigin = "center";
      el.appendChild(dot);

      const showPopup = () => {
        dot.style.transform = "scale(1.2)";
        popup
          .setLngLat([p.lon, p.lat])
          .setHTML(
            `<div style="font:600 12.5px -apple-system,sans-serif;color:#0F172A;white-space:nowrap">${p.county}</div>` +
            `<div style="font:700 15px -apple-system,sans-serif;color:${RISK_HEX[p.tier]}">${(p.value * 100).toFixed(1)}%` +
            `<span style="font:600 10.5px -apple-system,sans-serif;color:#8792A2;text-transform:uppercase;margin-left:5px">${p.tier}</span></div>`
          )
          .addTo(map);
      };
      const hidePopup = () => {
        dot.style.transform = "scale(1)";
        popup.remove();
      };
      el.onmouseenter = showPopup;
      el.onmouseleave = hidePopup;
      el.onclick = () => onSelect(p.county);

      marker = new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat([p.lon, p.lat]).addTo(map);
      existing[p.county] = marker;
    }

    for (const key of Object.keys(existing)) {
      if (!seen.has(key)) {
        existing[key].remove();
        delete existing[key];
      }
    }

    // Locator mode: a single point should be framed, not left at the
    // national default view.
    if (focusZoom != null && points.length === 1 && flownToRef.current !== points[0].county) {
      flownToRef.current = points[0].county;
      map.flyTo({ center: [points[0].lon, points[0].lat], zoom: focusZoom, duration: 900 });
    }
  }, [points, selected, onSelect, focusZoom]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
