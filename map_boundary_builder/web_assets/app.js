const form = document.querySelector("#runForm");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const fileMeta = document.querySelector("#fileMeta");
const runButton = document.querySelector("#runButton");
const statusText = document.querySelector("#statusText");
const percentText = document.querySelector("#percentText");
const progressFill = document.querySelector("#progressFill");
const timeline = document.querySelector("#timeline");
const workspaceTitle = document.querySelector("#workspaceTitle");
const inputPreview = document.querySelector("#inputPreview");
const overlayPreview = document.querySelector("#overlayPreview");
const boundaryMapEl = document.querySelector("#boundaryMap");
const boundarySvg = document.querySelector("#boundarySvg");
const boundaryEmpty = document.querySelector("#boundaryEmpty");
const geojsonPane = document.querySelector("#geojsonPane");
const metricGrid = document.querySelector("#metricGrid");
const downloadLink = document.querySelector("#downloadLink");
const copyButton = document.querySelector("#copyButton");
const tabs = [...document.querySelectorAll(".tab")];
const panes = [...document.querySelectorAll(".pane")];

let selectedFile = null;
let latestGeojson = null;
let eventSource = null;
let boundaryMap = null;
let latestBoundaryBounds = null;

const BOUNDARY_SOURCE_ID = "generated-boundary";
const BOUNDARY_FILL_ID = "generated-boundary-fill";
const BOUNDARY_LINE_ID = "generated-boundary-line";

const stageLabels = {
  queued: "Queued",
  inspect: "Inspect",
  extract: "Extract",
  georeference: "Georeference",
  export: "Export",
  complete: "Complete",
  error: "Error",
};

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  setSelectedFile(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  setSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    setStatus("Choose an image", 0, "error");
    return;
  }

  resetRun();
  runButton.disabled = true;
  runButton.querySelector("span").textContent = "Running";

  const formData = new FormData(form);
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Run failed to start.");
    }
    if (payload.status === "complete" && payload.artifacts) {
      applyInlineRun(payload);
    } else {
      connectEvents(payload.id);
    }
  } catch (error) {
    finishWithError(error.message);
  }
});

tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

copyButton.addEventListener("click", async () => {
  if (!latestGeojson) return;
  await navigator.clipboard.writeText(JSON.stringify(latestGeojson, null, 2));
  copyButton.textContent = "Copied";
  setTimeout(() => {
    copyButton.textContent = "Copy";
  }, 1000);
});

function setSelectedFile(file) {
  selectedFile = file;
  if (!file) return;
  fileName.textContent = file.name;
  fileMeta.textContent = `${formatBytes(file.size)} · ${file.type || "image"}`;
  inputPreview.src = URL.createObjectURL(file);
  inputPreview.classList.add("ready");
  document.querySelector("#inputPane").classList.add("has-content");
  workspaceTitle.textContent = file.name;
}

function connectEvents(runId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/runs/${runId}/events`);
  eventSource.addEventListener("update", async (message) => {
    const event = JSON.parse(message.data);
    applyEvent(event);
    if (event.status === "complete") {
      eventSource.close();
      await loadArtifacts(runId);
      runButton.disabled = false;
      runButton.querySelector("span").textContent = "Run Boundary";
    }
    if (event.status === "error") {
      eventSource.close();
      finishWithError(event.message);
    }
  });
  eventSource.onerror = () => {
    if (eventSource.readyState === EventSource.CLOSED) return;
  };
}

function applyInlineRun(status) {
  for (const event of status.events || []) {
    applyEvent(event);
  }
  const artifacts = status.artifacts || {};
  if (artifacts.overlay_data_url) {
    overlayPreview.src = artifacts.overlay_data_url;
    overlayPreview.classList.add("ready");
    document.querySelector("#overlayPane").classList.add("has-content");
  }
  if (artifacts.geojson_inline) {
    latestGeojson = artifacts.geojson_inline;
    geojsonPane.textContent = JSON.stringify(latestGeojson, null, 2);
    downloadLink.href = URL.createObjectURL(
      new Blob([JSON.stringify(latestGeojson, null, 2)], {
        type: "application/geo+json",
      }),
    );
    downloadLink.download = "boundary.geojson";
    downloadLink.classList.remove("disabled");
    downloadLink.removeAttribute("aria-disabled");
    copyButton.disabled = false;
    renderBoundary(latestGeojson);
  }
  if (status.summary) {
    renderMetrics(status.summary);
    workspaceTitle.textContent = `${status.summary.city || status.city} boundary`;
  }
  setStatus("Boundary export ready", 100, "complete");
  activateTab(artifacts.overlay_data_url ? "overlay" : "boundary");
  runButton.disabled = false;
  runButton.querySelector("span").textContent = "Run Boundary";
}

function applyEvent(event) {
  const label = stageLabels[event.stage] || event.stage;
  const percent = Number(event.percent || 0);
  setStatus(event.message || label, percent, event.status);
  const item = document.createElement("li");
  item.className = event.status === "error" ? "active error" : "active";
  item.innerHTML = `<b>${escapeHtml(label)}</b>${escapeHtml(event.message || "")}`;
  timeline.prepend(item);
  while (timeline.children.length > 8) {
    timeline.lastElementChild.remove();
  }
}

function setStatus(message, percent, status = "running") {
  statusText.textContent = message;
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  percentText.textContent = `${clamped}%`;
  progressFill.style.width = `${clamped}%`;
  if (status === "error") {
    progressFill.style.background = "var(--coral)";
  } else {
    progressFill.style.background = "";
  }
}

async function loadArtifacts(runId) {
  const status = await fetchJson(`/api/runs/${runId}`);
  const artifacts = status.artifacts || {};
  if (artifacts.input) {
    inputPreview.src = artifacts.input;
    inputPreview.classList.add("ready");
  }
  if (artifacts.overlay) {
    overlayPreview.src = artifacts.overlay;
    overlayPreview.classList.add("ready");
    document.querySelector("#overlayPane").classList.add("has-content");
  }
  if (artifacts.geojson) {
    latestGeojson = await fetchJson(artifacts.geojson);
    geojsonPane.textContent = JSON.stringify(latestGeojson, null, 2);
    downloadLink.href = artifacts.geojson;
    downloadLink.classList.remove("disabled");
    downloadLink.removeAttribute("aria-disabled");
    copyButton.disabled = false;
    renderBoundary(latestGeojson);
  }
  if (status.summary) {
    renderMetrics(status.summary);
    workspaceTitle.textContent = `${status.summary.city || status.city} boundary`;
  }
  activateTab(artifacts.overlay ? "overlay" : "boundary");
}

function renderMetrics(summary) {
  const metrics = [
    ["Confidence", formatNumber(summary.combined_confidence, 3)],
    ["Source", summary.georeference_source],
    ["Coverage", `${formatNumber(summary.coverage_ratio * 100, 2)}%`],
    ["Control Points", summary.control_points],
    ["Residual P90", `${formatNumber(summary.p90_residual_m, 0)} m`],
    ["Meters / PX", formatNumber(summary.meters_per_pixel, 2)],
    ["Rotation", `${formatNumber(summary.rotation_degrees, 2)} deg`],
    ["Geometry", summary.geometry_type],
  ];
  metricGrid.innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "n/a")}</strong></div>`)
    .join("");
}

function renderBoundary(geojson) {
  document.querySelector("#boundaryPane").classList.add("has-content");
  if (window.maplibregl && boundaryMapEl) {
    boundarySvg.classList.remove("ready");
    boundarySvg.innerHTML = "";
    boundaryMapEl.classList.add("ready");
    if (document.querySelector("#boundaryPane").classList.contains("active")) {
      renderBoundaryMap(geojson);
    }
    return;
  }
  renderBoundarySvg(geojson);
}

function renderBoundaryMap(geojson) {
  if (!window.maplibregl || !boundaryMapEl) {
    renderBoundarySvg(geojson);
    return;
  }
  const bounds = geojsonBounds(geojson);
  if (!bounds) {
    boundaryMapEl.classList.remove("ready");
    return;
  }
  latestBoundaryBounds = bounds;
  boundaryMapEl.classList.add("ready");
  boundaryEmpty.hidden = true;

  if (!boundaryMap) {
    boundaryMap = new maplibregl.Map({
      container: boundaryMapEl,
      style: "/static/openfreemap-boundary.json",
      center: [(bounds.minLon + bounds.maxLon) / 2, (bounds.minLat + bounds.maxLat) / 2],
      zoom: 11,
      attributionControl: { compact: true },
    });
    boundaryMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  }

  const draw = () => {
    boundaryMap.resize();
    upsertBoundaryLayers(geojson);
    fitBoundaryMap(bounds);
  };

  if (boundaryMap.loaded()) {
    draw();
  } else {
    boundaryMap.once("load", draw);
  }
}

function upsertBoundaryLayers(geojson) {
  const source = boundaryMap.getSource(BOUNDARY_SOURCE_ID);
  if (source) {
    source.setData(geojson);
  } else {
    boundaryMap.addSource(BOUNDARY_SOURCE_ID, {
      type: "geojson",
      data: geojson,
    });
  }

  if (!boundaryMap.getLayer(BOUNDARY_FILL_ID)) {
    boundaryMap.addLayer({
      id: BOUNDARY_FILL_ID,
      type: "fill",
      source: BOUNDARY_SOURCE_ID,
      paint: {
        "fill-color": "#0e6f5c",
        "fill-opacity": 0.24,
      },
    });
  }

  if (!boundaryMap.getLayer(BOUNDARY_LINE_ID)) {
    boundaryMap.addLayer({
      id: BOUNDARY_LINE_ID,
      type: "line",
      source: BOUNDARY_SOURCE_ID,
      paint: {
        "line-color": "#0e6f5c",
        "line-width": 4,
        "line-opacity": 0.96,
      },
    });
  }
}

function fitBoundaryMap(bounds) {
  boundaryMap.fitBounds(
    [
      [bounds.minLon, bounds.minLat],
      [bounds.maxLon, bounds.maxLat],
    ],
    {
      padding: { top: 60, right: 60, bottom: 60, left: 60 },
      duration: 500,
      maxZoom: 14,
    },
  );
}

function renderBoundarySvg(geojson) {
  const rings = extractRings(geojson);
  if (!rings.length) {
    boundarySvg.innerHTML = "";
    return;
  }
  const points = rings.flat();
  const lons = points.map((point) => point[0]);
  const lats = points.map((point) => point[1]);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const width = 1000;
  const height = 620;
  const padding = 44;
  const lonSpan = maxLon - minLon || 1;
  const latSpan = maxLat - minLat || 1;
  const scale = Math.min((width - padding * 2) / lonSpan, (height - padding * 2) / latSpan);
  const usedWidth = lonSpan * scale;
  const usedHeight = latSpan * scale;
  const offsetX = (width - usedWidth) / 2;
  const offsetY = (height - usedHeight) / 2;

  const pathData = rings
    .map((ring) =>
      ring
        .map(([lon, lat], index) => {
          const x = offsetX + (lon - minLon) * scale;
          const y = offsetY + (maxLat - lat) * scale;
          return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
        })
        .join(" ") + " Z",
    )
    .join(" ");

  boundarySvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  boundarySvg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    <path d="${pathData}" fill="rgba(47, 178, 143, 0.22)" stroke="#0e6f5c" stroke-width="5" stroke-linejoin="round"></path>
    <circle cx="${offsetX}" cy="${offsetY + usedHeight}" r="5" fill="#c88719"></circle>
    <text x="${offsetX}" y="${offsetY + usedHeight + 28}" fill="#657069" font-size="20">${minLon.toFixed(4)}, ${minLat.toFixed(4)}</text>
  `;
  boundarySvg.classList.add("ready");
  document.querySelector("#boundaryPane").classList.add("has-content");
}

function geojsonBounds(geojson) {
  const rings = extractRings(geojson);
  const points = rings.flat();
  if (!points.length) return null;
  const lons = points.map((point) => point[0]);
  const lats = points.map((point) => point[1]);
  return {
    minLon: Math.min(...lons),
    maxLon: Math.max(...lons),
    minLat: Math.min(...lats),
    maxLat: Math.max(...lats),
  };
}

function extractRings(geojson) {
  const features = geojson.features || [];
  return features.flatMap((feature) => {
    const geometry = feature.geometry || {};
    if (geometry.type === "Polygon") return geometry.coordinates;
    if (geometry.type === "MultiPolygon") return geometry.coordinates.flat();
    return [];
  });
}

function activateTab(name) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  panes.forEach((pane) => pane.classList.toggle("active", pane.dataset.pane === name));
  if (name === "boundary" && latestGeojson) {
    window.setTimeout(() => renderBoundaryMap(latestGeojson), 0);
  } else if (boundaryMap && latestBoundaryBounds) {
    window.setTimeout(() => {
      boundaryMap.resize();
      fitBoundaryMap(latestBoundaryBounds);
    }, 0);
  }
}

function resetRun() {
  latestGeojson = null;
  timeline.innerHTML = "";
  metricGrid.innerHTML = "";
  overlayPreview.removeAttribute("src");
  overlayPreview.classList.remove("ready");
  document.querySelector("#overlayPane").classList.remove("has-content");
  boundaryMapEl.classList.remove("ready");
  boundarySvg.innerHTML = "";
  boundarySvg.classList.remove("ready");
  boundaryEmpty.hidden = false;
  latestBoundaryBounds = null;
  if (boundaryMap?.getSource(BOUNDARY_SOURCE_ID)) {
    boundaryMap.getSource(BOUNDARY_SOURCE_ID).setData({ type: "FeatureCollection", features: [] });
  }
  geojsonPane.textContent = "{}";
  downloadLink.href = "#";
  downloadLink.classList.add("disabled");
  downloadLink.setAttribute("aria-disabled", "true");
  copyButton.disabled = true;
  setStatus("Starting", 0);
}

function finishWithError(message) {
  setStatus(message, 0, "error");
  runButton.disabled = false;
  runButton.querySelector("span").textContent = "Run Boundary";
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits > 0 ? Math.min(1, digits) : 0,
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
