const form = document.querySelector("#runForm");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const compactDropZone = document.querySelector("#compactDropZone");
const dropTargets = [...document.querySelectorAll("[data-drop-target]")];
const dropTitle = document.querySelector("#dropTitle");
const dropMeta = document.querySelector("#dropMeta");
const fileName = document.querySelector("#fileName");
const fileMeta = document.querySelector("#fileMeta");
const runButton = document.querySelector("#runButton");
const statusText = document.querySelector("#statusText");
const percentText = document.querySelector("#percentText");
const progressMeter = document.querySelector("#progressMeter");
const progressFill = document.querySelector("#progressFill");
const progressNote = document.querySelector("#progressNote");
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
const historyList = document.querySelector("#historyList");
const historyEmpty = document.querySelector("#historyEmpty");
const tabs = [...document.querySelectorAll(".tab")];
const panes = [...document.querySelectorAll(".pane")];

let selectedFile = null;
let latestGeojson = null;
let eventSource = null;
let boundaryMap = null;
let latestBoundaryBounds = null;
let progressValue = 0;
let activeProgressStep = null;
let stepStates = new Map();
let historyEntries = [];
let activeHistoryId = null;

const BOUNDARY_SOURCE_ID = "generated-boundary";
const BOUNDARY_FILL_ID = "generated-boundary-fill";
const BOUNDARY_LINE_ID = "generated-boundary-line";
const HISTORY_STORAGE_KEY = "mapBoundaryBuilder.history.v1";
const MAX_HISTORY_ENTRIES = 14;
const MAX_HISTORY_BYTES = 4_400_000;

const stageLabels = {
  queued: "Queued",
  inspect: "Inspect",
  extract: "Extract",
  georeference: "Georeference",
  export: "Export",
  complete: "Complete",
  error: "Error",
};

const progressSteps = [
  {
    key: "labels",
    title: "Read labels",
    shortTitle: "Labels",
    idle: "Waiting for a screenshot",
    running: "Browser OCR is scanning map text.",
    done: "Map labels captured.",
  },
  {
    key: "extract",
    title: "Extract area",
    shortTitle: "Area",
    idle: "Waiting for image pixels",
    running: "Finding the colored service boundary.",
    done: "Service pixels traced.",
  },
  {
    key: "georeference",
    title: "Locate map",
    shortTitle: "Locate",
    idle: "Waiting for map evidence",
    running: "Matching labels and roads to geography.",
    done: "Map transform fitted.",
  },
  {
    key: "export",
    title: "Build export",
    shortTitle: "Export",
    idle: "Waiting for transform",
    running: "Writing GeoJSON and previews.",
    done: "GeoJSON ready.",
  },
];

resetProgressSteps();
renderProgressSteps();
historyEntries = loadHistoryEntries();
renderHistory();
updateRunButton();

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  setSelectedFile(file);
});

dropTargets.forEach((target) => {
  target.addEventListener("dragover", (event) => {
    event.preventDefault();
    target.classList.add("dragging");
  });

  target.addEventListener("dragleave", () => target.classList.remove("dragging"));

  target.addEventListener("drop", (event) => {
    event.preventDefault();
    target.classList.remove("dragging");
    const [file] = event.dataTransfer.files;
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
    setSelectedFile(file);
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    setStatus("Choose an image", 0, "error", {
      note: "Drop a screenshot in the workspace first.",
    });
    return;
  }

  resetRun();
  runButton.disabled = true;
  runButton.querySelector("span").textContent = "Running";

  const formData = new FormData(form);
  formData.set("image", selectedFile, selectedFile.name);
  try {
    const clientLabels = await runClientOcr(selectedFile);
    if (clientLabels.length) {
      formData.append("ocr_labels", JSON.stringify(clientLabels));
    }
    markProgressStep("labels", "done");
    setStatus("Sending image to builder", 42, "running", {
      step: "extract",
      note: "The backend is starting extraction and georeferencing.",
    });
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

historyList.addEventListener("click", (event) => {
  const menuSummary = event.target.closest(".history-menu summary");
  if (menuSummary) {
    window.setTimeout(() => {
      const menu = menuSummary.closest(".history-menu");
      if (menu?.open) openHistoryMenu(menu);
    }, 0);
    return;
  }

  const actionButton = event.target.closest("[data-history-action]");
  if (actionButton) {
    event.preventDefault();
    const item = actionButton.closest("[data-history-id]");
    const id = item?.dataset.historyId;
    if (!id) return;
    const action = actionButton.dataset.historyAction;
    if (action === "star") toggleHistoryStar(id);
    if (action === "delete") deleteHistoryEntry(id);
    return;
  }

  const loadButton = event.target.closest("[data-history-load]");
  if (loadButton) {
    const entry = historyEntries.find((item) => item.id === loadButton.dataset.historyLoad);
    if (entry) restoreHistoryEntry(entry);
  }
});

historyList.addEventListener("toggle", (event) => {
  const menu = event.target;
  if (!menu.matches?.(".history-menu") || !menu.open) return;
  openHistoryMenu(menu);
}, true);

historyList.addEventListener("scroll", closeHistoryMenus);
window.addEventListener("resize", closeHistoryMenus);

document.addEventListener("click", (event) => {
  if (event.target.closest(".history-menu")) return;
  closeHistoryMenus();
});

function setSelectedFile(file) {
  selectedFile = file;
  if (!file) return;
  activeHistoryId = null;
  progressValue = 0;
  resetProgressSteps();
  stepStates.set("labels", {
    state: "pending",
    message: "Ready to scan map text.",
  });
  renderProgressSteps();
  clearGeneratedArtifacts();
  fileName.textContent = file.name;
  fileMeta.textContent = `${formatBytes(file.size)} · ${file.type || "image"}`;
  dropTitle.textContent = file.name;
  dropMeta.textContent = `${formatBytes(file.size)} · ${file.type || "image"}`;
  inputPreview.src = URL.createObjectURL(file);
  inputPreview.classList.add("ready");
  document.querySelector("#inputPane").classList.add("has-content");
  dropZone.classList.add("has-file");
  compactDropZone.classList.add("has-file");
  workspaceTitle.textContent = file.name;
  updateRunButton();
  setStatus("Image ready", 0, "idle", {
    note: "Review settings, then run the boundary export.",
  });
  renderHistory();
  activateTab("input");
}

async function runClientOcr(file) {
  if (!window.Tesseract || !file) return [];
  try {
    const labels = [];
    markProgressStep("labels", "running", "Browser OCR is scanning map text.");
    setStatus("Reading map labels", 6, "running", {
      step: "labels",
      note: "This happens locally in your browser before upload.",
    });
    const result = await recognizeImageLabels(file, "Reading map labels in browser", 4, 20);
    labels.push(...labelsFromOcrResult(result));

    if (countUsefulLabels(labels) < 30) {
      const canvas = await imageFileToCanvas(file);
      const enhanced = makeOcrCanvas(canvas, "neutral-dark");
      setStatus("Enhancing map labels", 26, "running", {
        step: "labels",
        note: "Trying a higher-contrast OCR pass for faint labels.",
      });
      const enhancedResult = await recognizeImageLabels(enhanced, "Enhancing map labels in browser", 24, 12);
      labels.push(...labelsFromOcrResult(enhancedResult));
    }

    if (countUsefulLabels(labels) < 30) {
      const canvas = await imageFileToCanvas(file);
      const neutral = makeOcrCanvas(canvas, "neutral");
      setStatus("Checking map labels again", 34, "running", {
        step: "labels",
        note: "One final pass helps sparse or low-contrast screenshots.",
      });
      const neutralResult = await recognizeImageLabels(neutral, "Checking map labels again", 32, 8);
      labels.push(...labelsFromOcrResult(neutralResult));
    }

    return dedupeClientLabels(labels);
  } catch (error) {
    console.warn("Browser OCR unavailable", error);
    return [];
  }
}

async function recognizeImageLabels(image, statusMessage, start, span) {
  return window.Tesseract.recognize(image, "eng", {
    tessedit_pageseg_mode: "11",
    logger: (event) => {
      if (event.status === "recognizing text") {
        setStatus(statusMessage, start + Math.round((event.progress || 0) * span), "running", {
          step: "labels",
        });
      }
    },
  });
}

function labelsFromOcrResult(result) {
  const words = Array.isArray(result?.data?.words)
    ? result.data.words
    : parseTesseractTsv(result?.data?.tsv || "");
  return words.map(wordToLabel).filter(Boolean);
}

function imageFileToCanvas(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0);
      URL.revokeObjectURL(image.src);
      resolve(canvas);
    };
    image.onerror = reject;
    image.src = URL.createObjectURL(file);
  });
}

function makeOcrCanvas(source, mode) {
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  const outputContext = canvas.getContext("2d", { willReadFrequently: true });
  const image = sourceContext.getImageData(0, 0, source.width, source.height);
  const data = image.data;
  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];
    const serviceFill = isServiceFillPixel(red, green, blue);
    const nextRed = serviceFill ? 245 : red;
    const nextGreen = serviceFill ? 245 : green;
    const nextBlue = serviceFill ? 245 : blue;
    if (mode === "neutral-dark") {
      const gray = Math.round(0.299 * nextRed + 0.587 * nextGreen + 0.114 * nextBlue);
      const value = gray < 125 ? 0 : 255;
      data[index] = value;
      data[index + 1] = value;
      data[index + 2] = value;
    } else {
      data[index] = nextRed;
      data[index + 1] = nextGreen;
      data[index + 2] = nextBlue;
    }
    data[index + 3] = 255;
  }
  outputContext.putImageData(image, 0, 0);
  return canvas;
}

function isServiceFillPixel(red, green, blue) {
  const brightBlue = blue >= 135 && green >= 80 && red <= 110 && blue - red >= 55;
  const greenFill = green >= 120 && green > red + 25 && green > blue + 5;
  return brightBlue || greenFill;
}

function countUsefulLabels(labels) {
  return labels.filter((label) => isUsefulLabelText(label.text)).length;
}

function isUsefulLabelText(text) {
  const compact = String(text || "").replace(/\s+/g, "");
  if (compact.length < 3 || /^\d+$/.test(compact)) return false;
  const letters = (compact.match(/[A-Za-z]/g) || []).length;
  return letters >= Math.max(3, Math.floor(compact.length / 2));
}

function dedupeClientLabels(labels) {
  const best = new Map();
  for (const label of labels) {
    const key = `${label.text.toLowerCase()}|${Math.round(label.x / 20)}|${Math.round(label.y / 20)}`;
    const old = best.get(key);
    if (!old || label.confidence > old.confidence) {
      best.set(key, label);
    }
  }
  return [...best.values()].sort((a, b) => b.confidence - a.confidence);
}

function parseTesseractTsv(tsv) {
  const lines = String(tsv || "").trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split("\t");
  const index = Object.fromEntries(headers.map((header, position) => [header, position]));
  return lines.slice(1).map((line) => {
    const cells = line.split("\t");
    return {
      text: cells[index.text] || "",
      left: Number(cells[index.left] || 0),
      top: Number(cells[index.top] || 0),
      width: Number(cells[index.width] || 0),
      height: Number(cells[index.height] || 0),
      confidence: Number(cells[index.conf] || 0),
    };
  });
}

function wordToLabel(word) {
  const text = String(word?.text || "").trim();
  const box = word?.bbox || {};
  const x0 = Number(box.x0 ?? word?.x0 ?? word?.left ?? 0);
  const y0 = Number(box.y0 ?? word?.y0 ?? word?.top ?? 0);
  const x1 = Number(box.x1 ?? word?.x1 ?? x0 + Number(word?.width || 0));
  const y1 = Number(box.y1 ?? word?.y1 ?? y0 + Number(word?.height || 0));
  const width = x1 - x0;
  const height = y1 - y0;
  if (!text || width <= 0 || height <= 0) return null;
  return {
    text,
    x: x0 + width / 2,
    y: y0 + height / 2,
    width,
    height,
    confidence: Number(word?.confidence ?? word?.conf ?? 50),
  };
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
        markAllProgressStepsDone();
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
  markAllProgressStepsDone();
  activateTab(artifacts.overlay_data_url ? "overlay" : "boundary");
  queueHistorySave({
    id: status.id,
    filename: status.filename,
    city: status.city,
    summary: status.summary,
    geojson: latestGeojson,
    overlaySrc: artifacts.overlay_data_url,
  });
  runButton.disabled = false;
  runButton.querySelector("span").textContent = "Run Boundary";
}

function applyEvent(event) {
  const label = stageLabels[event.stage] || event.stage;
  const step = progressStepForEvent(event);
  if (event.status === "complete") {
    markAllProgressStepsDone();
  } else if (event.status === "error") {
    markProgressStep(step || activeProgressStep || "georeference", "error", event.message || label);
  } else if (step) {
    markPreviousProgressStepsDone(step);
    markProgressStep(step, "running", humanProgressMessage(event));
  }
  setStatus(event.message || label, progressPercentForEvent(event), event.status, {
    step,
    note: humanProgressNote(event),
  });
}

function setStatus(message, percent, status = "running", options = {}) {
  statusText.textContent = message;
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  progressValue = status === "complete" ? 100 : Math.max(progressValue, clamped);
  const step = options.step || activeProgressStep;
  percentText.textContent = progressLabel(status, step);
  progressNote.textContent = options.note || defaultProgressNote(status, step);
  progressFill.style.width = `${progressValue}%`;
  progressMeter.setAttribute("aria-valuenow", String(progressValue));
  if (status === "error") {
    progressFill.style.background = "var(--coral)";
    progressMeter.classList.add("error");
  } else {
    progressFill.style.background = "";
    progressMeter.classList.remove("error");
  }
}

function resetProgressSteps() {
  stepStates = new Map(
    progressSteps.map((step) => [
      step.key,
      {
        state: "pending",
        message: step.idle,
      },
    ]),
  );
  activeProgressStep = null;
  renderProgressSteps();
}

function markProgressStep(key, state, message) {
  if (!key || !stepStates.has(key)) return;
  activeProgressStep = state === "done" ? activeProgressStep : key;
  const step = progressSteps.find((item) => item.key === key);
  stepStates.set(key, {
    state,
    message: message || step?.[state] || step?.running || "",
  });
  renderProgressSteps();
}

function markPreviousProgressStepsDone(key) {
  const index = progressSteps.findIndex((step) => step.key === key);
  if (index < 0) return;
  for (const step of progressSteps.slice(0, index)) {
    const old = stepStates.get(step.key);
    if (old?.state !== "done") {
      stepStates.set(step.key, {
        state: "done",
        message: step.done,
      });
    }
  }
  renderProgressSteps();
}

function markAllProgressStepsDone() {
  for (const step of progressSteps) {
    stepStates.set(step.key, {
      state: "done",
      message: step.done,
    });
  }
  activeProgressStep = "export";
  renderProgressSteps();
}

function renderProgressSteps() {
  timeline.innerHTML = progressSteps
    .map((step) => {
      const status = stepStates.get(step.key) || { state: "pending", message: step.idle };
      return `
        <li class="${escapeHtml(status.state)}">
          <span class="step-dot" aria-hidden="true"></span>
          <span>
            <b>${escapeHtml(step.shortTitle || step.title)}</b>
            <small>${escapeHtml(progressStateLabel(status.state))}</small>
          </span>
        </li>
      `;
    })
    .join("");
}

function progressStateLabel(state) {
  if (state === "running") return "Active";
  if (state === "done") return "Done";
  if (state === "error") return "Issue";
  return "Waiting";
}

function progressStepForEvent(event) {
  return {
    queued: "extract",
    inspect: "extract",
    extract: "extract",
    georeference: "georeference",
    export: "export",
    complete: "export",
  }[event.stage] || activeProgressStep;
}

function progressPercentForEvent(event) {
  const percent = Number(event.percent || 0);
  if (event.status === "complete" || event.stage === "complete") return 100;
  if (event.status === "error") return progressValue;
  if (event.stage === "queued") return 42;
  if (event.stage === "inspect") return 46;
  if (event.stage === "extract") return percent < 30 ? 54 : 64;
  if (event.stage === "georeference") {
    if (percent < 60) return 74;
    if (percent < 75) return 82;
    return 88;
  }
  if (event.stage === "export") return 94;
  return percent;
}

function humanProgressMessage(event) {
  if (event.stage === "queued") return "Run queued.";
  if (event.stage === "inspect") return "Reading image size and metadata.";
  if (event.stage === "extract") return event.percent < 30 ? "Finding service-area pixels." : "Service pixels traced.";
  if (event.stage === "georeference") return "Matching map labels, roads, and geography.";
  if (event.stage === "export") return "Writing GeoJSON and previews.";
  return event.message || stageLabels[event.stage] || "";
}

function humanProgressNote(event) {
  if (event.stage === "queued" || event.stage === "inspect") {
    return "The image is being prepared on the backend.";
  }
  if (event.stage === "extract") {
    return "The colored service-area shape is being separated from the map.";
  }
  if (event.stage === "georeference") {
    return "Large regions can take longer while road alignment is checked.";
  }
  if (event.stage === "export") {
    return "The boundary, overlay, and download are being finalized.";
  }
  if (event.stage === "complete") {
    return "GeoJSON and previews are ready.";
  }
  return "";
}

function progressLabel(status, stepKey) {
  if (status === "complete") return "Done";
  if (status === "error") return "Issue";
  const index = progressSteps.findIndex((step) => step.key === stepKey);
  return index >= 0 ? `Step ${index + 1}/${progressSteps.length}` : selectedFile ? "Ready" : "Idle";
}

function defaultProgressNote(status, stepKey) {
  if (status === "complete") return "GeoJSON and previews are ready.";
  if (status === "error") return "The run needs attention.";
  const step = progressSteps.find((item) => item.key === stepKey);
  if (!step) return selectedFile ? "Review settings, then run the boundary export." : "Add a map screenshot to start.";
  return stepStates.get(step.key)?.message || step.running;
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
  queueHistorySave({
    id: status.id,
    filename: status.filename,
    city: status.city,
    summary: status.summary,
    geojson: latestGeojson,
    overlaySrc: artifacts.overlay,
  });
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

function queueHistorySave(payload) {
  if (!payload.geojson) return;
  saveHistoryEntry(payload).catch((error) => {
    console.warn("Could not save generation history", error);
  });
}

async function saveHistoryEntry(payload) {
  const existing = historyEntries.find((entry) => entry.id === String(payload.id));
  const title = historyTitle(payload);
  const [inputImage, overlayImage] = await Promise.all([
    imageUrlToStoredDataUrl(inputPreview.src),
    imageUrlToStoredDataUrl(payload.overlaySrc),
  ]);
  const entry = {
    id: String(payload.id || Date.now()),
    title,
    filename: payload.filename || selectedFile?.name || "Map screenshot",
    city: payload.summary?.city || payload.city || "Auto",
    createdAt: existing?.createdAt || Date.now(),
    starred: Boolean(existing?.starred),
    summary: payload.summary || null,
    geojson: payload.geojson,
    inputImage,
    overlayImage,
  };
  upsertHistoryEntry(entry);
}

function historyTitle(payload) {
  const city = payload.summary?.city || payload.city;
  if (city && city !== "Auto") return `${city} boundary`;
  if (payload.filename) return payload.filename;
  return selectedFile?.name || "Generated boundary";
}

function upsertHistoryEntry(entry) {
  activeHistoryId = entry.id;
  historyEntries = [entry, ...historyEntries.filter((item) => item.id !== entry.id)];
  persistHistoryEntries();
  renderHistory();
}

function loadHistoryEntries() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return sortHistoryEntries(
      parsed.filter((entry) => entry && entry.id && entry.geojson).map((entry) => ({
        ...entry,
        id: String(entry.id),
        createdAt: Number(entry.createdAt) || Date.now(),
        starred: Boolean(entry.starred),
      })),
    );
  } catch (error) {
    console.warn("Could not load generation history", error);
    return [];
  }
}

function persistHistoryEntries() {
  let entries = sortHistoryEntries(historyEntries).slice(0, MAX_HISTORY_ENTRIES);
  entries = trimHistoryEntries(entries);
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries));
    historyEntries = entries;
  } catch (error) {
    const lighterEntries = trimHistoryEntries(entries.map((entry) => ({
      ...entry,
      inputImage: null,
      overlayImage: null,
    })));
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(lighterEntries));
      historyEntries = lighterEntries;
    } catch (retryError) {
      console.warn("Could not persist generation history", retryError);
    }
  }
}

function trimHistoryEntries(entries) {
  const next = [...entries];
  while (next.length > 1 && JSON.stringify(next).length > MAX_HISTORY_BYTES) {
    let removeIndex = next.length - 1;
    for (let index = next.length - 1; index >= 0; index -= 1) {
      if (!next[index].starred) {
        removeIndex = index;
        break;
      }
    }
    next.splice(removeIndex, 1);
  }
  return next;
}

function sortHistoryEntries(entries) {
  return [...entries].sort((a, b) => {
    if (Boolean(a.starred) !== Boolean(b.starred)) return a.starred ? -1 : 1;
    return Number(b.createdAt || 0) - Number(a.createdAt || 0);
  });
}

function renderHistory() {
  const entries = sortHistoryEntries(historyEntries);
  historyEmpty.hidden = entries.length > 0;
  historyList.hidden = entries.length === 0;
  historyList.innerHTML = entries.map(renderHistoryEntry).join("");
}

function renderHistoryEntry(entry) {
  const thumb = entry.inputImage || entry.overlayImage;
  const detail = historyDetail(entry);
  const starred = entry.starred ? `<span class="history-star" aria-label="Starred"></span>` : "";
  const classes = [
    "history-item",
    entry.starred ? "starred" : "",
    entry.id === activeHistoryId ? "active" : "",
  ].filter(Boolean).join(" ");
  return `
    <li class="${escapeHtml(classes)}" data-history-id="${escapeHtml(entry.id)}">
      <button class="history-main" type="button" data-history-load="${escapeHtml(entry.id)}">
        <span class="history-thumb">${thumb ? `<img src="${escapeHtml(thumb)}" alt="" />` : ""}</span>
        <span class="history-copy">
          <strong>${starred}${escapeHtml(entry.title)}</strong>
          <span class="history-meta">${escapeHtml(formatHistoryTime(entry.createdAt))}</span>
          <span class="history-detail">${escapeHtml(detail)}</span>
        </span>
      </button>
      <details class="history-menu">
        <summary aria-label="Generation actions">...</summary>
        <div class="history-menu-panel">
          <button type="button" data-history-action="star">${entry.starred ? "Unstar" : "Star"}</button>
          <button type="button" data-history-action="delete">Delete</button>
        </div>
      </details>
    </li>
  `;
}

function historyDetail(entry) {
  const summary = entry.summary || {};
  const pieces = [];
  if (summary.combined_confidence != null) pieces.push(`${formatNumber(summary.combined_confidence, 2)} confidence`);
  if (summary.control_points != null) pieces.push(`${summary.control_points} points`);
  if (summary.geometry_type) pieces.push(summary.geometry_type);
  return pieces.join(" · ") || entry.filename || "GeoJSON saved";
}

function formatHistoryTime(timestamp) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(Number(timestamp) || Date.now()));
}

function toggleHistoryStar(id) {
  historyEntries = historyEntries.map((entry) => (
    entry.id === id ? { ...entry, starred: !entry.starred } : entry
  ));
  persistHistoryEntries();
  renderHistory();
  closeHistoryMenus();
}

function deleteHistoryEntry(id) {
  if (activeHistoryId === id) activeHistoryId = null;
  historyEntries = historyEntries.filter((entry) => entry.id !== id);
  persistHistoryEntries();
  renderHistory();
}

function restoreHistoryEntry(entry) {
  closeHistoryMenus();
  activeHistoryId = entry.id;
  selectedFile = null;
  imageInput.value = "";
  clearGeneratedArtifacts();
  latestGeojson = entry.geojson;
  geojsonPane.textContent = JSON.stringify(latestGeojson, null, 2);
  downloadLink.href = URL.createObjectURL(
    new Blob([JSON.stringify(latestGeojson, null, 2)], {
      type: "application/geo+json",
    }),
  );
  downloadLink.download = `${entry.title || "boundary"}.geojson`;
  downloadLink.classList.remove("disabled");
  downloadLink.removeAttribute("aria-disabled");
  copyButton.disabled = false;

  if (entry.inputImage) {
    inputPreview.src = entry.inputImage;
    inputPreview.classList.add("ready");
    document.querySelector("#inputPane").classList.add("has-content");
  } else {
    inputPreview.removeAttribute("src");
    inputPreview.classList.remove("ready");
    document.querySelector("#inputPane").classList.remove("has-content");
  }

  if (entry.overlayImage) {
    overlayPreview.src = entry.overlayImage;
    overlayPreview.classList.add("ready");
    document.querySelector("#overlayPane").classList.add("has-content");
  }

  fileName.textContent = entry.title;
  fileMeta.textContent = "Loaded from local history";
  dropTitle.textContent = "Drop map screenshot";
  dropMeta.textContent = "PNG, JPG, WebP, TIFF";
  dropZone.classList.remove("has-file");
  compactDropZone.classList.add("has-file");
  workspaceTitle.textContent = entry.title;
  if (entry.summary) renderMetrics(entry.summary);
  renderBoundary(latestGeojson);
  markAllProgressStepsDone();
  setStatus("Loaded from history", 100, "complete", {
    note: "Previous GeoJSON and previews are restored locally.",
  });
  activateTab("boundary");
  updateRunButton();
  renderHistory();
}

function closeHistoryMenus() {
  document.querySelectorAll(".history-menu[open]").forEach((menu) => {
    menu.open = false;
    menu.style.removeProperty("--history-menu-left");
    menu.style.removeProperty("--history-menu-top");
  });
}

function openHistoryMenu(menu) {
  document.querySelectorAll(".history-menu[open]").forEach((openMenu) => {
    if (openMenu !== menu) openMenu.open = false;
  });
  positionHistoryMenu(menu);
}

function positionHistoryMenu(menu) {
  const summary = menu.querySelector("summary");
  if (!summary) return;
  const rect = summary.getBoundingClientRect();
  const panelWidth = 120;
  const panelHeight = 84;
  const margin = 8;
  const left = Math.min(
    window.innerWidth - panelWidth - margin,
    Math.max(margin, rect.right - panelWidth),
  );
  let top = rect.bottom + 4;
  if (top + panelHeight > window.innerHeight - margin) {
    top = Math.max(margin, rect.top - panelHeight - 4);
  }
  menu.style.setProperty("--history-menu-left", `${left}px`);
  menu.style.setProperty("--history-menu-top", `${top}px`);
}

function imageUrlToStoredDataUrl(src) {
  return new Promise((resolve) => {
    if (!src) {
      resolve(null);
      return;
    }
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      try {
        const maxSize = 520;
        const scale = Math.min(1, maxSize / Math.max(image.naturalWidth, image.naturalHeight));
        const width = Math.max(1, Math.round(image.naturalWidth * scale));
        const height = Math.max(1, Math.round(image.naturalHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        context.fillStyle = "#f8f6ee";
        context.fillRect(0, 0, width, height);
        context.drawImage(image, 0, 0, width, height);
        resolve(canvas.toDataURL("image/jpeg", 0.78));
      } catch (error) {
        console.warn("Could not snapshot history image", error);
        resolve(null);
      }
    };
    image.onerror = () => resolve(null);
    image.src = src;
  });
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
  progressValue = 0;
  resetProgressSteps();
  markProgressStep("labels", "running", "Browser OCR is scanning map text.");
  clearGeneratedArtifacts();
  setStatus("Preparing image", 2, "running", {
    step: "labels",
    note: "Reading labels locally before the backend run starts.",
  });
}

function clearGeneratedArtifacts() {
  latestGeojson = null;
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
}

function finishWithError(message) {
  markProgressStep(activeProgressStep || "georeference", "error", message);
  setStatus(message, progressValue, "error", {
    note: "The run stopped before a reliable boundary could be exported.",
  });
  runButton.disabled = false;
  runButton.querySelector("span").textContent = "Run Boundary";
  updateRunButton();
}

function updateRunButton() {
  if (runButton.disabled && runButton.querySelector("span").textContent === "Running") return;
  runButton.disabled = !selectedFile;
  runButton.querySelector("span").textContent = selectedFile ? "Run Boundary" : "Add image first";
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
