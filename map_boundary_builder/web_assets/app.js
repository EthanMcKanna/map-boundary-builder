const form = document.querySelector("#runForm");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const dropTargets = [...document.querySelectorAll("[data-drop-target]")];
const dropTitle = document.querySelector("#dropTitle");
const dropMeta = document.querySelector("#dropMeta");
const brandButton = document.querySelector("#brandButton");
const brandHomeLink = document.querySelector("#brandHomeLink");
const brandMark = document.querySelector(".brand-mark");
const runButton = document.querySelector("#runButton");
const runButtonLabel = runButton.querySelector(".primary-action-label");
const progressPanel = document.querySelector("#progressPanel");
const statusText = document.querySelector("#statusText");
const percentText = document.querySelector("#percentText");
const progressMeter = document.querySelector("#progressMeter");
const progressFill = document.querySelector("#progressFill");
const progressNote = document.querySelector("#progressNote");
const timeline = document.querySelector("#timeline");
const reportPanel = document.querySelector("#reportPanel");
const reportText = document.querySelector("#reportText");
const reportButton = document.querySelector("#reportButton");
const reportLink = document.querySelector("#reportLink");
const reportTrigger = document.querySelector("#reportTrigger");
const reportDialog = document.querySelector("#reportDialog");
const reportForm = document.querySelector("#reportForm");
const reportIssueType = document.querySelector("#reportIssueType");
const reportUserNote = document.querySelector("#reportUserNote");
const reportCloseButton = document.querySelector("#reportCloseButton");
const reportCancelButton = document.querySelector("#reportCancelButton");
const reportSubmitButton = document.querySelector("#reportSubmitButton");
const reportFormStatus = document.querySelector("#reportFormStatus");
const reportIssueLink = document.querySelector("#reportIssueLink");
const workspaceTitle = document.querySelector("#workspaceTitle");
const inputPane = document.querySelector("#inputPane");
const imageToggle = document.querySelector("#imageToggle");
const imageModeButtons = [...document.querySelectorAll("[data-image-mode]")];
const inputPreview = document.querySelector("#inputPreview");
const overlayPreview = document.querySelector("#overlayPreview");
const boundaryMapEl = document.querySelector("#boundaryMap");
const boundarySvg = document.querySelector("#boundarySvg");
const boundaryEmpty = document.querySelector("#boundaryEmpty");
const geojsonPane = document.querySelector("#geojsonPane");
const outputMenu = document.querySelector("#outputMenu");
const downloadLink = document.querySelector("#downloadLink");
const copyButton = document.querySelector("#copyButton");
const historyList = document.querySelector("#historyList");
const historyEmpty = document.querySelector("#historyEmpty");
const settingsButton = document.querySelector("#settingsButton");
const settingsDialog = document.querySelector("#settingsDialog");
const settingsCloseButton = document.querySelector("#settingsCloseButton");
const tabs = [...document.querySelectorAll(".tab")];
const panes = [...document.querySelectorAll(".pane")];
const themeModeButtons = [...document.querySelectorAll("[data-theme-mode]")];
const themeColorMeta = document.querySelector('meta[name="theme-color"]');
const iconLinks = [...document.querySelectorAll('link[rel="icon"], link[rel="apple-touch-icon"]')];

let selectedFile = null;
let latestGeojson = null;
let eventSource = null;
let boundaryMap = null;
let latestBoundaryBounds = null;
let currentBoundaryMapStyle = null;
let progressValue = 0;
let estimatedProgressTimer = null;
let estimatedProgressStartedAt = 0;
let activeProgressStep = null;
let stepStates = new Map();
let historyEntries = [];
let activeHistoryId = null;
let renamingHistoryId = null;
let latestRunId = null;
let latestRunError = null;
let latestRunEvents = [];
let latestRunStatus = "idle";
let latestRunSummary = null;
let activeReportStatus = "completed";
let copyFeedbackTimeout = null;
let activeImageMode = "original";

const BOUNDARY_SOURCE_ID = "generated-boundary";
const BOUNDARY_FILL_ID = "generated-boundary-fill";
const BOUNDARY_LINE_ID = "generated-boundary-line";
const HISTORY_STORAGE_KEY = "mapBoundaryBuilder.history.v1";
const THEME_STORAGE_KEY = "mapBoundaryBuilder.theme.v1";
const THEME_MODES = new Set(["system", "light", "dark"]);
const RUN_BUTTON_LABELS = {
  empty: "Choose image",
  ready: "Build boundary",
  running: "Building",
};
const EMPTY_DROP_TITLE = "Drop or paste map screenshot";
const EMPTY_DROP_META = "PNG, JPG, WebP, TIFF, SVG";
const CLIPBOARD_IMAGE_EXTENSIONS = new Map([
  ["image/png", "png"],
  ["image/jpeg", "jpg"],
  ["image/webp", "webp"],
  ["image/gif", "gif"],
  ["image/tiff", "tiff"],
  ["image/svg+xml", "svg"],
]);
const MAX_HISTORY_ENTRIES = 14;
const MAX_HISTORY_BYTES = 4_400_000;
const MAX_HISTORY_TITLE_LENGTH = 80;
const COPY_BUTTON_IDLE_HTML = copyButton.innerHTML;
const systemThemeMedia = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
const iconAssets = {
  light: "/static/boundary-builder-icon.png",
  dark: "/static/boundary-builder-icon-dark.png",
};
let activeThemeMode = loadThemeMode();

const stageLabels = {
  queued: "Queued",
  ocr: "OCR",
  inspect: "Inspect",
  extract: "Extract",
  georeference: "Georeference",
  export: "Export",
  complete: "Complete",
  error: "Error",
};

const progressSteps = [
  {
    key: "prepare",
    title: "Prepare",
    shortTitle: "Prep",
    idle: "Waiting",
    running: "Preparing image.",
    done: "Image ready.",
  },
  {
    key: "extract",
    title: "Trace area",
    shortTitle: "Area",
    idle: "Waiting",
    running: "Tracing service pixels.",
    done: "Area traced.",
  },
  {
    key: "labels",
    title: "Read text",
    shortTitle: "Text",
    idle: "Waiting",
    running: "Reading map labels.",
    done: "Labels read.",
  },
  {
    key: "georeference",
    title: "Fit map",
    shortTitle: "Fit",
    idle: "Waiting",
    running: "Matching map evidence.",
    done: "Map fitted.",
  },
  {
    key: "export",
    title: "Export",
    shortTitle: "Export",
    idle: "Waiting",
    running: "Writing files.",
    done: "GeoJSON ready.",
  },
];

const estimatedProgressStages = [
  {
    afterMs: 0,
    key: "prepare",
    percent: 6,
    message: "Preparing image",
    note: "Queued for processing.",
  },
  {
    afterMs: 1200,
    key: "extract",
    percent: 24,
    message: "Tracing boundary",
    note: "Finding service-area pixels.",
  },
  {
    afterMs: 4200,
    key: "labels",
    percent: 44,
    message: "Reading labels",
    note: "OCR and place hints are running.",
  },
  {
    afterMs: 9000,
    key: "georeference",
    percent: 68,
    message: "Fitting map",
    note: "Matching labels and roads.",
  },
  {
    afterMs: 18000,
    key: "export",
    percent: 88,
    message: "Finalizing export",
    note: "Writing GeoJSON and preview.",
  },
];

applyThemeMode(activeThemeMode, { persist: false });
resetProgressSteps();
renderProgressSteps();
historyEntries = loadHistoryEntries();
renderHistory();
updateRunButton();

themeModeButtons.forEach((button) => {
  button.addEventListener("click", () => applyThemeMode(button.dataset.themeMode));
});

settingsButton.addEventListener("click", () => {
  if (settingsDialog.open) {
    closeSettingsDialog();
  } else {
    openSettingsDialog();
  }
});

settingsCloseButton.addEventListener("click", closeSettingsDialog);

settingsDialog.addEventListener("click", (event) => {
  if (event.target === settingsDialog) closeSettingsDialog();
});

settingsDialog.addEventListener("close", () => {
  settingsButton.setAttribute("aria-expanded", "false");
});

systemThemeMedia?.addEventListener?.("change", () => {
  if (activeThemeMode === "system") applyThemeMode("system", { persist: false });
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  setSelectedFile(file);
});

document.addEventListener("paste", handleClipboardPaste);

runButton.addEventListener("click", (event) => {
  if (selectedFile || isRunButtonRunning()) return;
  event.preventDefault();
  imageInput.click();
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
    selectImageFile(file);
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    imageInput.click();
    return;
  }

  resetRun();
  startEstimatedProgress();
  setRunButtonState("running");

  try {
    const uploadFile = await prepareRunImage(selectedFile);
    const formData = new FormData(form);
    formData.set("image", uploadFile, uploadFile.name);
    markProgressStep("prepare", "running", "Uploading image.");
    setStatus("Uploading image", 8, "running", {
      step: "prepare",
      note: "Sending screenshot to the builder.",
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
      latestRunId = payload.id || null;
      connectEvents(payload.id);
    }
  } catch (error) {
    finishWithError(error.message);
  }
});

tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

imageModeButtons.forEach((button) => {
  button.addEventListener("click", () => setImageMode(button.dataset.imageMode));
});

copyButton.addEventListener("click", async () => {
  if (!latestGeojson) return;
  await navigator.clipboard.writeText(JSON.stringify(latestGeojson, null, 2));
  closeOutputMenu();
  setCopyCommandCopied(true);
  copyFeedbackTimeout = setTimeout(() => {
    setCopyCommandCopied(false);
  }, 1000);
});
downloadLink.addEventListener("click", closeOutputMenu);

brandButton.addEventListener("click", startNewRun);
brandHomeLink.addEventListener("click", (event) => {
  event.preventDefault();
  startNewRun();
});
reportButton.addEventListener("click", () => openReportDialog("failed"));
reportTrigger.addEventListener("click", () => {
  closeOutputMenu();
  openReportDialog("completed");
});
reportForm.addEventListener("submit", submitGenerationReport);
reportCloseButton.addEventListener("click", closeReportDialog);
reportCancelButton.addEventListener("click", closeReportDialog);

document.addEventListener("click", (event) => {
  if (outputMenu?.open && !outputMenu.contains(event.target)) closeOutputMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeOutputMenu();
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
    if (action === "rename") startHistoryRename(id);
    if (action === "download") downloadHistoryGeojson(id);
    if (action === "delete") deleteHistoryEntry(id);
    return;
  }

  const cancelRenameButton = event.target.closest("[data-history-rename-cancel]");
  if (cancelRenameButton) {
    event.preventDefault();
    cancelHistoryRename();
    return;
  }

  const loadButton = event.target.closest("[data-history-load]");
  if (loadButton) {
    const entry = historyEntries.find((item) => item.id === loadButton.dataset.historyLoad);
    if (entry) restoreHistoryEntry(entry);
  }
});

historyList.addEventListener("submit", (event) => {
  const form = event.target.closest("[data-history-rename]");
  if (!form) return;
  event.preventDefault();
  const input = form.querySelector(".history-rename-input");
  const title = normalizeHistoryTitle(input?.value || "");
  if (!title) {
    input?.focus();
    input?.select();
    return;
  }
  renameHistoryEntry(form.dataset.historyRename, title);
});

historyList.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!event.target.closest("[data-history-rename]")) return;
  event.preventDefault();
  cancelHistoryRename();
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

function loadThemeMode() {
  try {
    const storedMode = localStorage.getItem(THEME_STORAGE_KEY);
    return THEME_MODES.has(storedMode) ? storedMode : "system";
  } catch (error) {
    return "system";
  }
}

function applyThemeMode(mode, options = {}) {
  const nextMode = THEME_MODES.has(mode) ? mode : "system";
  const resolvedMode = resolvedThemeMode(nextMode);
  activeThemeMode = nextMode;
  document.documentElement.dataset.theme = nextMode;
  document.documentElement.dataset.resolvedTheme = resolvedMode;
  themeModeButtons.forEach((button) => {
    const isActive = button.dataset.themeMode === nextMode;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
  if (themeColorMeta) {
    themeColorMeta.content = resolvedMode === "dark" ? "#101512" : "#f8f6ee";
  }
  syncThemeIcons(resolvedMode);
  if (options.persist !== false) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextMode);
    } catch (error) {
      console.warn("Could not save theme preference", error);
    }
  }
  syncBoundaryMapTheme();
}

function syncThemeIcons(resolvedMode) {
  const icon = iconAssets[resolvedMode] || iconAssets.light;
  if (brandMark && brandMark.getAttribute("src") !== icon) {
    brandMark.src = icon;
  }
  iconLinks.forEach((link) => {
    if (link.getAttribute("href") !== icon) link.href = icon;
  });
}

function resolvedThemeMode(mode = activeThemeMode) {
  if (mode === "dark" || mode === "light") return mode;
  return systemThemeMedia?.matches ? "dark" : "light";
}

function boundaryMapStyleUrl() {
  return resolvedThemeMode() === "dark"
    ? "/static/openfreemap-dark.json"
    : "/static/openfreemap-boundary.json";
}

function syncBoundaryMapTheme() {
  if (!boundaryMap) return;
  const nextStyle = boundaryMapStyleUrl();
  if (currentBoundaryMapStyle === nextStyle) {
    updateBoundaryLayerPaint();
    return;
  }
  currentBoundaryMapStyle = nextStyle;
  const redraw = () => {
    if (latestGeojson) renderBoundaryMap(latestGeojson);
  };
  boundaryMap.once("styledata", () => window.setTimeout(redraw, 0));
  boundaryMap.setStyle(nextStyle);
}

function openSettingsDialog() {
  settingsButton.setAttribute("aria-expanded", "true");
  if (settingsDialog.showModal) {
    settingsDialog.showModal();
  } else {
    settingsDialog.setAttribute("open", "");
  }
}

function closeSettingsDialog() {
  if (settingsDialog.open && settingsDialog.close) {
    settingsDialog.close();
  } else {
    settingsDialog.removeAttribute("open");
    settingsButton.setAttribute("aria-expanded", "false");
  }
}

function handleClipboardPaste(event) {
  const file = clipboardImageFile(event.clipboardData);
  if (!file) return;
  if (isEditablePasteTarget(event.target) && event.clipboardData?.getData("text/plain")) return;

  event.preventDefault();
  if (isRunButtonRunning()) {
    setStatus("Run in progress", progressValue, latestRunStatus, {
      note: "Finish the active run before adding another screenshot.",
    });
    return;
  }
  selectImageFile(file);
}

function clipboardImageFile(clipboardData) {
  const items = Array.from(clipboardData?.items || []);
  const fileItem = items.find((item) => item.kind === "file" && isImageMime(item.type));
  const pastedFile = fileItem?.getAsFile?.()
    || Array.from(clipboardData?.files || []).find((file) => isImageMime(file.type));
  if (!pastedFile) return null;

  const type = pastedFile.type || fileItem?.type || "image/png";
  const name = pastedFile.name || clipboardImageName(type);
  return new File([pastedFile], name, {
    type,
    lastModified: pastedFile.lastModified || Date.now(),
  });
}

function isImageMime(type) {
  return typeof type === "string" && type.toLowerCase().startsWith("image/");
}

function clipboardImageName(type) {
  const extension = CLIPBOARD_IMAGE_EXTENSIONS.get(type) || "png";
  const timestamp = new Date().toISOString().replace(/\.\d{3}Z$/, "Z").replace(/[:.]/g, "-");
  return `clipboard-map-${timestamp}.${extension}`;
}

function isEditablePasteTarget(target) {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest("input, textarea, select, [contenteditable=''], [contenteditable='true']"));
}

function selectImageFile(file) {
  updateFileInput(file);
  setSelectedFile(file);
}

function updateFileInput(file) {
  if (!file || typeof DataTransfer === "undefined") return;
  try {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
  } catch (error) {
    console.warn("Could not mirror selected image into file input", error);
  }
}

function setSelectedFile(file) {
  selectedFile = file;
  if (!file) return;
  activeHistoryId = null;
  renamingHistoryId = null;
  latestRunId = null;
  latestRunError = null;
  latestRunEvents = [];
  latestRunStatus = "idle";
  latestRunSummary = null;
  progressValue = 0;
  stopEstimatedProgress();
  resetProgressSteps();
  stepStates.set("prepare", {
    state: "pending",
    message: "Ready to run.",
  });
  renderProgressSteps();
  clearGeneratedArtifacts();
  dropTitle.textContent = file.name;
  dropMeta.textContent = `${formatBytes(file.size)} · ${file.type || "image"}`;
  inputPreview.src = URL.createObjectURL(file);
  inputPreview.classList.add("ready");
  setImageMode("original");
  updateImagePane();
  dropZone.classList.add("has-file");
  workspaceTitle.textContent = file.name;
  updateRunButton();
  updateReportTrigger();
  hideFailureReport();
  setStatus("Image ready", 0, "idle", {
    note: "Review settings, then run the boundary export.",
  });
  renderHistory();
  activateTab("input");
}

async function prepareRunImage(file) {
  if (!isSvgFile(file)) return file;
  markProgressStep("prepare", "running", "Converting vector map.");
  setStatus("Rasterizing SVG map", 4, "running", {
    step: "prepare",
    note: "Converting vector upload before extraction.",
  });
  const canvas = await svgFileToCanvas(file);
  const blob = await canvasToBlob(canvas, "image/png");
  if (!blob) {
    throw new Error("Could not rasterize SVG upload.");
  }
  return new File([blob], `${fileBaseName(file.name)}.png`, {
    type: "image/png",
    lastModified: file.lastModified,
  });
}

function isSvgFile(file) {
  return file?.type === "image/svg+xml" || /\.svgz?$/i.test(file?.name || "");
}

function fileBaseName(filename) {
  return (filename || "map-upload").replace(/\.[^.]+$/, "") || "map-upload";
}

async function svgFileToCanvas(file) {
  const targetSize = svgRasterSize(await file.text());
  return imageFileToCanvas(file, targetSize);
}

function svgRasterSize(svgText) {
  const fallback = { width: 1600, height: 1000 };
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const svg = doc.documentElement;
  if (!svg || svg.nodeName.toLowerCase() !== "svg") return fallback;

  const viewBox = (svg.getAttribute("viewBox") || "")
    .trim()
    .split(/[\s,]+/)
    .map((value) => Number.parseFloat(value));
  const viewBoxWidth = viewBox.length === 4 && Number.isFinite(viewBox[2]) ? viewBox[2] : null;
  const viewBoxHeight = viewBox.length === 4 && Number.isFinite(viewBox[3]) ? viewBox[3] : null;
  const width = parseSvgLength(svg.getAttribute("width")) || viewBoxWidth || fallback.width;
  const height = parseSvgLength(svg.getAttribute("height")) || viewBoxHeight || fallback.height;
  const maxDimension = 4096;
  const scale = Math.min(1, maxDimension / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

function parseSvgLength(value) {
  if (!value) return null;
  const trimmed = value.trim();
  if (trimmed.endsWith("%")) return null;
  const parsed = Number.parseFloat(trimmed);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function imageFileToCanvas(file, targetSize = null) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = targetSize?.width || image.naturalWidth || 1;
      canvas.height = targetSize?.height || image.naturalHeight || 1;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(image.src);
      resolve(canvas);
    };
    image.onerror = reject;
    image.src = URL.createObjectURL(file);
  });
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

function connectEvents(runId) {
  if (eventSource) eventSource.close();
  latestRunId = runId;
  eventSource = new EventSource(`/api/runs/${runId}/events`);
  eventSource.addEventListener("update", async (message) => {
    const event = JSON.parse(message.data);
    applyEvent(event);
      if (event.status === "complete") {
        eventSource.close();
        stopEstimatedProgress();
        await loadArtifacts(runId);
        markAllProgressStepsDone();
        updateRunButton();
      }
    if (event.status === "error") {
      eventSource.close();
      stopEstimatedProgress();
      finishWithError(event.message);
    }
  });
  eventSource.onerror = () => {
    if (eventSource.readyState === EventSource.CLOSED) return;
  };
}

function applyInlineRun(status) {
  stopEstimatedProgress();
  latestRunId = status.id || latestRunId;
  latestRunEvents = status.events || latestRunEvents;
  latestRunStatus = "completed";
  latestRunSummary = status.summary || null;
  for (const event of status.events || []) {
    applyEvent(event);
  }
  const artifacts = status.artifacts || {};
  if (artifacts.overlay_data_url) {
    overlayPreview.src = artifacts.overlay_data_url;
    overlayPreview.classList.add("ready");
    setImageMode("overlay");
    updateImagePane();
  }
  if (artifacts.geojson_inline) {
    latestGeojson = artifacts.geojson_inline;
    geojsonPane.textContent = JSON.stringify(latestGeojson, null, 2);
    downloadLink.href = URL.createObjectURL(
      new Blob([JSON.stringify(latestGeojson, null, 2)], {
        type: "application/geo+json",
      }),
    );
    setGeojsonDownloadName(downloadLink, historyTitle(status));
    downloadLink.classList.remove("disabled");
    downloadLink.removeAttribute("aria-disabled");
    copyButton.disabled = false;
    showOutputActions();
    renderBoundary(latestGeojson);
  }
  if (status.summary) workspaceTitle.textContent = `${status.summary.city || status.city} boundary`;
  setStatus("Boundary export ready", 100, "complete");
  markAllProgressStepsDone();
  activateTab(artifacts.overlay_data_url ? "input" : "boundary");
  queueHistorySave({
    id: status.id,
    filename: status.filename,
    city: status.city,
    summary: status.summary,
    geojson: latestGeojson,
    overlaySrc: artifacts.overlay_data_url,
  });
  updateRunButton();
  updateReportTrigger();
}

function applyEvent(event) {
  if (event.status === "complete" || event.status === "error" || event.stage !== "queued") {
    stopEstimatedProgress();
  }
  latestRunEvents = [...latestRunEvents, event].slice(-20);
  const label = stageLabels[event.stage] || event.stage;
  const step = progressStepForEvent(event);
  if (event.status === "complete") {
    latestRunStatus = "completed";
    markAllProgressStepsDone();
  } else if (event.status === "error") {
    latestRunStatus = "failed";
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
  setProgressPanelState(status);
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

function setProgressPanelState(status) {
  const isRunning = status === "running" || status === "queued";
  const isError = status === "error";
  progressPanel.hidden = !(isRunning || isError);
  progressPanel.classList.toggle("is-running", isRunning);
}

function startEstimatedProgress() {
  stopEstimatedProgress();
  estimatedProgressStartedAt = performance.now();
  applyEstimatedProgress();
  estimatedProgressTimer = window.setInterval(applyEstimatedProgress, 350);
}

function stopEstimatedProgress() {
  if (!estimatedProgressTimer) return;
  window.clearInterval(estimatedProgressTimer);
  estimatedProgressTimer = null;
}

function applyEstimatedProgress() {
  if (latestRunStatus !== "running") {
    stopEstimatedProgress();
    return;
  }
  const elapsed = performance.now() - estimatedProgressStartedAt;
  let milestone = estimatedProgressStages[0];
  for (const stage of estimatedProgressStages) {
    if (elapsed >= stage.afterMs) milestone = stage;
  }
  markPreviousProgressStepsDone(milestone.key);
  markProgressStep(milestone.key, "running", milestone.note);
  setStatus(milestone.message, milestone.percent, "running", {
    step: milestone.key,
    note: milestone.note,
  });
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
        <li class="${escapeHtml(status.state)}" title="${escapeHtml(status.message)}">
          <span class="step-dot" aria-hidden="true"></span>
          <span>
            <b>${escapeHtml(step.shortTitle || step.title)}</b>
          </span>
        </li>
      `;
    })
    .join("");
}

function progressStepForEvent(event) {
  return {
    queued: "prepare",
    inspect: "prepare",
    extract: "extract",
    ocr: "labels",
    georeference: "georeference",
    export: "export",
    complete: "export",
  }[event.stage] || activeProgressStep;
}

function progressPercentForEvent(event) {
  const percent = Number(event.percent || 0);
  if (event.status === "complete" || event.stage === "complete") return 100;
  if (event.status === "error") return progressValue;
  if (event.stage === "queued") return 4;
  if (event.stage === "inspect") return 12;
  if (event.stage === "extract") return percent < 30 ? 28 : 38;
  if (event.stage === "ocr") return 52;
  if (event.stage === "georeference") {
    if (percent < 60) return 64;
    if (percent < 75) return 76;
    return 84;
  }
  if (event.stage === "export") return 92;
  return percent;
}

function humanProgressMessage(event) {
  if (event.stage === "queued") return "Run queued.";
  if (event.stage === "inspect") return "Preparing image.";
  if (event.stage === "extract") return event.percent < 30 ? "Tracing boundary." : "Area traced.";
  if (event.stage === "ocr") return "Reading labels.";
  if (event.stage === "georeference") return "Fitting map.";
  if (event.stage === "export") return "Finalizing export.";
  return event.message || stageLabels[event.stage] || "";
}

function humanProgressNote(event) {
  if (event.stage === "queued" || event.stage === "inspect") {
    return "The image is being prepared.";
  }
  if (event.stage === "ocr") {
    return "Reading labels for georeferencing.";
  }
  if (event.stage === "extract") {
    return "Separating the service area from the map.";
  }
  if (event.stage === "georeference") {
    return "Matching labels and roads.";
  }
  if (event.stage === "export") {
    return "Writing GeoJSON and preview.";
  }
  if (event.stage === "complete") {
    return "GeoJSON and preview are ready.";
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
  latestRunId = status.id || runId;
  latestRunStatus = status.status === "complete" ? "completed" : status.status || latestRunStatus;
  latestRunSummary = status.summary || null;
  const artifacts = status.artifacts || {};
  if (artifacts.input) {
    inputPreview.src = artifacts.input;
    inputPreview.classList.add("ready");
    updateImagePane();
  }
  if (artifacts.overlay) {
    overlayPreview.src = artifacts.overlay;
    overlayPreview.classList.add("ready");
    setImageMode("overlay");
    updateImagePane();
  }
  if (artifacts.geojson) {
    latestGeojson = await fetchJson(artifacts.geojson);
    geojsonPane.textContent = JSON.stringify(latestGeojson, null, 2);
    downloadLink.href = artifacts.geojson;
    setGeojsonDownloadName(downloadLink, historyTitle(status));
    downloadLink.classList.remove("disabled");
    downloadLink.removeAttribute("aria-disabled");
    copyButton.disabled = false;
    showOutputActions();
    renderBoundary(latestGeojson);
  }
  if (status.summary) workspaceTitle.textContent = `${status.summary.city || status.city} boundary`;
  activateTab(artifacts.overlay ? "input" : "boundary");
  queueHistorySave({
    id: status.id,
    filename: status.filename,
    city: status.city,
    summary: status.summary,
    geojson: latestGeojson,
    overlaySrc: artifacts.overlay,
  });
  updateReportTrigger();
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
    title: existing?.renamedAt ? existing.title : title,
    filename: payload.filename || selectedFile?.name || "Map screenshot",
    city: payload.summary?.city || payload.city || "Auto",
    createdAt: existing?.createdAt || Date.now(),
    starred: Boolean(existing?.starred),
    renamedAt: existing?.renamedAt || null,
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
        title: normalizeHistoryTitle(entry.title || entry.filename || "Generated boundary") || "Generated boundary",
        createdAt: Number(entry.createdAt) || Date.now(),
        starred: Boolean(entry.starred),
        renamedAt: Number(entry.renamedAt) || null,
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
  const isRenaming = entry.id === renamingHistoryId;
  const starred = entry.starred
    ? `<svg class="history-star" viewBox="0 0 20 20" aria-label="Starred"><path d="M10 1.8l2.4 5 5.4.8-3.9 3.8.9 5.4-4.8-2.6-4.8 2.6.9-5.4-3.9-3.8 5.4-.8L10 1.8z"></path></svg>`
    : "";
  const classes = [
    "history-item",
    entry.starred ? "starred" : "",
    entry.id === activeHistoryId ? "active" : "",
  ].filter(Boolean).join(" ");
  return `
    <li class="${escapeHtml(classes)}" data-history-id="${escapeHtml(entry.id)}">
      ${isRenaming ? renderHistoryRenameEntry(entry, thumb) : renderHistoryLoadEntry(entry, thumb, starred, detail)}
      ${isRenaming ? "" : `
        <details class="history-menu">
          <summary aria-label="Generation actions"><span class="kebab-icon" aria-hidden="true"><span></span><span></span><span></span></span></summary>
          <div class="history-menu-panel">
            <button type="button" data-history-action="rename">Rename</button>
            <button type="button" data-history-action="download">Download GeoJSON</button>
            <button type="button" data-history-action="star">${entry.starred ? "Unstar" : "Star"}</button>
            <button type="button" data-history-action="delete">Delete</button>
          </div>
        </details>
      `}
    </li>
  `;
}

function renderHistoryLoadEntry(entry, thumb, starred, detail) {
  return `
    <button class="history-main" type="button" data-history-load="${escapeHtml(entry.id)}">
      <span class="history-thumb">${thumb ? `<img src="${escapeHtml(thumb)}" alt="" />` : ""}</span>
      <span class="history-copy">
        <strong><span class="history-title-row">${starred}<span class="history-title-text">${escapeHtml(entry.title)}</span></span></strong>
        <span class="history-meta">${escapeHtml(formatHistoryTime(entry.createdAt))}</span>
        <span class="history-detail">${escapeHtml(detail)}</span>
      </span>
    </button>
  `;
}

function renderHistoryRenameEntry(entry, thumb) {
  return `
    <div class="history-main history-main-edit">
      <span class="history-thumb">${thumb ? `<img src="${escapeHtml(thumb)}" alt="" />` : ""}</span>
      <form class="history-rename-form" data-history-rename="${escapeHtml(entry.id)}">
        <input
          class="history-rename-input"
          type="text"
          aria-label="Run name"
          maxlength="${MAX_HISTORY_TITLE_LENGTH}"
          value="${escapeHtml(entry.title)}"
          autocomplete="off"
        />
        <span class="history-rename-actions">
          <button class="history-rename-save" type="submit">Save</button>
          <button class="history-rename-cancel" type="button" data-history-rename-cancel>Cancel</button>
        </span>
      </form>
    </div>
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
  renamingHistoryId = null;
  historyEntries = historyEntries.map((entry) => (
    entry.id === id ? { ...entry, starred: !entry.starred } : entry
  ));
  persistHistoryEntries();
  renderHistory();
  closeHistoryMenus();
}

function startHistoryRename(id) {
  renamingHistoryId = id;
  closeHistoryMenus();
  renderHistory();
  window.setTimeout(() => {
    const input = [...historyList.querySelectorAll("[data-history-rename]")]
      .find((form) => form.dataset.historyRename === id)
      ?.querySelector(".history-rename-input");
    input?.focus();
    input?.select();
  }, 0);
}

function cancelHistoryRename() {
  renamingHistoryId = null;
  renderHistory();
}

function renameHistoryEntry(id, title) {
  const nextTitle = normalizeHistoryTitle(title);
  if (!nextTitle) return;
  let renamed = false;
  historyEntries = historyEntries.map((entry) => {
    if (entry.id !== id) return entry;
    renamed = true;
    return {
      ...entry,
      title: nextTitle,
      renamedAt: Date.now(),
    };
  });
  if (!renamed) return;
  renamingHistoryId = null;
  persistHistoryEntries();
  if (activeHistoryId === id) {
    workspaceTitle.textContent = nextTitle;
    if (!downloadLink.classList.contains("disabled")) {
      setGeojsonDownloadName(downloadLink, nextTitle);
    }
  }
  renderHistory();
}

function downloadHistoryGeojson(id) {
  const entry = historyEntries.find((item) => item.id === id);
  if (!entry?.geojson) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(entry.geojson, null, 2)], {
      type: "application/geo+json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  setGeojsonDownloadName(link, entry.title);
  link.style.display = "none";
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
  closeHistoryMenus();
}

function deleteHistoryEntry(id) {
  if (activeHistoryId === id) activeHistoryId = null;
  if (renamingHistoryId === id) renamingHistoryId = null;
  historyEntries = historyEntries.filter((entry) => entry.id !== id);
  persistHistoryEntries();
  renderHistory();
}

function restoreHistoryEntry(entry) {
  closeHistoryMenus();
  renamingHistoryId = null;
  activeHistoryId = entry.id;
  selectedFile = null;
  latestRunId = entry.id;
  latestRunError = null;
  latestRunEvents = [];
  latestRunStatus = "completed";
  latestRunSummary = entry.summary || null;
  imageInput.value = "";
  clearGeneratedArtifacts();
  latestGeojson = entry.geojson;
  geojsonPane.textContent = JSON.stringify(latestGeojson, null, 2);
  downloadLink.href = URL.createObjectURL(
    new Blob([JSON.stringify(latestGeojson, null, 2)], {
      type: "application/geo+json",
    }),
  );
  setGeojsonDownloadName(downloadLink, entry.title);
  downloadLink.classList.remove("disabled");
  downloadLink.removeAttribute("aria-disabled");
  copyButton.disabled = false;
  showOutputActions();

  if (entry.inputImage) {
    inputPreview.src = entry.inputImage;
    inputPreview.classList.add("ready");
  } else {
    inputPreview.removeAttribute("src");
    inputPreview.classList.remove("ready");
  }

  if (entry.overlayImage) {
    overlayPreview.src = entry.overlayImage;
    overlayPreview.classList.add("ready");
  }
  setImageMode(entry.overlayImage ? "overlay" : "original");
  updateImagePane();

  dropTitle.textContent = EMPTY_DROP_TITLE;
  dropMeta.textContent = EMPTY_DROP_META;
  dropZone.classList.remove("has-file");
  workspaceTitle.textContent = entry.title;
  markAllProgressStepsDone();
  hideFailureReport();
  setStatus("Loaded from history", 100, "complete", {
    note: "Previous GeoJSON and previews are restored locally.",
  });
  activateTab("boundary");
  renderBoundary(latestGeojson);
  updateRunButton();
  updateReportTrigger();
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
  const panel = menu.querySelector(".history-menu-panel");
  const panelWidth = panel?.offsetWidth || 164;
  const panelHeight = panel?.offsetHeight || 120;
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

function normalizeHistoryTitle(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_HISTORY_TITLE_LENGTH);
}

function geojsonDownloadName(title) {
  const safeTitle = normalizeHistoryTitle(title)
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\.+$/g, "")
    .trim();
  return `${safeTitle || "boundary"}.geojson`;
}

function setGeojsonDownloadName(link, title) {
  link.download = geojsonDownloadName(title);
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
        context.fillStyle = cssVariable("--paper", "#f8f6ee");
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
      renderBoundaryMapWhenVisible(geojson);
    }
    return;
  }
  renderBoundarySvg(geojson);
}

function renderBoundaryMapWhenVisible(geojson) {
  window.requestAnimationFrame(() => {
    renderBoundaryMap(geojson);
    window.setTimeout(() => {
      if (boundaryMap && latestBoundaryBounds) {
        boundaryMap.resize();
        fitBoundaryMap(latestBoundaryBounds);
      } else {
        renderBoundaryMap(geojson);
      }
    }, 80);
  });
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
    currentBoundaryMapStyle = boundaryMapStyleUrl();
    boundaryMap = new maplibregl.Map({
      container: boundaryMapEl,
      style: currentBoundaryMapStyle,
      center: [(bounds.minLon + bounds.maxLon) / 2, (bounds.minLat + bounds.maxLat) / 2],
      zoom: 11,
      attributionControl: { compact: true },
    });
    boundaryMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  } else if (currentBoundaryMapStyle !== boundaryMapStyleUrl()) {
    currentBoundaryMapStyle = boundaryMapStyleUrl();
    boundaryMap.once("styledata", () => {
      if (latestGeojson) window.setTimeout(() => renderBoundaryMap(latestGeojson), 0);
    });
    boundaryMap.setStyle(currentBoundaryMapStyle);
  }

  const draw = () => {
    try {
      // map.loaded() can be false after the one-time load event.
      // Defer only if the style itself is unavailable.
      boundaryMap.resize();
      upsertBoundaryLayers(geojson);
      fitBoundaryMap(bounds);
    } catch (error) {
      if (isBoundaryStyleLoadingError(error)) {
        boundaryMap.once("styledata", () => window.setTimeout(draw, 0));
        return;
      }
      throw error;
    }
  };

  draw();
}

function isBoundaryStyleLoadingError(error) {
  const message = String(error?.message || error || "");
  return message.includes("Style is not done loading") || message.includes("style is not loaded");
}

function updateBoundaryLayerPaint() {
  if (!boundaryMap) return;
  try {
    const color = cssVariable("--green", "#0e6f5c");
    if (boundaryMap.getLayer(BOUNDARY_FILL_ID)) {
      boundaryMap.setPaintProperty(BOUNDARY_FILL_ID, "fill-color", color);
    }
    if (boundaryMap.getLayer(BOUNDARY_LINE_ID)) {
      boundaryMap.setPaintProperty(BOUNDARY_LINE_ID, "line-color", color);
    }
  } catch (error) {
    if (!isBoundaryStyleLoadingError(error)) throw error;
  }
}

function cssVariable(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
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
        "fill-color": cssVariable("--green", "#0e6f5c"),
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
        "line-color": cssVariable("--green", "#0e6f5c"),
        "line-width": 4,
        "line-opacity": 0.96,
      },
    });
  }
  updateBoundaryLayerPaint();
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
    <path d="${pathData}" fill="var(--svg-boundary-fill)" stroke="var(--green)" stroke-width="5" stroke-linejoin="round"></path>
    <circle cx="${offsetX}" cy="${offsetY + usedHeight}" r="5" fill="var(--gold)"></circle>
    <text x="${offsetX}" y="${offsetY + usedHeight + 28}" fill="var(--muted)" font-size="20">${minLon.toFixed(4)}, ${minLat.toFixed(4)}</text>
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
  if (name === "overlay") name = "input";
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  panes.forEach((pane) => pane.classList.toggle("active", pane.dataset.pane === name));
  if (name === "boundary" && latestGeojson) {
    renderBoundaryMapWhenVisible(latestGeojson);
  } else if (boundaryMap && latestBoundaryBounds) {
    window.setTimeout(() => {
      boundaryMap.resize();
      fitBoundaryMap(latestBoundaryBounds);
    }, 0);
  }
}

function setImageMode(mode) {
  activeImageMode = mode === "overlay" ? "overlay" : "original";
  updateImagePane();
}

function updateImagePane() {
  const hasOriginal = inputPreview.classList.contains("ready");
  const hasOverlay = overlayPreview.classList.contains("ready");
  if (activeImageMode === "overlay" && !hasOverlay) activeImageMode = "original";
  if (activeImageMode === "original" && !hasOriginal && hasOverlay) activeImageMode = "overlay";
  inputPane.classList.toggle("has-content", hasOriginal || hasOverlay);
  imageToggle.hidden = !(hasOriginal && hasOverlay);
  imageModeButtons.forEach((button) => {
    const mode = button.dataset.imageMode;
    button.classList.toggle("active", mode === activeImageMode);
    button.disabled = mode === "overlay" && !hasOverlay;
  });
  inputPreview.classList.toggle("preview-visible", hasOriginal && activeImageMode === "original");
  overlayPreview.classList.toggle("preview-visible", hasOverlay && activeImageMode === "overlay");
}

function startNewRun() {
  if (isRunButtonRunning()) return;
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  stopEstimatedProgress();
  selectedFile = null;
  activeHistoryId = null;
  latestRunId = null;
  latestRunError = null;
  latestRunEvents = [];
  latestRunStatus = "idle";
  latestRunSummary = null;
  imageInput.value = "";
  progressValue = 0;
  resetProgressSteps();
  clearGeneratedArtifacts();
  inputPreview.removeAttribute("src");
  inputPreview.classList.remove("ready");
  updateImagePane();
  dropZone.classList.remove("has-file");
  dropTitle.textContent = EMPTY_DROP_TITLE;
  dropMeta.textContent = EMPTY_DROP_META;
  workspaceTitle.textContent = "Ready for a screenshot";
  setCopyCommandCopied(false);
  hideFailureReport();
  setStatus("Idle", 0, "idle", {
    note: "Add a map screenshot to start.",
  });
  renderHistory();
  updateRunButton();
  updateReportTrigger();
  activateTab("input");
}

function resetRun() {
  stopEstimatedProgress();
  progressValue = 0;
  latestRunId = null;
  latestRunError = null;
  latestRunEvents = [];
  latestRunStatus = "running";
  latestRunSummary = null;
  hideFailureReport();
  resetProgressSteps();
  markProgressStep("prepare", "running", "Preparing image.");
  clearGeneratedArtifacts();
  setStatus("Preparing image", 2, "running", {
    step: "prepare",
    note: "Queued for processing.",
  });
}

function clearGeneratedArtifacts() {
  latestGeojson = null;
  overlayPreview.removeAttribute("src");
  overlayPreview.classList.remove("ready");
  setImageMode("original");
  updateImagePane();
  boundaryMapEl.classList.remove("ready");
  boundarySvg.innerHTML = "";
  boundarySvg.classList.remove("ready");
  boundaryEmpty.hidden = false;
  latestBoundaryBounds = null;
  if (boundaryMap?.getSource(BOUNDARY_SOURCE_ID)) {
    boundaryMap.getSource(BOUNDARY_SOURCE_ID).setData({ type: "FeatureCollection", features: [] });
  }
  geojsonPane.textContent = "{}";
  hideOutputActions();
}

function finishWithError(message) {
  stopEstimatedProgress();
  latestRunError = message || "Generation failed.";
  latestRunStatus = "failed";
  markProgressStep(activeProgressStep || "georeference", "error", message);
  setStatus(message, progressValue, "error", {
    note: "The run stopped before a reliable boundary could be exported.",
  });
  showFailureReport();
  updateRunButton();
}

function showFailureReport() {
  if (!selectedFile) return;
  reportPanel.hidden = false;
  reportButton.hidden = false;
  reportButton.disabled = false;
  reportButton.textContent = "Report issue";
  reportLink.hidden = true;
  reportLink.href = "#";
  reportText.textContent = "Tell us what went wrong. Your uploaded screenshot will be public in the GitHub issue.";
}

function hideFailureReport() {
  reportPanel.hidden = true;
  reportButton.disabled = false;
  reportButton.hidden = false;
  reportButton.textContent = "Report issue";
  reportLink.hidden = true;
  reportLink.href = "#";
  reportText.textContent = "Tell us what went wrong. Your uploaded screenshot will be public in the GitHub issue.";
}

function openReportDialog(status) {
  if (!canReportGeneration(status)) return;
  activeReportStatus = status;
  reportForm.reset();
  reportIssueType.value = status === "failed" ? "Generation failed" : "Boundary shape is wrong";
  reportFormStatus.textContent = "";
  reportFormStatus.className = "report-form-status";
  reportIssueLink.hidden = true;
  reportIssueLink.href = "#";
  reportSubmitButton.disabled = false;
  reportSubmitButton.textContent = "Create Issue";
  if (reportDialog.showModal) {
    reportDialog.showModal();
  } else {
    reportDialog.setAttribute("open", "");
  }
}

function closeReportDialog() {
  if (reportDialog.open && reportDialog.close) {
    reportDialog.close();
  } else {
    reportDialog.removeAttribute("open");
  }
}

async function submitGenerationReport(event) {
  event.preventDefault();
  if (!canReportGeneration(activeReportStatus)) return;
  reportSubmitButton.disabled = true;
  reportSubmitButton.textContent = "Creating";
  reportFormStatus.className = "report-form-status";
  reportFormStatus.textContent = "Uploading the screenshot and run details to a public GitHub issue.";
  try {
    const reportImage = await reportImageFile();
    const formData = new FormData();
    formData.set("image", reportImage, reportImage.name || reportFilename());
    formData.set("issue_type", reportIssueType.value);
    formData.set("generation_status", activeReportStatus);
    formData.set("user_note", reportUserNote.value.trim());
    formData.set("error", latestRunError || reportStatusMessage(activeReportStatus));
    formData.set("run_id", latestRunId || "");
    formData.set("events", JSON.stringify(latestRunEvents));
    formData.set("settings", JSON.stringify(collectRunSettings()));
    formData.set("summary", JSON.stringify(latestRunSummary || {}));
    formData.set("user_agent", navigator.userAgent);
    formData.set("page_url", window.location.href);
    const response = await fetch("/api/reports", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Report failed.");
    reportFormStatus.textContent = "Issue created. The screenshot is now public in GitHub for debugging.";
    reportFormStatus.classList.add("success");
    reportIssueLink.href = payload.issue_url;
    reportIssueLink.hidden = false;
    if (activeReportStatus === "failed") {
      reportText.textContent = "Issue created. The screenshot is now public in GitHub for debugging.";
      reportLink.href = payload.issue_url;
      reportLink.hidden = false;
      reportButton.hidden = true;
    }
    reportSubmitButton.textContent = "Created";
  } catch (error) {
    reportFormStatus.textContent = error.message || "Could not create the GitHub issue.";
    reportFormStatus.classList.add("error");
    reportSubmitButton.disabled = false;
    reportSubmitButton.textContent = "Try Again";
    if (activeReportStatus === "failed") {
      reportText.textContent = reportFormStatus.textContent;
      reportButton.disabled = false;
      reportButton.textContent = "Try Again";
    }
  }
}

function canReportGeneration(status) {
  if (status === "failed") return Boolean(selectedFile);
  return Boolean(latestGeojson && reportImageSource());
}

function updateReportTrigger() {
  const canReport = canReportGeneration("completed");
  reportTrigger.disabled = !canReport;
  reportTrigger.hidden = !canReport;
}

function showOutputActions() {
  if (!latestGeojson) {
    hideOutputActions();
    return;
  }
  outputMenu.hidden = false;
  downloadLink.classList.remove("disabled");
  downloadLink.removeAttribute("aria-disabled");
  copyButton.disabled = false;
  setCopyCommandCopied(false);
  updateReportTrigger();
}

function hideOutputActions() {
  closeOutputMenu();
  outputMenu.hidden = true;
  downloadLink.href = "#";
  downloadLink.classList.add("disabled");
  downloadLink.setAttribute("aria-disabled", "true");
  copyButton.disabled = true;
  setCopyCommandCopied(false);
  updateReportTrigger();
}

function closeOutputMenu() {
  if (outputMenu) outputMenu.open = false;
}

function setCopyCommandCopied(copied) {
  if (copyFeedbackTimeout) {
    clearTimeout(copyFeedbackTimeout);
    copyFeedbackTimeout = null;
  }
  copyButton.innerHTML = copied
    ? "<span>Copied</span><small>Ready to paste</small>"
    : COPY_BUTTON_IDLE_HTML;
}

async function reportImageFile() {
  const source = reportImageSource();
  if (!source) throw new Error("No screenshot is available to report.");
  if (source instanceof File) return prepareReportImage(source);
  if (source.startsWith("data:")) {
    return prepareReportImage(dataUrlToFile(source, reportFilename()));
  }
  const response = await fetch(source);
  if (!response.ok) throw new Error("Could not read the screenshot for the report.");
  const blob = await response.blob();
  return prepareReportImage(new File([blob], reportFilename(), { type: blob.type || "image/png" }));
}

function reportImageSource() {
  if (selectedFile) return selectedFile;
  const src = inputPreview.getAttribute("src");
  return src || null;
}

function reportFilename() {
  const base = selectedFile?.name || workspaceTitle.textContent || "generation";
  const safe = String(base)
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\.+$/g, "")
    .trim();
  return `${safe || "generation"}.png`;
}

function reportStatusMessage(status) {
  if (status === "failed") return latestRunError || "Generation failed.";
  return "Completed generation was reported by the user.";
}

function dataUrlToFile(dataUrl, filename) {
  const [header, data] = dataUrl.split(",");
  const mime = /data:([^;]+)/.exec(header)?.[1] || "image/png";
  const binary = atob(data || "");
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new File([bytes], filename, { type: mime });
}

async function prepareReportImage(file) {
  if (file.size <= 7_500_000) return file;
  const canvas = await imageFileToCanvas(file);
  const maxSize = 2200;
  const scale = Math.min(1, maxSize / Math.max(canvas.width, canvas.height));
  const output = document.createElement("canvas");
  output.width = Math.max(1, Math.round(canvas.width * scale));
  output.height = Math.max(1, Math.round(canvas.height * scale));
  const context = output.getContext("2d");
  context.drawImage(canvas, 0, 0, output.width, output.height);
  const blob = await new Promise((resolve) => output.toBlob(resolve, "image/jpeg", 0.82));
  if (!blob) return file;
  return new File([blob], `${file.name.replace(/\.[^.]+$/, "") || "generation"}-report.jpg`, {
    type: "image/jpeg",
  });
}

function collectRunSettings() {
  return Object.fromEntries(
    [...new FormData(form).entries()]
      .filter(([key]) => key !== "image")
      .map(([key, value]) => [key, String(value)]),
  );
}

function updateRunButton() {
  const isRunning = isRunButtonRunning();
  brandButton.disabled = isRunning;
  brandHomeLink.setAttribute("aria-disabled", String(isRunning));
  if (isRunning) {
    brandHomeLink.setAttribute("tabindex", "-1");
  } else {
    brandHomeLink.removeAttribute("tabindex");
  }
  if (isRunning) return;
  setRunButtonState(selectedFile ? "ready" : "empty");
}

function setRunButtonState(state) {
  const nextState = Object.prototype.hasOwnProperty.call(RUN_BUTTON_LABELS, state) ? state : "empty";
  runButton.dataset.state = nextState;
  runButton.disabled = nextState === "running";
  runButtonLabel.textContent = RUN_BUTTON_LABELS[nextState];
  if (nextState === "empty") {
    runButton.title = "Choose a map screenshot";
  } else {
    runButton.removeAttribute("title");
  }
}

function isRunButtonRunning() {
  return runButton.dataset.state === "running" && (latestRunStatus === "running" || latestRunStatus === "queued");
}

function hasResettableWorkspaceState() {
  return Boolean(
    selectedFile ||
    latestGeojson ||
    activeHistoryId ||
    latestRunId ||
    latestRunStatus !== "idle" ||
    inputPreview.classList.contains("ready") ||
    overlayPreview.classList.contains("ready"),
  );
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
