import "./style.css";
import { loadCoords, type PointData } from "./api";
import { ClusterScene, CATEGORY_COLORS } from "./scene";

const root = document.getElementById("app")!;

root.innerHTML = `
  <div class="relative w-full h-screen overflow-hidden">
    <!-- Header overlay -->
    <header class="absolute top-0 left-0 right-0 p-4 z-10 flex justify-between items-start pointer-events-none">
      <div class="pointer-events-auto">
        <h1 class="text-xl font-semibold tracking-tight">OpenCLIP Cluster Viz</h1>
        <p class="text-xs text-neutral-400 mt-1 max-w-md">
          Every photo's 1024-dim embedding projected to 3D with UMAP.
          Drag to orbit, scroll to zoom, click any point to open it.
        </p>
      </div>
      <div id="legend" class="pointer-events-auto bg-neutral-900/80 backdrop-blur border border-neutral-800
                              rounded-lg p-3 text-xs space-y-1.5 min-w-[180px]"></div>
    </header>

    <!-- Caveat in the bottom-left -->
    <div class="absolute bottom-3 left-4 z-10 text-xs text-neutral-500 max-w-md pointer-events-none">
      3D positions are a UMAP approximation - clusters are honest, exact distances are not.
      For precise similarity, use the search app.
    </div>

    <!-- The Three.js canvas mounts here -->
    <div id="canvas" class="absolute inset-0"></div>

    <!-- Hover tooltip -->
    <div id="tooltip"
         class="hidden pointer-events-none absolute z-20 px-2 py-1 rounded bg-neutral-900/90
                border border-neutral-700 text-xs text-neutral-200 shadow-lg"
         style="left: 0; top: 0;"></div>

    <!-- Loading / error overlay -->
    <div id="overlay" class="absolute inset-0 z-30 flex items-center justify-center bg-neutral-950">
      <p class="text-neutral-400 text-sm">Loading 1024-dim embeddings…</p>
    </div>

    <!-- Side panel -->
    <aside id="panel"
           class="absolute top-0 right-0 h-full w-full sm:w-96 bg-neutral-900/95 backdrop-blur
                  border-l border-neutral-800 transform translate-x-full transition-transform
                  duration-300 z-20 flex flex-col">
      <div class="flex items-start justify-between p-4 border-b border-neutral-800">
        <div>
          <h2 id="panel-title" class="font-medium text-sm break-all"></h2>
          <p id="panel-meta" class="text-xs text-neutral-400 mt-1"></p>
        </div>
        <button id="panel-close" type="button" aria-label="Close"
                class="shrink-0 w-7 h-7 rounded-full bg-neutral-800 hover:bg-red-500
                       border border-neutral-700 text-neutral-100 text-xs
                       flex items-center justify-center transition-colors">
          ✕
        </button>
      </div>
      <div class="flex-1 overflow-y-auto p-4">
        <img id="panel-image" class="w-full rounded-lg border border-neutral-800" alt="" />
      </div>
    </aside>
  </div>
`;

const canvasEl = document.getElementById("canvas") as HTMLDivElement;
const tooltipEl = document.getElementById("tooltip") as HTMLDivElement;
const overlayEl = document.getElementById("overlay") as HTMLDivElement;
const legendEl = document.getElementById("legend") as HTMLDivElement;
const panelEl = document.getElementById("panel") as HTMLElement;
const panelTitle = document.getElementById("panel-title") as HTMLHeadingElement;
const panelMeta = document.getElementById("panel-meta") as HTMLParagraphElement;
const panelImage = document.getElementById("panel-image") as HTMLImageElement;
const panelClose = document.getElementById("panel-close") as HTMLButtonElement;

// Build the legend swatches from the colour map.
legendEl.innerHTML = Object.entries(CATEGORY_COLORS)
  .map(
    ([name, colour]) => `
      <div class="flex items-center gap-2">
        <span class="inline-block w-3 h-3 rounded-full" style="background:${colour}"></span>
        <span class="text-neutral-200">${name}</span>
      </div>
    `
  )
  .join("");

function showTooltip(text: string, x: number, y: number) {
  tooltipEl.textContent = text;
  tooltipEl.classList.remove("hidden");
  // Offset a few px from the cursor.
  tooltipEl.style.transform = `translate(${x + 12}px, ${y + 12}px)`;
}

function hideTooltip() {
  tooltipEl.classList.add("hidden");
}

function openPanel(point: PointData) {
  panelTitle.textContent = point.filename;
  const bits: string[] = [];
  if (point.category) bits.push(point.category);
  if (point.confidence != null) bits.push(`confidence ${point.confidence.toFixed(3)}`);
  panelMeta.textContent = bits.join(" · ") || "no metadata";
  panelImage.src = `/images/${encodeURIComponent(point.filename)}`;
  panelImage.alt = point.filename;
  panelEl.classList.remove("translate-x-full");
}

function closePanel() {
  panelEl.classList.add("translate-x-full");
}

panelClose.addEventListener("click", closePanel);

async function bootstrap() {
  try {
    const data = await loadCoords();
    if (data.length === 0) {
      overlayEl.innerHTML =
        `<p class="text-neutral-400 text-sm">coords.json is empty. Run <code>uv run generate_coords.py</code> first.</p>`;
      return;
    }
    const scene = new ClusterScene(canvasEl, {
      onHover: (point, x, y) => {
        if (point) showTooltip(point.filename, x, y);
        else hideTooltip();
      },
      onClick: (point) => openPanel(point),
    });
    scene.load(data);
    overlayEl.style.display = "none";
    console.log(`Loaded ${data.length} embedding points.`);
  } catch (err) {
    overlayEl.innerHTML =
      `<div class="text-center text-sm text-red-400 max-w-md p-6">
         <p class="font-medium">Could not load coordinates</p>
         <p class="mt-2 text-neutral-400">${(err as Error).message}</p>
       </div>`;
  }
}

bootstrap();
