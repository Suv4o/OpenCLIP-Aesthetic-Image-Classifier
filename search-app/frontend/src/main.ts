import "./style.css";
import { searchByText, searchByImage, type SearchHit } from "./api";

const root = document.getElementById("app")!;

root.innerHTML = `
  <main class="max-w-6xl mx-auto px-4 py-10 space-y-8">
    <header class="space-y-2">
      <h1 class="text-3xl font-semibold tracking-tight">OpenCLIP Photo Search</h1>
      <p class="text-neutral-400 text-sm">
        Search a personal landscape library by description, or drop in an image to find visually similar shots.
        Embeddings: <span class="font-mono text-neutral-300">ViT-H-14 / dfn5b</span>, 1024-dim, served from pgvector.
      </p>
    </header>

    <section class="space-y-3">
      <label class="block text-sm font-medium text-neutral-300" for="q">Search by text</label>
      <form id="text-form" class="flex gap-2">
        <div class="relative flex-1">
          <input
            id="q"
            name="q"
            type="text"
            placeholder='e.g. "foggy mountain at dawn", "neon city at night"'
            class="w-full rounded-lg bg-neutral-900 border border-neutral-800 pl-4 pr-11 py-3 text-base
                   focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/40"
            autocomplete="off"
          />
          <button
            id="clear-text"
            type="button"
            aria-label="Clear search"
            class="hidden absolute top-1/2 right-2 -translate-y-1/2 w-7 h-7 rounded-full
                   bg-neutral-800 hover:bg-red-500 border border-neutral-700 text-neutral-100 text-xs
                   flex items-center justify-center transition-colors"
          >
            ✕
          </button>
        </div>
        <button
          type="submit"
          class="rounded-lg bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-medium px-5 py-3
                 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Search
        </button>
      </form>
    </section>

    <section class="space-y-3">
      <label class="block text-sm font-medium text-neutral-300">Search by image</label>
      <div
        id="drop"
        class="rounded-lg border-2 border-dashed border-neutral-800 hover:border-emerald-500/50
               transition-colors py-10 px-6 text-center cursor-pointer bg-neutral-900/50"
      >
        <p class="text-neutral-400">
          Drag and drop a photo here, or
          <button id="pick" type="button" class="text-emerald-400 underline underline-offset-2 hover:text-emerald-300">
            choose a file
          </button>.
        </p>
        <p class="text-xs text-neutral-500 mt-1">JPG, PNG, WebP. The file stays on your machine — only the embedding is sent.</p>
        <input id="file" type="file" accept="image/*" class="hidden" />
        <div id="preview-wrap" class="hidden mt-6 flex justify-center">
          <div class="relative inline-block">
            <img id="preview" class="max-h-48 rounded-lg border border-neutral-800" alt="dropped preview" />
            <button
              id="clear-image"
              type="button"
              aria-label="Clear image"
              class="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-neutral-800 hover:bg-red-500
                     border border-neutral-700 text-neutral-100 text-xs flex items-center justify-center
                     transition-colors shadow-lg"
            >
              ✕
            </button>
          </div>
        </div>
      </div>
    </section>

    <section>
      <div id="status" class="text-sm text-neutral-400 mb-4"></div>
      <div id="results" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
    </section>
  </main>
`;

const textForm = document.getElementById("text-form") as HTMLFormElement;
const queryInput = document.getElementById("q") as HTMLInputElement;
const dropZone = document.getElementById("drop") as HTMLDivElement;
const pickBtn = document.getElementById("pick") as HTMLButtonElement;
const fileInput = document.getElementById("file") as HTMLInputElement;
const previewWrap = document.getElementById("preview-wrap") as HTMLDivElement;
const previewImg = document.getElementById("preview") as HTMLImageElement;
const clearImageBtn = document.getElementById("clear-image") as HTMLButtonElement;
const clearTextBtn = document.getElementById("clear-text") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const resultsEl = document.getElementById("results") as HTMLDivElement;

function setStatus(text: string) {
  statusEl.textContent = text;
}

function clearImage() {
  if (previewImg.src && previewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(previewImg.src);
  }
  previewImg.removeAttribute("src");
  previewWrap.classList.add("hidden");
  fileInput.value = "";
}

function clearResults() {
  resultsEl.innerHTML = "";
  setStatus("");
}

function syncTextClearVisibility() {
  clearTextBtn.classList.toggle("hidden", queryInput.value.length === 0);
}

function clearTextInput() {
  queryInput.value = "";
  syncTextClearVisibility();
}

function renderResults(hits: SearchHit[]) {
  if (hits.length === 0) {
    resultsEl.innerHTML = `<p class="col-span-full text-neutral-500 text-sm">No matches.</p>`;
    return;
  }
  resultsEl.innerHTML = hits
    .map(
      (hit) => `
      <a href="${hit.url}" target="_blank" rel="noopener"
         class="group relative block aspect-square overflow-hidden rounded-lg border border-neutral-800
                hover:border-emerald-500/60 transition-colors bg-neutral-900">
        <img src="${hit.url}" alt="${hit.filename}" loading="lazy"
             class="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105" />
        <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent
                    p-2 text-xs space-y-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <div class="font-medium truncate">${hit.filename}</div>
          ${hit.category ? `<div class="text-neutral-300">${hit.category}</div>` : ""}
        </div>
      </a>`
    )
    .join("");
}

// Requesting more than we'll likely show; the backend filters down by similarity.
const REQUEST_LIMIT = 24;

function formatResultStatus(hits: number, query: string): string {
  if (hits === 0) return `No matches for "${query}".`;
  if (hits === 1) return `Only the closest match for "${query}" — no strong matches found.`;
  return `${hits} matches for "${query}".`;
}

async function runTextSearch(query: string) {
  if (!query.trim()) return;
  clearImage();
  setStatus(`Searching for "${query}"…`);
  resultsEl.innerHTML = "";
  try {
    const hits = await searchByText(query, REQUEST_LIMIT);
    setStatus(formatResultStatus(hits.length, query));
    renderResults(hits);
  } catch (err) {
    setStatus(`Error: ${(err as Error).message}`);
  }
}

async function runImageSearch(file: File) {
  clearTextInput();
  setStatus(`Searching by "${file.name}"…`);
  resultsEl.innerHTML = "";
  if (previewImg.src && previewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(previewImg.src);
  }
  previewImg.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  try {
    const hits = await searchByImage(file, REQUEST_LIMIT);
    setStatus(formatResultStatus(hits.length, file.name));
    renderResults(hits);
  } catch (err) {
    setStatus(`Error: ${(err as Error).message}`);
  }
}

textForm.addEventListener("submit", (e) => {
  e.preventDefault();
  runTextSearch(queryInput.value);
});

pickBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) runImageSearch(file);
});

clearImageBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearImage();
  clearResults();
});

clearTextBtn.addEventListener("click", () => {
  clearTextInput();
  clearResults();
  queryInput.focus();
});

queryInput.addEventListener("input", syncTextClearVisibility);

// Drag and drop on the whole drop zone.
["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("border-emerald-500/70", "bg-neutral-900");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-emerald-500/70", "bg-neutral-900");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith("image/")) runImageSearch(file);
});
